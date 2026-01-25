# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Nicole (TN) library.
#
# Nicole (TN) is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Nicole (TN) is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Nicole (TN). If not, see <https://www.gnu.org/licenses/>.


from __future__ import annotations

"""Utilities for constructing canonical identity and fusion tensors.

This module provides helpers that build symmetry-aware tensors commonly used in
tensor network algorithms: a two-leg identity and a three-leg fusion isometry.
Both routines respect the block structure defined by Nicole indices and ensure
charge conservation across all generated blocks.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .index import Index, combine_indices
from .symmetry.base import AbelianGroup
from .symmetry.product import ProductGroup
from .tensor import Tensor
from .typing import Charge, Direction


def identity(index: Index, *, dtype=np.float64, itags: Optional[Tuple[str, str]] = None) -> Tensor:
    """Return a 2-leg identity tensor between `index` and its conjugate leg.

    Parameters
    ----------
    index:
        The index to be paired with its flipped counterpart.
    dtype:
        Data type for the identity matrices stored in each block.
    itags:
        Optional tuple of tags for the two tensor indices. Defaults to `("_init_", "_init_")`.

    Returns
    -------
    Tensor
        Tensor with two indices (original and flipped) whose blocks encode the
        identity matrices for each sector.
    """

    # Prepare the left leg and its flipped partner.
    left = index
    right = index.flip()
    if itags is None:
        itags = ("_init_", "_init_")

    blocks: Dict[tuple[Charge, Charge], np.ndarray] = {}
    # Populate diagonal blocks keyed by identical charges.
    for sector in left.sectors:
        q = sector.charge
        dim = sector.dim
        blocks[(q, q)] = np.eye(dim, dtype=dtype)

    return Tensor(indices=(left, right), itags=itags, data=blocks, dtype=dtype)


def isometry(
    first: Index,
    second: Index,
    *, # keyword-only parameters
    dtype=np.float64,
    itags: Optional[Tuple[str, str, str]] = None,
    fused_direction: Optional[Direction] = None,
) -> Tensor:
    """Return a 3-leg tensor that fuses ``first ⊗ second`` into a fused leg.

    Parameters
    ----------
    first, second:
        Input indices to be fused. They must share a symmetry group.
    dtype:
        Data type for the emitted fusion blocks.
    itags:
        Optional tuple of tags for the three tensor indices. Defaults to `("_init_", "_init_", "_init_")`.
    fused_direction:
        Optional direction for the fused leg. Defaults to the dual of `first`.

    Returns
    -------
    Tensor
        Three-leg tensor whose third index represents the fusion of the first two.

    Raises
    ------
    ValueError
        If the incoming indices do not belong to the same symmetry group.
    NotImplementedError
        When attempting to fuse non-Abelian indices (not yet supported).
    RuntimeError
        If internal bookkeeping detects a fusion shape mismatch.
    """
    if first.group != second.group:
        raise ValueError("Both indices must share the same symmetry group")
    group = first.group
    if not isinstance(group, (AbelianGroup, ProductGroup)):
        raise NotImplementedError("Fusion currently supports only Abelian groups")

    # Determine orientation of the fused leg; default to the dual of `first`.
    default_dir = first.direction.reverse()
    direction = fused_direction if fused_direction is not None else default_dir
    fused = combine_indices(direction, first, second)
    if itags is None:
        itags = ("_init_", "_init_", "_init_")

    # Track how many columns have been written per fused charge.
    offsets: Dict[Charge, int] = {sector.charge: 0 for sector in fused.sectors}
    blocks: Dict[tuple[Charge, Charge, Charge], np.ndarray] = {}
    dim_fused_map = fused.sector_dim_map()

    for sa in first.sectors:
        qa = sa.charge
        da = sa.dim
        for sb in second.sectors:
            qb = sb.charge
            db = sb.dim
            
            # Compute fused charge in a direction-aware way (matching combine_indices logic)
            # Charges being fused (IN) contribute as-is, already fused (OUT) contribute inverse
            contrib_a = qa if first.direction == Direction.IN else group.inverse(qa)
            contrib_b = qb if second.direction == Direction.IN else group.inverse(qb)
            total_contrib = group.fuse(contrib_a, contrib_b)
            # Fused index: inverse when direction is IN
            qf = group.inverse(total_contrib) if direction == Direction.IN else total_contrib
            
            fused_dim = dim_fused_map[qf]
            offset = offsets[qf]
            arr = np.zeros((da, db, fused_dim), dtype=dtype)
            # Fill a set of identity matrices at appropriate column offsets.
            for i in range(da):
                base = offset + i * db
                cols = slice(base, base + db)
                arr[i, :, cols] = np.eye(db, dtype=dtype)
            blocks[(qa, qb, qf)] = arr
            offsets[qf] = offset + da * db

    # Ensure each fused sector is completely populated.
    for q, offset in offsets.items():
        if offset != dim_fused_map[q]:
            raise RuntimeError("Fusion tensor construction mismatch")

    return Tensor(indices=(first, second, fused), itags=itags, data=blocks, dtype=dtype)


def isometry_n(
    indices: Sequence[Index],
    *,
    dtype=np.float64,
    itags: Optional[Sequence[str]] = None,
    direction: Direction = Direction.OUT,
) -> Tensor:
    """Return an (n+1)-leg tensor that fuses n indices into a single fused leg.

    This function constructs an n-to-1 isometry by sequentially applying 2-to-1
    isometries. Indices are fused in order of increasing dimension to minimize
    intermediate tensor sizes and computational complexity.

    The resulting isometry has n indices with directions opposite to the input
    indices (to enable natural contraction), plus one fused index whose direction
    is specified by the `direction` parameter.

    Parameters
    ----------
    indices:
        Sequence of indices to be fused. Must contain at least 2 indices, and all
        indices must share the same symmetry group.
    dtype:
        Data type for the emitted fusion blocks.
    itags:
        Optional sequence of tags for all tensor indices (n unfused + 1 fused).
        Length must be `len(indices) + 1`. Defaults to all `"_init_"`.
    direction:
        Direction for the fused output index. Defaults to `Direction.OUT`.

    Returns
    -------
    Tensor
        Tensor with (n+1) indices: the first n indices have opposite directions
        to the input indices, and the last index is the fused index with the
        specified direction.

    Raises
    ------
    ValueError
        If fewer than 2 indices are provided, if indices don't share the same
        symmetry group, or if `itags` length doesn't match `len(indices) + 1`.
    NotImplementedError
        When attempting to fuse non-Abelian indices (not yet supported).
    """
    # Avoid circular import by importing contract here
    from .contract import contract

    # Step 1: Input Validation
    n = len(indices)
    if n < 2:
        raise ValueError(f"Need at least 2 indices to fuse, got {n}")

    # Check all indices share the same symmetry group
    group = indices[0].group
    for i, idx in enumerate(indices[1:], start=1):
        if idx.group != group:
            raise ValueError(
                f"All indices must share the same symmetry group. "
                f"Index 0 has group {group}, but index {i} has group {idx.group}"
            )

    # Check group is Abelian or ProductGroup
    if not isinstance(group, (AbelianGroup, ProductGroup)):
        raise NotImplementedError("Fusion currently supports only Abelian groups")

    # Validate itags length if provided
    if itags is not None:
        if len(itags) != n + 1:
            raise ValueError(
                f"itags must have length {n + 1} (n indices + 1 fused), got {len(itags)}"
            )

    # Step 2: Flip All Input Indices
    # Flip directions to get opposite directions (for natural contraction)
    flipped_indices = [idx.flip() for idx in indices]
    
    # Step 3: Sort Indices by Dimension
    # Create list of (index, original_position) and sort by dimension
    indexed_indices = [(idx, i) for i, idx in enumerate(flipped_indices)]
    sorted_indices = sorted(indexed_indices, key=lambda x: x[0].dim)
    
    # Extract sorted indices and track their original positions
    sorted_idx_list = [idx for idx, _ in sorted_indices]
    original_positions = [orig_pos for _, orig_pos in sorted_indices]
    
    # Create inverse permutation to restore original order later
    inverse_perm = [0] * n
    for sorted_pos, orig_pos in enumerate(original_positions):
        inverse_perm[orig_pos] = sorted_pos

    # Step 4: Sequential Fusion
    # Start with the two smallest indices (positions 0 and 1 in sorted order)
    first_idx = sorted_idx_list[0]
    second_idx = sorted_idx_list[1]
    
    # Create first 2-to-1 isometry
    # For n=2 (base case), use the specified direction; otherwise use OUT for intermediate
    first_fused_dir = direction if n == 2 else Direction.OUT
    result = isometry(
        first_idx,
        second_idx,
        dtype=dtype,
        itags=(f"_iso_n_0", f"_iso_n_1", f"_iso_n_fused_0"),
        fused_direction=first_fused_dir,
    )
    
    # Sequentially fuse remaining indices
    for i in range(2, n):
        # Get the fused index from the previous result (last index)
        fused_idx = result.indices[-1]
        fused_tag = result.itags[-1]  # Get the tag of the fused index
        
        # Get next index to fuse
        next_idx = sorted_idx_list[i]
        
        # Flip the fused index to get opposite direction for contraction
        fused_idx_flipped = fused_idx.flip()
        
        # Determine if this is the last fusion
        is_last = (i == n - 1)
        # Use OUT for intermediate, use specified direction for final
        fused_dir = direction if is_last else Direction.OUT
        
        # Create new 2-to-1 isometry
        # Use the same tag for the first index to enable contraction
        new_iso = isometry(
            fused_idx_flipped,
            next_idx,
            dtype=dtype,
            itags=(fused_tag, f"_iso_n_{i}", f"_iso_n_fused_{i}"),
            fused_direction=fused_dir,
        )
        
        # Contract the new isometry with accumulated result
        # The fused index from result will contract with the first index of new_iso
        # They should have matching tags and opposite directions
        # Contract: last index of result with first index of new_iso (0)
        result_last_idx = len(result.indices) - 1
        result = contract(result, new_iso, axes=(result_last_idx, 0))
    
    # After all fusions, result has n indices (in sorted order) + 1 fused index
    # The indices are at positions 0, 1, ..., n-1, and the fused index is at position n
    
    # Step 5: Restore Original Index Order
    # Create permutation that moves indices back to their original positions
    # The fused index (currently at position n) stays at position n
    perm = inverse_perm + [n]
    result.permute(perm)
    
    # Step 6: Apply Tags
    if itags is not None:
        # Retag all indices with user-provided tags
        result.retag(itags)
    else:
        # Use default tags
        result.retag(["_init_"] * (n + 1))
    
    return result

