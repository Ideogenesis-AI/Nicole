# Copyright (C) 2025 Changkai Zhang.
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

"""Standalone tensor operators for functional-style tensor manipulation.

This module provides functional versions of tensor operations that return new
Tensor instances rather than modifying tensors in-place. These functions are
useful for functional programming patterns and for cases where immutability
is desired.

Functions
---------
conj(tensor)
    Return a new tensor with conjugated data and flipped index directions.
permute(tensor, order)
    Return a new tensor with permuted axes according to the provided order.
transpose(tensor, *order)
    Return a new tensor with transposed axes; defaults to reversing axis order.
getsub(tensor, block_indices)
    Return a new tensor containing only the specified blocks.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .index import Index
from .tensor import Tensor
from .typing import Charge, Sector


def conj(tensor: Tensor) -> Tensor:
    """Return a new tensor with conjugated data and flipped index directions.
    
    Parameters
    ----------
    tensor:
        The input tensor to conjugate.
    
    Returns
    -------
    Tensor
        A new tensor instance with:
        - Conjugated dense blocks (if dtype is complex)
        - All index directions flipped
        - All other attributes preserved
    """
    # Only conjugate data if dtype is complex
    if np.issubdtype(tensor.dtype, np.complexfloating):
        new_data = {k: np.conjugate(v) for k, v in tensor.data.items()}
    else:
        new_data = {k: v.copy() for k, v in tensor.data.items()}
    
    # Flip all index directions
    new_indices = tuple(idx.flip() for idx in tensor.indices)
    
    return Tensor(indices=new_indices, itags=tensor.itags, data=new_data, dtype=tensor.dtype, label=tensor.label)


def permute(tensor: Tensor, order: Sequence[int]) -> Tensor:
    """Return a new tensor with permuted axes according to the provided order.
    
    Parameters
    ----------
    tensor:
        The input tensor to permute.
    order:
        Sequence of axis indices specifying the new ordering. Must be a
        permutation of range(len(tensor.indices)).
    
    Returns
    -------
    Tensor
        A new tensor instance with reordered indices and transposed blocks.
    
    Raises
    ------
    ValueError
        If order is not a valid permutation.
    
    Examples
    --------
    >>> from nicole import permute, Tensor
    >>> # Assuming t is a 3-index tensor with indices [a, b, c]
    >>> t_perm = permute(t, [2, 0, 1])  # Reorder to [c, a, b]
    """
    if sorted(order) != list(range(len(tensor.indices))):
        raise ValueError("Invalid permutation order")
    
    new_indices = tuple(tensor.indices[i] for i in order)
    new_itags = tuple(tensor.itags[i] for i in order)
    new_data = {}
    
    for key, arr in tensor.data.items():
        new_key = tuple(key[i] for i in order)
        new_data[new_key] = np.transpose(arr, axes=order)
    
    return Tensor(indices=new_indices, itags=new_itags, data=new_data, dtype=tensor.dtype, label=tensor.label)


def transpose(tensor: Tensor, *order: int) -> Tensor:
    """Return a new tensor with transposed axes; defaults to reversing axis order.
    
    Parameters
    ----------
    tensor:
        The input tensor to transpose.
    *order:
        Optional axis indices specifying the new ordering. If not provided,
        defaults to reversing the axis order.
    
    Returns
    -------
    Tensor
        A new tensor instance with transposed axes.
    
    Examples
    --------
    >>> from nicole import transpose, Tensor
    >>> # Assuming t is a 3-index tensor with indices [a, b, c]
    >>> t_T = transpose(t)  # Reverse order to [c, b, a]
    >>> t_T2 = transpose(t, 1, 0, 2)  # Swap first two indices to [b, a, c]
    """
    if not order:
        order = tuple(reversed(range(len(tensor.indices))))
    return permute(tensor, order)


def getsub(tensor: Tensor, block_indices: Sequence[int]) -> Tensor:
    """Return a new tensor containing only the specified blocks.
    
    Parameters
    ----------
    tensor:
        The input tensor to extract blocks from.
    block_indices:
        Sequence of block indices (1-indexed, matching display numbering)
        specifying which blocks to include in the new tensor.
    
    Returns
    -------
    Tensor
        A new tensor instance containing only the specified blocks,
        with all other attributes (indices, itags, dtype, label) preserved.
    
    Raises
    ------
    IndexError
        If any block index is out of range.
    
    Examples
    --------
    >>> from nicole import getsub, Tensor
    >>> # Assuming t has 5 blocks numbered 1-5 in display
    >>> t_sub = getsub(t, [1, 3, 5])  # Extract blocks 1, 3, and 5
    """
    num_blocks = len(tensor.data)
    for i in block_indices:
        if i < 1 or i > num_blocks:
            raise IndexError(f"Block index {i} out of range [1, {num_blocks}]")
    
    new_data = {tensor.key(i): tensor.block(i).copy() for i in block_indices}
    
    return Tensor(
        indices=tensor.indices,
        itags=tensor.itags,
        data=new_data,
        dtype=tensor.dtype,
        label=tensor.label,
    )


def oplus(
    A: Tensor,
    B: Tensor,
    axes: Optional[Union[Sequence[int], Sequence[str]]] = None
) -> Tensor:
    """Direct sum of two tensors with selective axis merging.
    
    Combines two tensors by merging their sector structures along specified
    axes and arranging blocks in a block-diagonal fashion. Axes not specified
    must match exactly (same sectors, same dimensions).
    
    Parameters
    ----------
    A : Tensor
        First tensor
    B : Tensor
        Second tensor
    axes : Optional[Union[Sequence[int], Sequence[str]]], default=None
        Axes to merge. Can be:
        - None: merge all axes (default)
        - Sequence of integers: axis positions (e.g., [0, 2])
        - Sequence of strings: itag names (e.g., ['i', 'k'])
        Axes not specified must have identical Index structure in A and B.
    
    Returns
    -------
    Tensor
        Direct sum with merged indices on specified axes
    
    Raises
    ------
    ValueError
        If tensors have incompatible structure or if non-merged axes don't match exactly
    
    Examples
    --------
    >>> from nicole import Tensor, U1Group, Direction, Index, Sector, oplus
    >>> import numpy as np
    >>> 
    >>> group = U1Group()
    >>> 
    >>> # Example 1: Default - merge all axes
    >>> idx_A0 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    >>> idx_A1 = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2)))
    >>> A = Tensor.random([idx_A0, idx_A1], seed=1, itags=['i', 'j'])
    >>> 
    >>> idx_B0 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
    >>> idx_B1 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(2, 1)))
    >>> B = Tensor.random([idx_B0, idx_B1], seed=2, itags=['i', 'j'])
    >>> 
    >>> C = oplus(A, B)  # Merges both axes
    >>> # C.indices[0] has sectors: [(0, 3), (1, 3), (2, 2)]
    >>> # C.indices[1] has sectors: [(0, 5), (1, 2), (2, 1)]
    >>> 
    >>> # Example 2: Selective - merge only first axis
    >>> idx_A1_match = Index(Direction.IN, group, sectors=(Sector(0, 5),))
    >>> idx_B1_match = Index(Direction.IN, group, sectors=(Sector(0, 5),))  # Must match!
    >>> A2 = Tensor.random([idx_A0, idx_A1_match], seed=3, itags=['i', 'j'])
    >>> B2 = Tensor.random([idx_B0, idx_B1_match], seed=4, itags=['i', 'j'])
    >>> 
    >>> C2 = oplus(A2, B2, axes=[0])  # or axes=['i']
    >>> # C2.indices[0] has sectors: [(0, 3), (1, 3), (2, 2)]  ← merged
    >>> # C2.indices[1] has sectors: [(0, 5)]  ← unchanged (matched exactly)
    
    Notes
    -----
    - Blocks are arranged in a block-diagonal fashion along merged axes
    - For merged axes, dimensions add for sectors with the same charge
    - Non-merged axes must have identical sectors and dimensions
    - Charge conservation is maintained in the output tensor
    """
    # Step 1: Validate basic compatibility
    if len(A.indices) != len(B.indices):
        raise ValueError(
            f"Tensors must have the same number of indices, got {len(A.indices)} and {len(B.indices)}"
        )
    
    if len(A.indices) == 0:
        raise ValueError("Cannot apply direct sum to scalar tensors (0 indices)")
    
    # Step 2: Resolve axes argument
    n_axes = len(A.indices)
    if axes is None:
        # Default: merge all axes
        axes_int = list(range(n_axes))
    elif len(axes) > 0 and isinstance(axes[0], str):
        # Convert itags to integer positions
        axes_int = []
        for tag in axes:
            try:
                idx = A.itags.index(tag)
                axes_int.append(idx)
            except ValueError:
                raise ValueError(f"Itag '{tag}' not found in tensor A")
    else:
        # Already integers
        axes_int = list(axes)
    
    # Validate axes range
    for ax in axes_int:
        if not isinstance(ax, int) or ax < 0 or ax >= n_axes:
            raise ValueError(f"Invalid axis {ax}, must be in range [0, {n_axes})")
    
    # Remove duplicates and sort
    axes_int = sorted(set(axes_int))
    merged_axes = set(axes_int)
    non_merged_axes = set(range(n_axes)) - merged_axes
    
    # Step 3: Validate index compatibility
    for i in range(n_axes):
        idx_A = A.indices[i]
        idx_B = B.indices[i]
        
        # Basic checks for all axes
        if idx_A.group != idx_B.group:
            raise ValueError(
                f"Index {i}: Tensors must have the same symmetry group, "
                f"got {type(idx_A.group).__name__} and {type(idx_B.group).__name__}"
            )
        
        if idx_A.direction != idx_B.direction:
            raise ValueError(
                f"Index {i}: Tensors must have the same direction, "
                f"got {idx_A.direction} and {idx_B.direction}"
            )
        
        # Non-merged axes must match exactly
        if i in non_merged_axes:
            charges_A = set(idx_A.charges())
            charges_B = set(idx_B.charges())
            if charges_A != charges_B:
                raise ValueError(
                    f"Index {i} (non-merged): Must have identical charge sectors, "
                    f"got {charges_A} and {charges_B}"
                )
            
            dim_map_A = idx_A.sector_dim_map()
            dim_map_B = idx_B.sector_dim_map()
            for charge in charges_A:
                if dim_map_A[charge] != dim_map_B[charge]:
                    raise ValueError(
                        f"Index {i} (non-merged): Must have identical dimensions for charge {charge}, "
                        f"got {dim_map_A[charge]} and {dim_map_B[charge]}"
                    )
    
    # Step 4: Build output indices
    out_indices = []
    # Track sector information for each axis
    # merged_sector_info[axis][charge] = (dim_A, dim_B, dim_total)
    merged_sector_info: Dict[int, Dict[Charge, Tuple[int, int, int]]] = {}
    
    for i in range(n_axes):
        if i in merged_axes:
            # Merge sectors
            idx_A = A.indices[i]
            idx_B = B.indices[i]
            
            dim_map_A = idx_A.sector_dim_map()
            dim_map_B = idx_B.sector_dim_map()
            
            # Union of charges
            all_charges = set(idx_A.charges()) | set(idx_B.charges())
            
            sector_info = {}
            new_sectors = []
            
            for charge in sorted(all_charges):
                dim_A = dim_map_A.get(charge, 0)
                dim_B = dim_map_B.get(charge, 0)
                dim_total = dim_A + dim_B
                
                sector_info[charge] = (dim_A, dim_B, dim_total)
                new_sectors.append(Sector(charge, dim_total))
            
            merged_sector_info[i] = sector_info
            
            # Create new index with merged sectors
            new_index = Index(
                direction=idx_A.direction,
                group=idx_A.group,
                sectors=tuple(new_sectors)
            )
            out_indices.append(new_index)
        else:
            # Copy index from A (same as B by validation)
            out_indices.append(A.indices[i])
    
    # Step 5: Build output blocks
    # We need to identify all valid charge combinations and place blocks
    
    # First, collect all possible charge keys from A and B
    all_charge_keys = set(A.data.keys()) | set(B.data.keys())
    
    out_data = {}
    
    for charge_key in all_charge_keys:
        # Determine output shape for this charge combination
        out_shape = []
        for i, charge in enumerate(charge_key):
            if i in merged_axes:
                # Use merged dimension
                _, _, dim_total = merged_sector_info[i][charge]
                out_shape.append(dim_total)
            else:
                # Use exact dimension from A (same as B)
                dim_map_A = A.indices[i].sector_dim_map()
                out_shape.append(dim_map_A[charge])
        
        # Initialize output block with zeros
        out_block = np.zeros(out_shape, dtype=np.result_type(A.dtype, B.dtype))
        
        # Place block from A if it exists
        if charge_key in A.data:
            block_A = A.data[charge_key]
            # Build slices for placing block_A
            slices_A = []
            for i, charge in enumerate(charge_key):
                if i in merged_axes:
                    # Use offset 0:dim_A
                    dim_A, _, _ = merged_sector_info[i][charge]
                    slices_A.append(slice(0, dim_A))
                else:
                    # Use full dimension
                    slices_A.append(slice(None))
            
            out_block[tuple(slices_A)] = block_A
        
        # Place block from B if it exists
        if charge_key in B.data:
            block_B = B.data[charge_key]
            # Build slices for placing block_B
            slices_B = []
            for i, charge in enumerate(charge_key):
                if i in merged_axes:
                    # Use offset dim_A:dim_total
                    dim_A, dim_B, dim_total = merged_sector_info[i][charge]
                    slices_B.append(slice(dim_A, dim_total))
                else:
                    # Use full dimension
                    slices_B.append(slice(None))
            
            out_block[tuple(slices_B)] = block_B
        
        # Only add non-zero blocks
        if np.any(out_block != 0):
            out_data[charge_key] = out_block
    
    # Step 6: Create and return output tensor
    return Tensor(
        indices=tuple(out_indices),
        itags=A.itags,
        data=out_data,
        dtype=np.result_type(A.dtype, B.dtype),
        label=A.label
    )

