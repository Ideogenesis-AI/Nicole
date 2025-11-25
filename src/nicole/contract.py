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

"""Tensor contraction helpers for symmetry-aware tensors.

This module provides functions for contracting pairs of tensors along specified
index pairs while preserving charge conservation rules enforced by the symmetry
groups associated with the indices. The `contract` function implements the
general tensor contraction operation, while `trace` and `partial_trace`
provide specialised variants for reducing tensors along entire axes or subsets
of axes, respectively.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .blocks import BlockKey
from .index import Index
from .symmetry.base import AbelianGroup
from .tensor import Tensor
from .typing import Charge, Direction


def _dir_weight(idx: Index, charge: Charge) -> Tuple[AbelianGroup, Charge]:
    """Return the symmetry group and orientation-adjusted charge contribution."""
    group = idx.group
    if not isinstance(group, AbelianGroup):
        raise NotImplementedError("Only Abelian contraction supported initially")
    return group, (charge if idx.direction == Direction.OUT else group.inverse(charge))


def contract(
    A: Tensor,
    B: Tensor,
    pairs: Optional[Sequence[Tuple[int, int]]] = None,
    perm: Optional[Sequence[int]] = None,
) -> Tensor:
    """Contract two tensors along provided index pairs while respecting symmetry.

    Parameters
    ----------
    A, B:
        Input tensors to be contracted.
    pairs:
        Optional sequence of integer index pairs (axis in `A`, axis in `B`) to contract.
        If None, automatically contracts all indices where itags match and directions
        are opposite. If provided, validates that each pair has matching itags and
        opposite directions.
    perm:
        Optional permutation for the resulting tensor axes. If provided, the axes
        of the contracted tensor will be reordered according to this sequence.

    Returns
    -------
    Tensor
        Tensor whose indices are the non-contracted axes of `A` followed by those
        of `B`, populated with blocks that satisfy charge conservation.
    
    Raises
    ------
    ValueError
        If manually specified pairs have mismatched itags or non-opposite directions,
        or if no valid contraction pairs are found.
    """
    # Determine contraction pairs
    if pairs is None:
        # Automatic mode: find all pairs where itags match and directions are opposite
        axes = []
        used_B = set()
        for ia, tag_a in enumerate(A.itags):
            for ib, tag_b in enumerate(B.itags):
                if ib in used_B:
                    continue
                if tag_a == tag_b and A.indices[ia].direction != B.indices[ib].direction:
                    axes.append((ia, ib))
                    used_B.add(ib)
                    break
        if not axes:
            raise ValueError(
                "No valid contraction pairs found. Indices must have matching itags "
                "and opposite directions."
            )
    else:
        # Manual mode: validate matching itags and opposite directions
        axes = list(pairs)
        
        for ia, ib in axes:
            # Check bounds
            if ia < 0 or ia >= len(A.indices):
                raise ValueError(f"Index {ia} out of range for tensor A")
            if ib < 0 or ib >= len(B.indices):
                raise ValueError(f"Index {ib} out of range for tensor B")
            
            # Check matching itags
            if A.itags[ia] != B.itags[ib]:
                raise ValueError(
                    f"Contraction pair ({ia}, {ib}) has mismatched itags: "
                    f"'{A.itags[ia]}' (A) != '{B.itags[ib]}' (B). "
                    f"Contracted indices must have matching itags."
                )
            
            # Check opposite directions
            if A.indices[ia].direction == B.indices[ib].direction:
                raise ValueError(
                    f"Contraction pair ({ia}, {ib}) with itag '{A.itags[ia]}' has same direction: "
                    f"{A.indices[ia].direction}. Contracted indices must have opposite directions."
                )

    # Validate that the contraction pairs have matching symmetry groups.
    for ia, ib in axes:
        if A.indices[ia].group != B.indices[ib].group:
            raise ValueError("Contraction requires matching groups on paired indices")

    # Identify the contracted axes.
    contracted_A = {ia for ia, _ in axes}
    contracted_B = {ib for _, ib in axes}
    out_indices = tuple(idx for i, idx in enumerate(A.indices) if i not in contracted_A) + tuple(
        idx for i, idx in enumerate(B.indices) if i not in contracted_B
    )
    out_itags = tuple(tag for i, tag in enumerate(A.itags) if i not in contracted_A) + tuple(
        tag for i, tag in enumerate(B.itags) if i not in contracted_B
    )

    # Allocate the output blocks.
    out_blocks: Dict[BlockKey, np.ndarray] = {}
    # Iterate over all admissible blocks in the input tensors.
    for keyA, arrA in A.data.items():
        for keyB, arrB in B.data.items():
            # Check if the blocks are compatible for contraction.
            ok = True
            for ia, ib in axes:
                # Validate charge conservation for the pair.
                group, qa = _dir_weight(A.indices[ia], keyA[ia])
                _, qb = _dir_weight(B.indices[ib], keyB[ib])
                if not group.equal(group.fuse(qa, qb), group.neutral):
                    ok = False
                    break
                # Ensure matching dimensions.
                if arrA.shape[ia] != arrB.shape[ib]:
                    ok = False
                    break
            if not ok:
                continue
            # Perform the tensor contraction.
            axesA = [ia for ia, _ in axes]
            axesB = [ib for _, ib in axes]
            res = np.tensordot(arrA, arrB, axes=(axesA, axesB))
            # Build the output charge key from the surviving axes.
            out_key = tuple(keyA[i] for i in range(len(keyA)) if i not in contracted_A) + tuple(
                keyB[i] for i in range(len(keyB)) if i not in contracted_B
            )
            # Add the result to the output blocks.
            if out_key in out_blocks:
                out_blocks[out_key] = out_blocks[out_key] + res
            else:
                out_blocks[out_key] = res

    result = Tensor(
        indices=out_indices,
        itags=out_itags,
        data=out_blocks,
        dtype=np.result_type(A.dtype, B.dtype)
    )

    # Apply permutation if requested.
    if perm is not None:
        result.permute(perm)
    
    return result


def trace(T: Tensor, pairs: Sequence[Tuple[int, int]] | Sequence[Tuple[str, str]]) -> Tensor:
    """Trace over pairs of indices on a tensor, preserving symmetry constraints."""
    if pairs and isinstance(pairs[0][0], str):  # type: ignore[index]
        name_to_axis = {tag: i for i, tag in enumerate(T.itags)}
        axes = [(name_to_axis[a], name_to_axis[b]) for a, b in pairs]  # type: ignore[arg-type]
    else:
        axes = pairs  # type: ignore[assignment]
    contracted = set(i for p in axes for i in p)
    keep_axes = [i for i in range(len(T.indices)) if i not in contracted]
    out_indices = tuple(T.indices[i] for i in keep_axes)
    out_itags = tuple(T.itags[i] for i in keep_axes)
    out_blocks: Dict[BlockKey, np.ndarray] = {}
    for key, arr in T.data.items():
        ok = True
        for a, b in axes:
            group = T.indices[a].group
            if not isinstance(group, AbelianGroup):
                raise NotImplementedError("Only Abelian trace supported")
            qa = key[a]
            qb = key[b]
            if T.indices[a].direction == T.indices[b].direction:
                ok = False
                break
            if not group.equal(qa, qb):
                ok = False
                break
            if arr.shape[a] != arr.shape[b]:
                ok = False
                break
        if not ok:
            continue
        # Move traced axes to the end for convenient reshaping.
        axes_order = keep_axes + [a for p in axes for a in p]
        permuted = np.transpose(arr, axes=axes_order)
        keep_shape = [arr.shape[i] for i in keep_axes]
        traced_shapes = [arr.shape[a] for a, _ in axes]
        reshaped = permuted.reshape((*keep_shape, *traced_shapes, *traced_shapes))
        for k in range(len(axes)):
            reshaped = np.trace(reshaped, axis1=len(keep_shape) + k, axis2=len(keep_shape) + k + len(axes))
        out_key = tuple(key[i] for i in keep_axes)
        out_blocks[out_key] = out_blocks.get(out_key, 0) + reshaped

    return Tensor(indices=out_indices, itags=out_itags, data=out_blocks, dtype=T.dtype)


def partial_trace(T: Tensor, axes: Sequence[int] | Sequence[str]) -> Tensor:
    """Trace over a subset of axes specified as a flat list of pairs."""
    if axes and isinstance(axes[0], str):  # type: ignore[index]
        name_to_axis = {tag: i for i, tag in enumerate(T.itags)}
        iaxes = [name_to_axis[a] for a in axes]  # type: ignore[arg-type]
    else:
        iaxes = list(axes)  # type: ignore[assignment]
    if len(iaxes) % 2 != 0:
        raise ValueError("Partial trace requires an even number of axes (paired)")
    pairs = [(iaxes[i], iaxes[i + 1]) for i in range(0, len(iaxes), 2)]
    return trace(T, pairs)


