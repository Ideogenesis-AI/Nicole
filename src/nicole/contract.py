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
        or if no valid contraction pairs are found, or if automatic detection
        encounters ambiguous pairing.

    Examples
    --------
    **Automatic contraction** (recommended for most cases):

    >>> # Tensors with matching itags and opposite directions
    >>> A = Tensor.random([idx_a, idx_b], itags=["left", "mid"])
    >>> B = Tensor.random([idx_b_flip, idx_c], itags=["mid", "right"])
    >>> result = contract(A, B)  # Automatically contracts "mid" indices
    >>> result.itags
    ('left', 'right')

    **Manual contraction** with integer pairs:

    >>> # Explicitly specify which indices to contract
    >>> A = Tensor.random([idx_i, idx_j], itags=["i", "j"])
    >>> B = Tensor.random([idx_j_flip, idx_k], itags=["j", "k"])
    >>> result = contract(A, B, pairs=[(1, 0)])
    >>> result.itags
    ('i', 'k')

    **Multiple contractions**:

    >>> # Contract multiple index pairs at once
    >>> A = Tensor.random([idx_a, idx_b, idx_c], itags=["a", "b", "c"])
    >>> B = Tensor.random([idx_b_flip, idx_c_flip, idx_d], itags=["b", "c", "d"])
    >>> result = contract(A, B)  # Contracts both "b" and "c"
    >>> result.itags
    ('a', 'd')

    **Using permutation** to reorder output:

    >>> # Contract and then permute the result
    >>> A = Tensor.random([idx_i, idx_j], itags=["i", "j"])
    >>> B = Tensor.random([idx_j_flip, idx_k], itags=["j", "k"])
    >>> result = contract(A, B, pairs=[(1, 0)], perm=[1, 0])
    >>> result.itags  # Swapped from default order
    ('k', 'i')

    **Resolving ambiguity** with manual pairs:

    >>> # When automatic detection is ambiguous, specify explicitly
    >>> A = Tensor.random([idx_a, idx_a], itags=["x", "x"])  # Duplicate tags
    >>> B = Tensor.random([idx_a_flip, idx_a_flip], itags=["x", "x"])
    >>> # contract(A, B) would raise ValueError due to ambiguity
    >>> result = contract(A, B, pairs=[(0, 0), (1, 1)])  # Explicitly pair them

    Notes
    -----
    The automatic detection mode checks for unique pairings. If an itag appears
    multiple times with valid opposite directions, manual specification is required
    to avoid ambiguity.

    The output tensor has indices ordered as: non-contracted indices from A,
    followed by non-contracted indices from B. Use the `perm` parameter to
    reorder if needed.
    """
    # Determine contraction pairs
    if pairs is None:
        # Automatic mode: find all pairs where itags match and directions are opposite
        # Check for ambiguity: each index in A should match at most one index in B, and vice versa
        for ia, tag_a in enumerate(A.itags):
            matches = sum(
                1 for ib, tag_b in enumerate(B.itags)
                if tag_a == tag_b and A.indices[ia].direction != B.indices[ib].direction
            )
            if matches > 1:
                raise ValueError(
                    f"Ambiguous automatic contraction: index {ia} (itag '{tag_a}') in tensor A "
                    f"matches {matches} indices in tensor B. Please specify pairs explicitly."
                )
        
        for ib, tag_b in enumerate(B.itags):
            matches = sum(
                1 for ia, tag_a in enumerate(A.itags)
                if tag_a == tag_b and A.indices[ia].direction != B.indices[ib].direction
            )
            if matches > 1:
                raise ValueError(
                    f"Ambiguous automatic contraction: index {ib} (itag '{tag_b}') in tensor B "
                    f"matches {matches} indices in tensor A. Please specify pairs explicitly."
                )
        
        # Build unique pairing
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
    """Trace over pairs of indices on a single tensor while preserving symmetry.

    This function performs a partial trace by summing over diagonal elements of
    specified index pairs. Each pair must have matching charges, opposite directions,
    and equal dimensions within each symmetry block.

    Parameters
    ----------
    T:
        Tensor to be traced.
    pairs:
        Sequence of index pairs to trace over. Each pair (a, b) specifies two indices
        of the tensor to trace. Entries can be integer axis positions or string itag names.
        All indices in pairs must have opposite directions and matching charges.

    Returns
    -------
    Tensor
        Tensor with the traced indices removed. The remaining indices retain their
        original order.

    Raises
    ------
    NotImplementedError
        If the tensor uses non-Abelian symmetry groups.
    ValueError
        If paired indices have the same direction, mismatched charges, or
        incompatible dimensions.

    Notes
    -----
    The trace operation sums over matching diagonal entries: Tr(A) = Σᵢ Aᵢᵢ.
    For multiple pairs, traces are performed sequentially. The order of pairs
    does not affect the final result for commuting traces.

    Examples
    --------
    >>> # Trace over indices 0 and 1 of a 3-index tensor
    >>> result = trace(T, pairs=[(0, 1)])
    >>> 
    >>> # Trace over multiple pairs using itag names
    >>> result = trace(T, pairs=[("left", "right"), ("top", "bottom")])
    """
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
        # Trace each pair sequentially; after each trace, dimensions are reduced by 2
        for k in range(len(axes)):
            # After k traces, we've removed 2*k dimensions
            # So the next pair starts at len(keep_shape)
            reshaped = np.trace(reshaped, axis1=len(keep_shape), axis2=len(keep_shape) + len(axes) - k)
        # Ensure reshaped is a proper ndarray (not a scalar)
        if not isinstance(reshaped, np.ndarray):
            reshaped = np.array(reshaped)
        out_key = tuple(key[i] for i in keep_axes)
        if out_key in out_blocks:
            result = out_blocks[out_key] + reshaped
            # Ensure result is also an ndarray
            if not isinstance(result, np.ndarray):
                result = np.array(result)
            out_blocks[out_key] = result
        else:
            out_blocks[out_key] = reshaped

    return Tensor(indices=out_indices, itags=out_itags, data=out_blocks, dtype=T.dtype)


def partial_trace(T: Tensor, axes: Sequence[int] | Sequence[str]) -> Tensor:
    """Trace over a subset of indices specified as a flat sequential list.

    This is a convenience wrapper around `trace` that accepts a flat list of axes
    and automatically pairs them sequentially: [a₀, a₁, a₂, a₃, ...] becomes
    pairs [(a₀, a₁), (a₂, a₃), ...].

    Parameters
    ----------
    T:
        Tensor to be traced.
    axes:
        Flat sequence of indices to trace over. Must have even length. Indices
        are paired sequentially: first with second, third with fourth, etc.
        Can be integer axis positions or string itag names.

    Returns
    -------
    Tensor
        Tensor with the specified indices traced out. Remaining indices retain
        their original order.

    Raises
    ------
    ValueError
        If the number of axes is odd (cannot form complete pairs).
    NotImplementedError
        If the tensor uses non-Abelian symmetry groups.

    Examples
    --------
    >>> # Trace indices 0 with 1, and 2 with 3
    >>> result = partial_trace(T, axes=[0, 1, 2, 3])
    >>> # Equivalent to: trace(T, pairs=[(0, 1), (2, 3)])
    >>>
    >>> # Using itag names
    >>> result = partial_trace(T, axes=["left", "right", "top", "bottom"])
    >>> # Equivalent to: trace(T, pairs=[("left", "right"), ("top", "bottom")])

    Notes
    -----
    This function is particularly useful when you have a natural sequential
    ordering of index pairs, such as tracing out entangled pairs in a quantum
    system or reducing tensor products in a systematic way.
    """
    if axes and isinstance(axes[0], str):  # type: ignore[index]
        name_to_axis = {tag: i for i, tag in enumerate(T.itags)}
        iaxes = [name_to_axis[a] for a in axes]  # type: ignore[arg-type]
    else:
        iaxes = list(axes)  # type: ignore[assignment]
    if len(iaxes) % 2 != 0:
        raise ValueError("Partial trace requires an even number of axes (paired)")
    pairs = [(iaxes[i], iaxes[i + 1]) for i in range(0, len(iaxes), 2)]
    return trace(T, pairs)


