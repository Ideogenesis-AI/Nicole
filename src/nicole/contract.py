# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Nicole library.
#
# Nicole is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Nicole is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Nicole. If not, see <https://www.gnu.org/licenses/>.


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

import torch

from .blocks import BlockKey, BlockSchema
from .index import Index
from .symmetry import SymmetryGroup
from .symmetry.delegate import compute_xsymbol, compute_rsymbol, Bridge
from .tensor import Tensor
from .typing import Charge, Direction


def _dir_weight(idx: Index, charge: Charge) -> Tuple[SymmetryGroup, Charge]:
    """Return the symmetry group and orientation-adjusted charge contribution."""
    group = idx.group
    return group, (charge if idx.direction == Direction.OUT else group.dual(charge))


def _detect_contraction_pairs(
    A: Tensor,
    B: Tensor,
    excl_A: set[int] = None,
    excl_B: set[int] = None,
) -> list[tuple[int, int]]:
    """Detect contraction pairs automatically with optional exclusions.
    
    Parameters
    ----------
    A, B:
        Input tensors.
    excl_A:
        Set of axes in A to exclude from contraction.
    excl_B:
        Set of axes in B to exclude from contraction.
    
    Returns
    -------
    list[tuple[int, int]]
        List of axis pairs to contract.
    
    Raises
    ------
    ValueError
        If ambiguous pairing is detected or no valid pairs found.
    """
    if excl_A is None:
        excl_A = set()
    if excl_B is None:
        excl_B = set()
    
    # Check for ambiguity: each index in A should match at most one index in B, and vice versa
    for ia, tag_a in enumerate(A.itags):
        if ia in excl_A:
            continue
        matches = sum(
            1 for ib, tag_b in enumerate(B.itags)
            if ib not in excl_B and tag_a == tag_b and A.indices[ia].direction != B.indices[ib].direction
        )
        if matches > 1:
            raise ValueError(
                f"Ambiguous automatic contraction: index {ia} (itag '{tag_a}') in tensor A "
                f"matches {matches} indices in tensor B. Please specify axes explicitly."
            )
    
    for ib, tag_b in enumerate(B.itags):
        if ib in excl_B:
            continue
        matches = sum(
            1 for ia, tag_a in enumerate(A.itags)
            if ia not in excl_A and tag_a == tag_b and A.indices[ia].direction != B.indices[ib].direction
        )
        if matches > 1:
            raise ValueError(
                f"Ambiguous automatic contraction: index {ib} (itag '{tag_b}') in tensor B "
                f"matches {matches} indices in tensor A. Please specify axes explicitly."
            )
    
    # Build unique pairing
    axes_list = []
    used_B = set()
    for ia, tag_a in enumerate(A.itags):
        if ia in excl_A:
            continue
        for ib, tag_b in enumerate(B.itags):
            if ib in used_B or ib in excl_B:
                continue
            if tag_a == tag_b and A.indices[ia].direction != B.indices[ib].direction:
                axes_list.append((ia, ib))
                used_B.add(ib)
                break
    
    if not axes_list:
        if excl_A or excl_B:
            raise ValueError(
                "No valid contraction pairs found after applying exclusions. "
                "Indices must have matching itags and opposite directions."
            )
        else:
            raise ValueError(
                "No valid contraction pairs found. Indices must have matching itags "
                "and opposite directions."
            )
    
    return axes_list


def contract(
    A: Tensor,
    B: Tensor,
    axes: Optional[Tuple[int, int] | Tuple[Sequence[int], Sequence[int]]] = None,
    excl: Optional[Tuple[Sequence[int], Sequence[int]]] = None,
    perm: Optional[Sequence[int]] = None,
) -> Tensor:
    """Contract two tensors along provided index pairs while respecting symmetry.

    Parameters
    ----------
    A, B:
        Input tensors to be contracted.
    axes:
        Optional specification of axes to contract. Can be either:
        - Single pair: (axis_in_A, axis_in_B) for contracting one pair
        - Multiple pairs: ([axes_in_A], [axes_in_B]) for contracting multiple pairs
        Similar to np.tensordot syntax. Validates that each pair has matching itags 
        and opposite directions. Mutually exclusive with `excl`.
    excl:
        Optional tuple of two sequences specifying axes to exclude from automatic contraction:
        ([excl_axes_in_A], [excl_axes_in_B]). When specified, automatically contracts all
        indices with matching itags and opposite directions, except those in the exclusion lists.
        Examples: excl=((0,), ()) excludes A's axis 0; excl=((), (1,)) excludes B's axis 1.
        Mutually exclusive with `axes`.
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
        If both axes and excl are specified, or if manually specified axes have
        mismatched itags or non-opposite directions, or if no valid contraction
        pairs are found, or if automatic detection encounters ambiguous pairing.

    Examples
    --------
    **Automatic contraction** (recommended for most cases):

    >>> # Tensors with matching itags and opposite directions
    >>> A = Tensor.random([idx_a, idx_b], itags=["left", "mid"])
    >>> B = Tensor.random([idx_b_flip, idx_c], itags=["mid", "right"])
    >>> result = contract(A, B)  # Automatically contracts "mid" indices
    >>> result.itags
    ('left', 'right')

    **Manual contraction with axes parameter:**

    >>> # Single pair: concise syntax (axis_in_A, axis_in_B)
    >>> A = Tensor.random([idx_i, idx_j], itags=["i", "j"])
    >>> B = Tensor.random([idx_j_flip, idx_k], itags=["j", "k"])
    >>> result = contract(A, B, axes=(1, 0))
    >>> result.itags
    ('i', 'k')

    >>> # Single pair: also works with sequence syntax
    >>> result = contract(A, B, axes=([1], [0]))

    **Multiple contractions:**

    >>> # Contract multiple index pairs at once
    >>> A = Tensor.random([idx_a, idx_b, idx_c], itags=["a", "b", "c"])
    >>> B = Tensor.random([idx_b_flip, idx_c_flip, idx_d], itags=["b", "c", "d"])
    >>> result = contract(A, B, axes=([1, 2], [0, 1]))  # or use automatic mode
    >>> result.itags
    ('a', 'd')

    **Using excl parameter for automatic contraction with exclusions:**

    >>> # Contract all matching itags except A's axis 0
    >>> result = contract(A, B, excl=((0,), ()))
    
    >>> # Contract all matching itags except B's axis 0
    >>> result = contract(A, B, excl=((), (0,)))
    
    >>> # Exclude axes from both tensors
    >>> result = contract(A, B, excl=((0, 1), (2,)))

    **Using permutation** to reorder output:

    >>> # Contract and then permute the result
    >>> A = Tensor.random([idx_i, idx_j], itags=["i", "j"])
    >>> B = Tensor.random([idx_j_flip, idx_k], itags=["j", "k"])
    >>> result = contract(A, B, axes=(1, 0), perm=[1, 0])
    >>> result.itags  # Swapped from default order
    ('k', 'i')

    **Resolving ambiguity** with manual axes:

    >>> # When automatic detection is ambiguous, specify explicitly
    >>> A = Tensor.random([idx_a, idx_a], itags=["x", "x"])  # Duplicate tags
    >>> B = Tensor.random([idx_a_flip, idx_a_flip], itags=["x", "x"])
    >>> # contract(A, B) would raise ValueError due to ambiguity
    >>> result = contract(A, B, axes=([0, 1], [0, 1]))  # Explicitly pair them

    Notes
    -----
    The automatic detection mode checks for unique pairings. If an itag appears
    multiple times with valid opposite directions, manual specification is required
    to avoid ambiguity.

    The output tensor has indices ordered as: non-contracted indices from A,
    followed by non-contracted indices from B. Use the `perm` parameter to
    reorder if needed.
    """
    # Validate mutually exclusive parameters
    if axes is not None and excl is not None:
        raise ValueError("Cannot specify both 'axes' and 'excl' parameters")
    
    # Determine contraction pairs
    if axes is not None:
        # Manual mode: convert axes tuple to pairs list
        if len(axes) != 2:
            raise ValueError(f"axes must be a tuple of length 2, got length {len(axes)}")
        
        # Check if axes is a single pair (int, int) or multiple pairs ([int, ...], [int, ...])
        if isinstance(axes[0], int) and isinstance(axes[1], int):
            # Single pair: (axis_in_A, axis_in_B)
            axes_list = [(axes[0], axes[1])]
        else:
            # Multiple pairs: ([axes_in_A], [axes_in_B])
            axes_A, axes_B = axes
            if len(axes_A) != len(axes_B):
                raise ValueError(
                    f"axes sequences must have same length: {len(axes_A)} != {len(axes_B)}"
                )
            axes_list = [(axes_A[i], axes_B[i]) for i in range(len(axes_A))]
        
        # Validate each pair
        for ia, ib in axes_list:
            # Check bounds
            if ia < 0 or ia >= len(A.indices):
                raise ValueError(f"Index position (axis) {ia} out of range for tensor A")
            if ib < 0 or ib >= len(B.indices):
                raise ValueError(f"Index position (axis) {ib} out of range for tensor B")
            
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
    elif excl is not None:
        # Automatic mode with exclusions
        if len(excl) != 2:
            raise ValueError(f"excl must be a tuple of two sequences, got length {len(excl)}")
        excl_A, excl_B = set(excl[0]), set(excl[1])
        axes_list = _detect_contraction_pairs(A, B, excl_A, excl_B)
    else:
        # Pure automatic mode
        axes_list = _detect_contraction_pairs(A, B)

    # Validate that the contraction pairs have matching symmetry groups.
    for ia, ib in axes_list:
        if A.indices[ia].group != B.indices[ib].group:
            raise ValueError("Contraction requires matching groups on paired indices")

    # Identify the contracted axes.
    contracted_A = {ia for ia, _ in axes_list}
    contracted_B = {ib for _, ib in axes_list}
    out_indices = tuple(idx for i, idx in enumerate(A.indices) if i not in contracted_A) + tuple(
        idx for i, idx in enumerate(B.indices) if i not in contracted_B
    )
    out_itags = tuple(tag for i, tag in enumerate(A.itags) if i not in contracted_A) + tuple(
        tag for i, tag in enumerate(B.itags) if i not in contracted_B
    )

    # Allocate the output blocks.
    out_blocks: Dict[BlockKey, torch.Tensor] = {}
    out_intw: Optional[Dict[BlockKey, Bridge]] = {} if A.intw is not None else None
    # Iterate over all admissible blocks in the input tensors.
    for keyA, arrA in A.data.items():
        for keyB, arrB in B.data.items():
            # Check if the blocks are compatible for contraction.
            ok = True
            for ia, ib in axes_list:
                # Validate charge conservation for the pair.
                group, qa = _dir_weight(A.indices[ia], keyA[ia])
                _, qb = _dir_weight(B.indices[ib], keyB[ib])
                
                # Check if fusion to neutral is allowed
                if group.is_abelian:
                    # Abelian: use fuse_unique
                    if not group.equal(group.fuse_unique(qa, qb), group.neutral):
                        ok = False
                        break
                else:
                    # Non-Abelian: check if neutral is in fuse_channels
                    if group.neutral not in group.fuse_channels(qa, qb):
                        ok = False
                        break
                
                # Ensure matching dimensions.
                if arrA.shape[ia] != arrB.shape[ib]:
                    ok = False
                    break
            if not ok:
                continue
            # Perform the tensor contraction.
            axesA = [ia for ia, _ in axes_list]
            axesB = [ib for _, ib in axes_list]
            
            # Build the output charge key from the surviving axes.
            out_key = tuple(keyA[i] for i in range(len(keyA)) if i not in contracted_A) + tuple(
                keyB[i] for i in range(len(keyB)) if i not in contracted_B
            )
            
            # For generic (non-Abelian) tensors, check if output block satisfies charge conservation
            if A.intw is not None:
                if not BlockSchema.charges_conserved(out_indices, out_key):
                    continue
            
            # Handle Abelian vs non-Abelian contraction
            if A.intw is None:
                # Abelian: simple tensordot
                res = torch.tensordot(arrA, arrB, dims=(axesA, axesB))
                bridge_res = None
            else:
                # Non-Abelian (SU(2)): contract physical axes only, preserve component dimensions
                # Block A: (...phys_A..., k_a), Block B: (...phys_B..., k_b)
                # Contract ONLY physical axes, keeping component dimensions separate
                
                # Promote dtype if needed (tensordot requires matching dtypes)
                dtype_result = torch.promote_types(arrA.dtype, arrB.dtype)
                if arrA.dtype != dtype_result:
                    arrA = arrA.to(dtype_result)
                if arrB.dtype != dtype_result:
                    arrB = arrB.to(dtype_result)
                
                res_phys = torch.tensordot(arrA, arrB, dims=(axesA, axesB))
                
                # After tensordot, the component dimensions are scattered in the result
                # We need to identify their positions and move them to the end
                
                bridge_a = A.intw[keyA]
                bridge_b = B.intw[keyB]
                k_a = bridge_a.num_components
                k_b = bridge_b.num_components
                n_kept_a = len(A.indices) - len(axesA)
                n_kept_b = len(B.indices) - len(axesB)
                
                if n_kept_a == 0 and n_kept_b == 0:
                    # Scalar: contract components using weight overlap matrix.
                    # The CG-basis overlap <CG_A|CG_B> = conj_phase·δ only
                    # when the contracted edges of A and B appear in the same internal
                    # slot order. When axesA ≠ axesB, a relative permutation
                    # align_perm = axesB ∘ axesA⁻¹ misaligns the two CG trees; we correct
                    # this by applying the corresponding R-symbol to bridge_b's weights
                    # before forming the overlap matrix.
                    weights_a = bridge_a.weights.to(dtype_result)
                    weights_b = bridge_b.weights.to(dtype_result)

                    # Compute align_perm such that align_perm[axesA[k]] = axesB[k]:
                    # this maps B's CG-tree edge slots into the same contracted
                    # order as A, so the OM-basis overlap becomes diagonal (δ).
                    align_perm = [0] * len(axesA)
                    for k, a in enumerate(axesA):
                        align_perm[a] = axesB[k]

                    # Apply R-symbol only when align_perm is non-trivial
                    if align_perm != list(range(len(axesA))):
                        r_b, _ = compute_rsymbol(bridge_b, align_perm)
                        weights_b = weights_b @ r_b.to(dtype_result)

                    weight_overlap = weights_a @ weights_b.T
                    res = (res_phys * weight_overlap).sum().reshape(())

                    # Conjugation phase from bridge_a (accounts for FS signs in A's
                    # CG tree; B's basis is now aligned to A's ordering)
                    phase = bridge_a.conj_phase()
                    res = res * phase

                    bridge_res = None
                else:
                    # Non-scalar: recouple k_a × k_b → k_c via X-symbol
                    k_c = k_a * k_b
                    
                    # Compute X-symbol and output CGSpec
                    x_symbol, spec_c = compute_xsymbol(bridge_a, bridge_b, axesA, axesB)
                    
                    # Component dims are at positions n_kept_a and (n_kept_a + 1 + n_kept_b)
                    # Move them to end: (...phys..., k_a, k_b) → (...phys..., k_c)
                    pos_k_a = n_kept_a
                    pos_k_b = n_kept_a + 1 + n_kept_b
                    res_phys = torch.moveaxis(res_phys, [pos_k_a, pos_k_b], [-2, -1])
                    res = res_phys.reshape(list(res_phys.shape[:-2]) + [k_c])
                    
                    # Combine weights: (k_a, om_a) ⊗ (k_b, om_b) × X(om_a, om_b, om_c) → (k_c, om_c)
                    weights_a = bridge_a.weights
                    weights_b = bridge_b.weights
                    weights_c = torch.einsum('ia,jb,abc->ijc', weights_a, weights_b, x_symbol)
                    
                    # Reshape to (k_a * k_b, om_c)
                    weights_c = weights_c.reshape(k_c, x_symbol.shape[2])
                    
                    # Create output bridge
                    bridge_res = Bridge(cgspec=spec_c, weights=weights_c)
            
            # Add the result to the output blocks.
            if out_key in out_blocks:
                if A.intw is None:
                    # Abelian: simple addition
                    out_blocks[out_key] = out_blocks[out_key] + res
                elif bridge_res is None:
                    # Non-Abelian scalar: simple addition
                    out_blocks[out_key] = out_blocks[out_key] + res
                else:
                    # Non-Abelian: use block_add to combine blocks and weights
                    out_blocks[out_key], out_intw[out_key] = BlockSchema.block_add(
                        out_blocks[out_key], out_intw[out_key],
                        res, bridge_res
                    )
            else:
                out_blocks[out_key] = res
                if out_intw is not None and bridge_res is not None:
                    out_intw[out_key] = bridge_res
    
    # For 0D scalars, ensure the result is a proper tensor
    # PyTorch maintains tensor type through operations, so this is less critical
    if len(out_indices) == 0 and () in out_blocks:
        out_blocks[()] = torch.as_tensor(out_blocks[()])

    # For scalar results or when all bridges are None, set intw to None
    if out_intw is not None:
        if len(out_indices) == 0:  # Scalar result
            out_intw = None
        elif out_intw and all(v is None for v in out_intw.values()):
            out_intw = None
    
    out_tensor = Tensor(
        indices=out_indices, itags=out_itags, data=out_blocks, intw=out_intw,
        dtype=torch.promote_types(A.dtype, B.dtype)
    )

    # Row-normalize intertwiner weights and redistribute norms into data to
    # prevent weights from decaying across successive contractions.
    out_tensor.regularize()

    # When order drops, OM typically drops, making accumulated components from
    # block_add likely to exceed OM. Compress to remove genuine rank deficiency.
    # The order-reduction condition is a heuristic to balance the performance
    # gain from compression against the SVD cost: order-preserving or
    # order-increasing contractions tend to have growing OM, so components are
    # unlikely to be redundant and SVD overhead is not justified.
    # Regularize first so that the SVD cutoff operates on well-scaled rows;
    # after compress the new weights (Vh rows) are already orthonormal, so no
    # second regularize is needed.
    if out_intw is not None and len(out_indices) < max(len(A.indices), len(B.indices)):
        out_tensor.compress()

    # Apply permutation if requested.
    if perm is not None:
        out_tensor.permute(perm, in_place=True)
    
    return out_tensor


def _detect_trace_pairs(T: Tensor, excl: set[int] = None) -> list[tuple[int, int]]:
    """Detect trace pairs automatically within a tensor with optional exclusions.
    
    Parameters
    ----------
    T:
        Input tensor.
    excl:
        Set of axes to exclude from tracing.
    
    Returns
    -------
    list[tuple[int, int]]
        List of axis pairs to trace.
    
    Raises
    ------
    ValueError
        If ambiguous pairing is detected or no valid pairs found.
    """
    if excl is None:
        excl = set()
    
    # Check for ambiguity: each index should match at most one other index
    for i, tag_i in enumerate(T.itags):
        if i in excl:
            continue
        matches = []
        for j, tag_j in enumerate(T.itags):
            if j == i or j in excl:
                continue
            if tag_i == tag_j and T.indices[i].direction != T.indices[j].direction:
                matches.append(j)
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous automatic trace: index {i} (itag '{tag_i}') "
                f"matches {len(matches)} other indices with opposite direction. "
                "Please specify axes explicitly."
            )
    
    # Build unique pairing
    pairs = []
    used = set()
    for i, tag_i in enumerate(T.itags):
        if i in excl or i in used:
            continue
        for j, tag_j in enumerate(T.itags):
            if j <= i or j in used or j in excl:
                continue
            if tag_i == tag_j and T.indices[i].direction != T.indices[j].direction:
                pairs.append((i, j))
                used.add(i)
                used.add(j)
                break
    
    if not pairs:
        if excl:
            raise ValueError(
                "No valid trace pairs found after applying exclusions. "
                "Indices must have matching itags and opposite directions."
            )
        else:
            raise ValueError(
                "No valid trace pairs found. Indices must have matching itags "
                "and opposite directions."
            )
    
    return pairs


def trace(
    T: Tensor,
    axes: Optional[Tuple[int, int] | Sequence[Tuple[int, int]]] = None,
    excl: Optional[int | str | Sequence[int] | Sequence[str]] = None,
) -> Tensor:
    """Trace over pairs of indices on a single tensor while preserving symmetry.

    This function performs a partial trace by summing over diagonal elements of
    specified index pairs. Each pair must have matching charges, opposite directions,
    and equal dimensions within each symmetry block.

    Parameters
    ----------
    T:
        Tensor to be traced.
    axes:
        Optional specification of axes to trace. Can be either:
        - Single pair: (axis_a, axis_b) for tracing one pair
        - Multiple pairs: [(a1, b1), (a2, b2), ...] for tracing multiple pairs
        Must use integer indices only. Validates that each pair has matching itags
        and opposite directions. Mutually exclusive with `excl`.
    excl:
        Optional specification of axes to exclude from automatic tracing.
        Can be a single int/str or sequence of ints/strs. When specified, automatically
        traces all indices with matching itags and opposite directions, except those
        in the exclusion list. Mutually exclusive with `axes`.

    Returns
    -------
    Tensor
        Tensor with the traced indices removed. The remaining indices retain their
        original order.

    Raises
    ------
    ValueError
        If both axes and excl are specified, or if paired indices have the same direction,
        mismatched charges, incompatible dimensions, or if no valid pairs are found,
        or if automatic detection encounters ambiguous pairing.

    Notes
    -----
    The trace operation sums over matching diagonal entries: Tr(A) = Σᵢ Aᵢᵢ.
    
    For multiple pairs, traces are performed sequentially to ensure consistency:
    `trace(T, [(a,b), (c,d)])` is equivalent to `trace(trace(T, (a,b)), (c,d))`.
    This guarantees that the result does not depend on whether pairs are traced
    simultaneously or one at a time. The order of pairs does not affect the
    final result for commuting traces.

    Examples
    --------
    **Automatic tracing** (recommended for most cases):

    >>> # Tensors with matching itags and opposite directions
    >>> T = Tensor.random([idx_a, idx_a_flip, idx_b], itags=["left", "left", "mid"])
    >>> result = trace(T)  # Automatically traces "left" pair

    **Manual tracing with axes parameter:**

    >>> # Single pair: concise syntax
    >>> result = trace(T, axes=(0, 1))

    >>> # Single pair: also works with sequence syntax
    >>> result = trace(T, axes=[(0, 1)])

    >>> # Multiple pairs
    >>> result = trace(T, axes=[(0, 1), (2, 3)])

    **Using excl parameter for automatic tracing with exclusions:**

    >>> # Trace all matching pairs except axis 0
    >>> result = trace(T, excl=0)

    >>> # Trace all matching pairs except "left" indices
    >>> result = trace(T, excl="left")

    >>> # Exclude multiple axes
    >>> result = trace(T, excl=[0, 2])
    """
    # Validate mutually exclusive parameters
    if axes is not None and excl is not None:
        raise ValueError("Cannot specify both 'axes' and 'excl' parameters")
    
    # Parse axes parameter
    if axes is not None:
        # Check if single pair (int, int) or multiple pairs
        if isinstance(axes, tuple) and len(axes) == 2 and isinstance(axes[0], int):
            pairs = [axes]
        else:
            pairs = list(axes)  # type: ignore[arg-type]
    else:
        # Automatic mode: detect pairs, possibly with exclusions
        excl_set = set()
        if excl is not None:
            # Parse exclusions
            if isinstance(excl, (int, str)):
                excl_items = [excl]
            else:
                excl_items = list(excl)
            
            # Convert string tags to indices
            for item in excl_items:
                if isinstance(item, str):
                    if item not in T.itags:
                        raise ValueError(f"Exclusion itag '{item}' not found in tensor")
                    # Find all indices with this tag
                    for i, tag in enumerate(T.itags):
                        if tag == item:
                            excl_set.add(i)
                else:
                    if item < 0 or item >= len(T.itags):
                        raise IndexError(f"Exclusion axis {item} out of range")
                    excl_set.add(item)
        
        pairs = _detect_trace_pairs(T, excl_set)
    
    # Base case: no pairs to trace
    if not pairs:
        return T
    
    # Recursive case: trace first pair, then recursively trace remaining pairs
    # This ensures consistency: trace(T, [pair1, pair2]) == trace(trace(T, pair1), pair2)
    if len(pairs) > 1:
        # Trace the first pair
        first_pair = pairs[0]
        a, b = first_pair
        traced_first = trace(T, axes=first_pair)
        
        # Adjust indices in remaining pairs after removing axes a and b
        # Axes are removed in order, so we need to account for both removals
        adjusted_pairs = []
        for pair in pairs[1:]:
            i, j = pair
            # Adjust for removal of axis a
            if i > a:
                i -= 1
            if j > a:
                j -= 1
            # Adjust for removal of axis b (note: b might have shifted after removing a)
            b_adjusted = b if b < a else b - 1
            if i > b_adjusted:
                i -= 1
            if j > b_adjusted:
                j -= 1
            adjusted_pairs.append((i, j))
        
        # Recursively trace the remaining pairs
        return trace(traced_first, axes=adjusted_pairs)
    
    # Base case: single pair to trace
    a, b = pairs[0]

    # Validate the pair upfront before touching any block data
    if T.indices[a].direction == T.indices[b].direction:
        raise ValueError(
            f"Trace axes {a} and {b} have the same direction "
            f"({T.indices[a].direction}). Traced pairs must have opposite directions."
        )

    group = T.indices[a].group

    contracted = {a, b}
    keep_axes = [i for i in range(len(T.indices)) if i not in contracted]
    out_indices = tuple(T.indices[i] for i in keep_axes)
    out_itags = tuple(T.itags[i] for i in keep_axes)
    out_blocks: Dict[BlockKey, torch.Tensor] = {}
    out_intw: Optional[Dict[BlockKey, Bridge]] = {} if T.intw is not None else None

    for key, arr in T.data.items():
        qa = key[a]
        qb = key[b]

        # Skip blocks where charges don't match or dimensions are incompatible
        if not group.equal(qa, qb):
            continue
        if arr.shape[a] != arr.shape[b]:
            continue

        # Data part: identical for Abelian and non-Abelian.
        # torch.diagonal moves the traced diagonal to the last axis; summing collapses it.
        # For non-Abelian blocks arr has a trailing component axis k that is preserved.
        diag = torch.diagonal(arr, dim1=a, dim2=b).sum(dim=-1)
        
        # Ensure diag is a proper tensor (not a scalar)
        if not isinstance(diag, torch.Tensor):
            diag = torch.tensor(diag)

        out_key = tuple(key[i] for i in keep_axes)

        if T.intw is None:
            # Abelian: plain accumulation.
            out_blocks[out_key] = out_blocks.get(out_key, 0) + diag
        else:
            # Non-Abelian (SU(2) / ProductGroup with SU(2)).
            # Tracing axes a and b is equivalent to contracting with a 2nd order identity
            # tensor on those indices. Build bridge_I at charge q = qa = qb with directions
            # reversed from T's traced indices and weight √(irrep_dim(q)), exactly as
            # identity() does, then get the X-symbol for that virtual contraction.
            q = qa
            irrep_dim = group.irrep_dim(q)
            w_I = torch.full((1, 1), float(irrep_dim ** 0.5), dtype=T.dtype)
            bridge_I = Bridge.from_block(
                group, (q, q),
                [T.indices[a].direction.reverse(), T.indices[b].direction.reverse()],
                weights=w_I,
            )
            bridge_T = T.intw[key]

            if len(keep_axes) == 0:
                # Scalar output: yuzuha cannot build a 0-edge CGSpec, so use the
                # weight-overlap formula directly (mirrors the scalar path in contract()).
                weight_overlap = bridge_I.weights @ bridge_T.weights.T  # (1, k_T)
                scalar_contrib = (diag.unsqueeze(0) * weight_overlap).sum()
                scalar_contrib = (scalar_contrib * bridge_I.conj_phase()).reshape(())
                if () in out_blocks:
                    out_blocks[()] = out_blocks[()] + scalar_contrib
                else:
                    out_blocks[()] = scalar_contrib
                # out_intw is set to None after the loop for scalar results
            else:
                # Skip blocks whose remaining charges cannot form a neutral tensor;
                # their trace contribution is zero by SU(2) selection rules.
                if not BlockSchema.charges_conserved(out_indices, out_key):
                    continue

                # Non-scalar: recouple via X-symbol.
                x_symbol, spec_c = compute_xsymbol(bridge_T, bridge_I, [a, b], [0, 1])

                # Combine weights: (k_T, om_T) ⊗ (1, 1) × X(om_T, 1, om_c) → (k_T, om_c)
                weights_c = torch.einsum(
                    'ia,jb,abc->ijc', bridge_T.weights, bridge_I.weights, x_symbol
                ).reshape(bridge_T.num_components, x_symbol.shape[2])
                bridge_res = Bridge(cgspec=spec_c, weights=weights_c)

                if out_key in out_blocks:
                    out_blocks[out_key], out_intw[out_key] = BlockSchema.block_add(
                        out_blocks[out_key], out_intw[out_key],
                        diag, bridge_res,
                    )
                else:
                    out_blocks[out_key] = diag
                    out_intw[out_key] = bridge_res

    # Scalar non-Abelian results carry no intertwiner.
    if out_intw is not None and len(out_indices) == 0:
        out_intw = None

    out_tensor = Tensor(
        indices=out_indices, itags=out_itags, data=out_blocks,
        intw=out_intw, dtype=T.dtype
    )

    # Trace always reduces order by 2, so OM always drops. Regularize first so
    # that the SVD cutoff in compress operates on well-scaled rows; the condition
    # is unconditional here (unlike contract) because order reduction is guaranteed.
    if out_intw is not None:
        out_tensor.regularize()
        out_tensor.compress()

    return out_tensor


