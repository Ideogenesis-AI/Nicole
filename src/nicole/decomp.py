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

"""Decomposition utilities for symmetry-aware Nicole (TN) tensors.

This module provides functions for decomposing tensors into their singular
value decomposition (SVD) components and eigenvalue decomposition.

Functions
---------
svd(T, axis, trunc=None)
    Low-level SVD returning U tensor, singular values dict, and Vh tensor.
    Returns singular values as 1D arrays for memory efficiency.

eig(T, trunc=None)
    Eigenvalue decomposition of square matrix returning U tensor and eigenvalues dict.
    Returns eigenvalues as 1D arrays for memory efficiency.

decomp(T, axis, mode="SVD", flow="><", itag=None, trunc=None)
    High-level decomposition with three modes:
    - "UR": Returns (U, R) where R = S*Vh
    - "SVD": Returns (U, S, Vh) with S as diagonal matrix tensor
    - "LV": Returns (L, V) where L = U*S
    The flow parameter controls arrow directions. The itag parameter customizes bond tags.
"""

from typing import Dict, List, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from .blocks import BlockKey
from .index import Index
from .tensor import Tensor
from .typing import Direction, Sector


def _axes_from_names(itags: Sequence[str], names: Sequence[str]) -> List[int]:
    """Translate itags into integer axes."""
    # Build a lookup once to avoid repeated linear searches.
    name_to_axis = {tag: i for i, tag in enumerate(itags)}
    return [name_to_axis[n] for n in names]


def svd(
    T: Tensor, 
    axis: int | str,
    trunc: Optional[Dict[str, Union[int, float]]] = None
) -> Tuple[Tensor, MutableMapping[BlockKey, np.ndarray], Tensor]:
    """Perform a symmetry-preserving SVD separating one axis from all others.

    Parameters
    ----------
    T:
        Tensor to be decomposed.
    axis:
        Axis to separate from all others. Can be an integer axis or itag.
        This axis forms the left partition, all others form the right partition.
    trunc:
        Truncation specification as a dict. If None, no truncation. Supported keys:
        - "nkeep": Keep at most n singular values globally (largest across all blocks)
        - "thresh": Keep singular values >= t per block
        Both can be specified together: thresh is applied first, then nkeep.

    Returns
    -------
    tuple[Tensor, MutableMapping[BlockKey, np.ndarray], Tensor]
        Triplet `(U, S_blocks, Vh)` where:
        - U has indices (left_index, bond_index)
        - S_blocks is a Dict mapping block keys to 1D arrays of singular values
        - Vh has indices (bond_index.flip(), *right_indices)
    
    Raises
    ------
    ValueError
        If trunc format is invalid or contains unsupported modes.
    
    Notes
    -----
    The singular values are returned as 1D arrays for memory efficiency.
    Use the `decomp()` function with mode="SVD" if you need S as a diagonal matrix tensor.
    
    For "nkeep" mode, truncation is applied globally: the top n singular values across
    all blocks are retained. For "thresh" mode, truncation is applied per block: each
    block independently keeps singular values >= threshold.
    
    When both modes are specified, "thresh" is applied first (per-block filtering),
    then "nkeep" is applied globally to the remaining singular values.
    
    Examples
    --------
    >>> # No truncation
    >>> U, S_blocks, Vh = svd(T, axis=0)
    >>> 
    >>> # Keep top 10 singular values
    >>> U, S_blocks, Vh = svd(T, axis=0, trunc={"nkeep": 10})
    >>> 
    >>> # Keep singular values >= 0.01
    >>> U, S_blocks, Vh = svd(T, axis=0, trunc={"thresh": 0.01})
    >>> 
    >>> # Apply both: first thresh, then nkeep
    >>> U, S_blocks, Vh = svd(T, axis=0, trunc={"thresh": 0.01, "nkeep": 10})
    """
    # Validate trunc parameter
    if trunc is not None:
        if not isinstance(trunc, dict):
            raise ValueError("trunc must be a dict with keys 'nkeep' and/or 'thresh'")
        
        unsupported = set(trunc.keys()) - {"nkeep", "thresh"}
        if unsupported:
            raise ValueError(f"Invalid truncation mode(s): {unsupported}. Must be 'nkeep' or 'thresh'")
    
    # Parse itags to integer axes
    if isinstance(axis, str):
        # Check for ambiguity: ensure the itag appears exactly once
        matching_axes = [i for i, tag in enumerate(T.itags) if tag == axis]
        if len(matching_axes) == 0:
            raise ValueError(f"itag '{axis}' not found in tensor")
        elif len(matching_axes) > 1:
            raise ValueError(
                f"Ambiguous axis specification: itag '{axis}' appears at "
                f"multiple positions {matching_axes}. Please use integer axis instead."
            )
        axis_idx = matching_axes[0]
    else:
        axis_idx = axis
        if axis_idx < 0 or axis_idx >= len(T.indices):
            raise ValueError(f"Axis index {axis_idx} out of range [0, {len(T.indices)})")
    
    # Define partitions: single axis vs all others
    left_axis = axis_idx
    right_axes = [i for i in range(len(T.indices)) if i != left_axis]
    
    # Build permutation to place left axis first
    perm = [left_axis] + right_axes
    
    # Get indices
    left_index = T.indices[left_axis]
    right_indices = tuple(T.indices[i] for i in right_axes)
    right_itags = tuple(T.itags[i] for i in right_axes)
    
    # Group blocks by left charge for proper general-purpose SVD
    # Structure: q_left -> list of (key, arr_perm, dims_right, mat)
    blocks_by_left_charge: Dict[tuple, List[Tuple[BlockKey, np.ndarray, Tuple[int, ...], np.ndarray]]] = {}
    
    for key, arr in T.data.items():
        # Permute array to [left_axis] + right_axes
        arr_perm = np.transpose(arr, axes=perm)
        
        # Get left charge and right charges
        q_left = key[left_axis]
        
        # Reshape to matrix: (dim_left, prod(dims_right))
        dim_left = arr_perm.shape[0]
        dims_right = arr_perm.shape[1:]
        dim_right_prod = int(np.prod(dims_right))
        mat = arr_perm.reshape(dim_left, dim_right_prod)
        
        # Group by left charge
        if q_left not in blocks_by_left_charge:
            blocks_by_left_charge[q_left] = []
        blocks_by_left_charge[q_left].append((key, arr_perm, dims_right, mat))
    
    # Perform SVD for each left charge sector by concatenating all blocks with same q_left
    svd_results: Dict[tuple, Tuple[np.ndarray, np.ndarray, Dict[BlockKey, np.ndarray]]] = {}
    bond_charge_dims: Dict[tuple, int] = {}
    
    for q_left, block_list in blocks_by_left_charge.items():
        # Concatenate all matrices with the same left charge horizontally
        mats = [mat for _, _, _, mat in block_list]
        concatenated_mat = np.concatenate(mats, axis=1)
        
        # Perform single SVD on concatenated matrix
        U, s, Vh = np.linalg.svd(concatenated_mat, full_matrices=False)
        
        # Apply per-block truncation for thresh mode
        if trunc is not None and "thresh" in trunc:
            keep_mask = s >= trunc["thresh"]
            U = U[:, keep_mask]
            s = s[keep_mask]
            Vh = Vh[keep_mask, :]
        
        # Skip this block if completely truncated
        if len(s) == 0:
            continue
        
        # Split Vh back to individual blocks
        Vh_dict: Dict[BlockKey, np.ndarray] = {}
        col_offset = 0
        for key, arr_perm, dims_right, mat in block_list:
            n_cols = mat.shape[1]
            Vh_block = Vh[:, col_offset:col_offset+n_cols]
            # Reshape back to original right dimensions
            rank = Vh_block.shape[0]
            Vh_reshaped = Vh_block.reshape((rank,) + dims_right)
            Vh_dict[key] = Vh_reshaped
            col_offset += n_cols
        
        # Store results grouped by left charge
        svd_results[q_left] = (U, s, Vh_dict)
        bond_charge_dims[q_left] = len(s)
    
    # Apply global truncation for nkeep mode
    if trunc is not None and "nkeep" in trunc:
        # Collect all singular values with their charges
        all_singular_values = []
        for q_left, (U, s, Vh_dict) in svd_results.items():
            for i, val in enumerate(s):
                all_singular_values.append((val, q_left, i))
        
        # Keep top nkeep singular values
        all_singular_values.sort(key=lambda x: x[0], reverse=True)
        keep_set = set((q, idx) for _, q, idx in all_singular_values[:trunc["nkeep"]])
        
        # Apply truncation to each block
        new_svd_results = {}
        new_bond_charge_dims = {}
        for q_left, (U, s, Vh_dict) in svd_results.items():
            # Find which indices to keep for this charge
            keep_indices = [i for i in range(len(s)) if (q_left, i) in keep_set]
            
            if len(keep_indices) > 0:
                # Truncate U, s, and Vh
                U_truncated = U[:, keep_indices]
                s_truncated = s[keep_indices]
                
                # Truncate all Vh blocks
                Vh_dict_truncated = {}
                for key, vh_block in Vh_dict.items():
                    Vh_dict_truncated[key] = vh_block[keep_indices, ...]
                
                new_svd_results[q_left] = (U_truncated, s_truncated, Vh_dict_truncated)
                new_bond_charge_dims[q_left] = len(s_truncated)
            # If no singular values kept for this charge, omit the block entirely
        
        svd_results = new_svd_results
        bond_charge_dims = new_bond_charge_dims
    
    # Build bond index with sectors from left charges
    bond_sectors = tuple(Sector(q, d) for q, d in sorted(bond_charge_dims.items(), key=lambda x: str(x[0])))
    bond_direction = left_index.direction.reverse()
    bond_index = Index(direction=bond_direction, group=left_index.group, sectors=bond_sectors)
    
    # Construct output blocks from grouped SVD results
    U_blocks: Dict[BlockKey, np.ndarray] = {}
    S_blocks: Dict[BlockKey, np.ndarray] = {}
    Vh_blocks: Dict[BlockKey, np.ndarray] = {}
    
    for q_left, (U, s, Vh_dict) in svd_results.items():
        rank = len(s)
        
        # For U tensor: indices (left_index, bond_index)
        # Block key: (q_left, q_left) since bond charge equals left charge
        U_key = (q_left, q_left)
        U_blocks[U_key] = U
        
        # For S: store singular values as 1D array (memory efficient)
        # Block key: (q_left, q_left)
        S_key = (q_left, q_left)
        S_blocks[S_key] = s.astype(np.result_type(T.dtype, float))
        
        # For Vh tensor: indices (bond_index.flip(), *right_indices)
        # Each block gets its corresponding Vh from the dictionary
        for key, Vh_reshaped in Vh_dict.items():
            q_right = tuple(key[i] for i in right_axes)
            Vh_key = (q_left,) + q_right
            Vh_blocks[Vh_key] = Vh_reshaped
    
    # Construct output tensors
    U_tensor = Tensor(
        indices=(left_index, bond_index),
        itags=(T.itags[left_axis], "_bond_L"),
        data=U_blocks,
        dtype=T.dtype
    )
    
    Vh_tensor = Tensor(
        indices=(bond_index.flip(),) + right_indices,
        itags=("_bond_R",) + right_itags,
        data=Vh_blocks,
        dtype=T.dtype
    )
    
    return U_tensor, S_blocks, Vh_tensor


def eig(
    T: Tensor,
    itag: Optional[str] = None,
    trunc: Optional[Dict[str, Union[int, float]]] = None
) -> Tuple[Tensor, MutableMapping[BlockKey, np.ndarray]]:
    """Perform eigenvalue decomposition of a square matrix tensor.

    Parameters
    ----------
    T:
        Square matrix tensor to be decomposed. Must have exactly 2 indices
        with matching charge structure (opposite directions).
    itag:
        Index tag for the bond dimension. If None, uses default tag "_bond_eig".
    trunc:
        Truncation specification as a dict. If None, no truncation. Supported keys:
        - "nkeep": Keep at most n eigenvalues globally (largest by magnitude)
        - "thresh": Keep eigenvalues with |eigenvalue| >= t per block
        Both can be specified together: thresh is applied first, then nkeep.

    Returns
    -------
    tuple[Tensor, MutableMapping[BlockKey, np.ndarray]]
        Pair `(U, D)` where:
        - U has indices (row_index, bond_index) containing eigenvectors as columns
        - D is a Dict mapping block keys to 1D arrays of eigenvalues
        
        The decomposition satisfies: T @ U = U @ diag(D) for each block
    
    Raises
    ------
    ValueError
        If T is not a square matrix, or if indices are not compatible,
        or if trunc format is invalid or contains unsupported modes.
    
    Notes
    -----
    The eigenvalues are returned as 1D arrays for memory efficiency.
    Eigenvalues can be complex even for real matrices.
    
    For "nkeep" mode, truncation is applied globally: the top n eigenvalues by
    magnitude across all blocks are retained. For "thresh" mode, truncation is
    applied per block: each block independently keeps eigenvalues with |λ| >= threshold.
    
    When both modes are specified, "thresh" is applied first (per-block filtering),
    then "nkeep" is applied globally to the remaining eigenvalues.
    
    The eigenvectors are stored in columns of U, normalized such that U is unitary
    (or as close as the eigendecomposition provides).
    
    Examples
    --------
    >>> # No truncation
    >>> U, D_blocks = eig(T)
    >>> 
    >>> # Keep top 5 eigenvalues
    >>> U, D_blocks = eig(T, trunc={"nkeep": 5})
    >>> 
    >>> # Keep eigenvalues with |λ| >= 0.1
    >>> U, D_blocks = eig(T, trunc={"thresh": 0.1})
    >>> 
    >>> # Apply both: first thresh, then nkeep
    >>> U, D_blocks = eig(T, trunc={"thresh": 0.1, "nkeep": 5})
    """
    # Validate input tensor
    if len(T.indices) != 2:
        raise ValueError(f"eig requires a square matrix, got {len(T.indices)} indices")
    
    row_index, col_index = T.indices
    
    # Check that indices have opposite directions (required for eigendecomposition)
    if row_index.direction == col_index.direction:
        raise ValueError(
            f"Indices must have opposite directions for square matrix. "
            f"Got both {row_index.direction}"
        )
    
    # Validate trunc parameter
    if trunc is not None:
        if not isinstance(trunc, dict):
            raise ValueError("trunc must be a dict with keys 'nkeep' and/or 'thresh'")
        
        unsupported = set(trunc.keys()) - {"nkeep", "thresh"}
        if unsupported:
            raise ValueError(f"Invalid truncation mode(s): {unsupported}. Must be 'nkeep' or 'thresh'")
    
    # Set bond tag
    bond_tag = itag if itag is not None else "_bond_eig"
    
    # Perform eigendecomposition block by block
    # Only diagonal blocks (same charge) can exist for square matrix
    eig_results: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}
    bond_charge_dims: Dict[tuple, int] = {}
    
    for key, arr in T.data.items():
        q_row = key[0]
        # Note: charge conservation ensures q_row == key[1] for square matrices
        
        # Perform eigendecomposition
        # np.linalg.eig returns (eigenvalues, eigenvectors)
        # eigenvectors[:, i] is the eigenvector for eigenvalues[i]
        eigenvalues, eigenvectors = np.linalg.eig(arr)
        
        # Apply per-block truncation for thresh mode
        if trunc is not None and "thresh" in trunc:
            keep_mask = np.abs(eigenvalues) >= trunc["thresh"]
            eigenvalues = eigenvalues[keep_mask]
            eigenvectors = eigenvectors[:, keep_mask]
        
        # Skip this block if completely truncated
        if len(eigenvalues) == 0:
            continue
        
        # Store results
        eig_results[q_row] = (eigenvectors, eigenvalues)
        bond_charge_dims[q_row] = len(eigenvalues)
    
    # Apply global truncation for nkeep mode
    if trunc is not None and "nkeep" in trunc:
        # Collect all eigenvalues with their charges
        all_eigenvalues = []
        for q, (eigvecs, eigvals) in eig_results.items():
            for i, val in enumerate(eigvals):
                all_eigenvalues.append((np.abs(val), q, i))
        
        # Keep top nkeep eigenvalues by magnitude
        all_eigenvalues.sort(key=lambda x: x[0], reverse=True)
        keep_set = set((q, idx) for _, q, idx in all_eigenvalues[:trunc["nkeep"]])
        
        # Apply truncation to each block
        new_eig_results = {}
        new_bond_charge_dims = {}
        for q, (eigvecs, eigvals) in eig_results.items():
            # Find which indices to keep for this charge
            keep_indices = [i for i in range(len(eigvals)) if (q, i) in keep_set]
            
            if len(keep_indices) > 0:
                # Truncate eigenvectors and eigenvalues
                eigvecs_truncated = eigvecs[:, keep_indices]
                eigvals_truncated = eigvals[keep_indices]
                
                new_eig_results[q] = (eigvecs_truncated, eigvals_truncated)
                new_bond_charge_dims[q] = len(eigvals_truncated)
            # If no eigenvalues kept for this charge, omit the block entirely
        
        eig_results = new_eig_results
        bond_charge_dims = new_bond_charge_dims
    
    # Build bond index with sectors from charges
    bond_sectors = tuple(Sector(q, d) for q, d in sorted(bond_charge_dims.items(), key=lambda x: str(x[0])))
    bond_direction = row_index.direction.reverse()
    bond_index = Index(direction=bond_direction, group=row_index.group, sectors=bond_sectors)
    
    # Construct output blocks
    U_blocks: Dict[BlockKey, np.ndarray] = {}
    D_blocks: Dict[BlockKey, np.ndarray] = {}
    
    for q, (eigvecs, eigvals) in eig_results.items():
        # For U tensor: indices (row_index, bond_index)
        # Block key: (q, q) since bond charge equals row charge
        U_key = (q, q)
        U_blocks[U_key] = eigvecs
        
        # For D: store eigenvalues as 1D array (memory efficient)
        # Block key: (q, q)
        D_key = (q, q)
        # Preserve complex type if eigenvalues are complex
        D_blocks[D_key] = eigvals.astype(np.result_type(T.dtype, np.complex128))
    
    # Construct output tensor
    U_tensor = Tensor(
        indices=(row_index, bond_index),
        itags=(T.itags[0], bond_tag),
        data=U_blocks,
        dtype=np.result_type(T.dtype, np.complex128)  # May be complex
    )
    
    return U_tensor, D_blocks


def decomp(
    T: Tensor,
    axes: Union[int, str, Sequence[Union[int, str]]],
    mode: str = "SVD",
    flow: str = "><",
    itag: Optional[Union[str, Tuple[str, str]]] = None,
    trunc: Optional[Dict[str, Union[int, float]]] = None
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Perform tensor decomposition with flexible output modes.
    
    Parameters
    ----------
    T:
        Tensor to be decomposed.
    axes:
        Index or indices to separate from all others. Can be:
        - Single integer position or string tag
        - Sequence of integer positions or string tags (merges multiple axes first)
    mode:
        Decomposition mode:
        - "UR": Returns (U, R) where R = S*Vh (singular values multiplied into Vh)
        - "SVD": Returns (U, S, Vh) where S is diagonal matrix tensor (full SVD)
        - "LV": Returns (L, V) where L = U*S (singular values multiplied into U)
    flow:
        Arrow direction control. Default is "><" (both arrows incoming).
        - For SVD mode: Controls S matrix arrow directions ("><", ">>", or "<<")
        - For UR mode: Both ">>" and "><" normalize to ">>" (outward bonds); "<<" is also accepted
        - For LV mode: Both "<<" and "><" normalize to "<<" (inward bonds); ">>" is also accepted
        Note: The underlying svd naturally produces ">>" or "<<" depending on left_index.direction.
        This parameter uses tensor.flip() to adjust from the natural flow to the desired flow.
    itag:
        Index tag(s) for the bond dimension(s). Can be:
        - None: Use default tags "_bond_L" and "_bond_R"
        - str: Use same tag for both left and right bonds
        - tuple[str, str]: Use (left_tag, right_tag) for left and right bonds respectively
    trunc:
        Truncation specification as a dict. If None, no truncation. Supported keys:
        - "nkeep": Keep at most n singular values globally
        - "thresh": Keep singular values >= t per block
        Both can be specified together: thresh is applied first, then nkeep.
    
    Returns
    -------
    tuple[Tensor, Tensor] or tuple[Tensor, Tensor, Tensor]
        - "UR" mode: (U, R) where R incorporates singular values
        - "SVD" mode: (U, S, Vh) with S as diagonal matrix tensor
        - "LV" mode: (L, V) where L incorporates singular values
    
    Raises
    ------
    ValueError
        If mode is not one of "UR", "SVD", or "LV", or if flow is not ">>", "<<", or "><"
    
    Examples
    --------
    >>> from nicole import Tensor, decomp, U1Group, Direction, Index, Sector
    >>> 
    >>> # Create a sample tensor
    >>> group = U1Group()
    >>> idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    >>> idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    >>> T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=1)
    >>> 
    >>> # UR mode: Get U and R=S*Vh (most efficient for reconstruction)
    >>> U, R = decomp(T, axes=0, mode="UR")
    >>> 
    >>> # SVD mode: Get full SVD with diagonal S
    >>> U, S, Vh = decomp(T, axes=0, mode="SVD")
    >>> 
    >>> # LV mode: Get L=U*S and V
    >>> L, V = decomp(T, axes=0, mode="LV")
    
    Notes
    -----
    - UR and LV modes are more memory and computationally efficient than SVD mode
    - SVD mode constructs a full diagonal matrix tensor for S
    - All modes produce mathematically equivalent decompositions
    - When multiple axes are specified, they are first merged using an n-to-1 isometry,
      decomposed, and then the U tensor is unmerged back to the original axes
    """
    # Import merge_axes here to avoid circular dependency
    from .operators import merge_axes
    from .contract import contract
    
    # Check if axes is a sequence (multiple axes)
    is_multi_axis = isinstance(axes, (list, tuple))
    
    if is_multi_axis:
        # Multiple axes: merge, decompose, unmerge
        axes_list = axes
        if len(axes_list) < 2:
            raise ValueError("When providing a sequence of axes, must specify at least 2 axes")
        
        # Merge the specified axes
        merged_T, iso_conj = merge_axes(T, axes_list, merged_tag="_decomp_merged_")
        
        # The merged index is now at position 0 (merge_axes places it first)
        # Decompose on the merged axis
        result = decomp(
            merged_T,
            axes=0,  # Merged index is at position 0
            mode=mode,
            flow=flow,
            itag=itag,
            trunc=trunc
        )
        
        # Unmerge the U tensor (first element of result)
        if mode == "SVD":
            U, S, Vh = result
            # Unmerge U by contracting with conjugate isometry
            # The merged index is at position 0 of U, and at last position of iso_conj
            iso_conj_last_idx = len(iso_conj.indices) - 1
            U_unmerged = contract(iso_conj, U, axes=(iso_conj_last_idx, 0))
            return U_unmerged, S, Vh
        else:  # mode == "UR" or "LV"
            first, second = result
            # For UR mode, first is U; for LV mode, first is L
            # Both have the merged index at position 0
            # The merged index is at position 0 of first, and at last position of iso_conj
            iso_conj_last_idx = len(iso_conj.indices) - 1
            first_unmerged = contract(iso_conj, first, axes=(iso_conj_last_idx, 0))
            return first_unmerged, second
    
    # Single axis: original behavior
    # Validate mode
    mode = mode.upper()
    if mode not in ("UR", "SVD", "LV"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'UR', 'SVD', or 'LV'")
    
    # Validate flow parameter
    if flow not in (">>", "<<", "><"):
        raise ValueError(f"Invalid flow '{flow}'. Must be '>>', '<<', or '><'")
    
    # Parse itag parameter
    if itag is None:
        bond_tag_left = "_bond_L"
        bond_tag_right = "_bond_R"
    elif isinstance(itag, str):
        bond_tag_left = itag
        bond_tag_right = itag
    elif isinstance(itag, tuple) and len(itag) == 2:
        bond_tag_left, bond_tag_right = itag
    else:
        raise ValueError("itag must be None, a string, or a tuple of two strings")
    
    # Parse axes to get left_index (do this once to avoid duplication in svd)
    if isinstance(axes, str):
        matching_axes = [i for i, tag in enumerate(T.itags) if tag == axes]
        if len(matching_axes) == 0:
            raise ValueError(f"itag '{axes}' not found in tensor")
        elif len(matching_axes) > 1:
            raise ValueError(
                f"Ambiguous axis specification: itag '{axes}' appears at "
                f"multiple positions {matching_axes}. Please use integer axis instead."
            )
        axis_idx = matching_axes[0]
    else:
        axis_idx = axes
    
    # Determine natural flow from svd based on left_index direction
    left_index = T.indices[axis_idx]
    # Natural flow is "<<" if left_index is OUT, ">>" if left_index is IN
    natural_flow = "<<" if left_index.direction == Direction.OUT else ">>"
    
    # Perform SVD to get U, singular values dict, and Vh (pass integer axis_idx)
    U, S_blocks, Vh = svd(T, axis_idx, trunc=trunc)
    
    # Update U and Vh bond tags to use custom tags
    U.retag({U.itags[1]: bond_tag_left})
    Vh.retag({Vh.itags[0]: bond_tag_right})
    
    if mode == "SVD":
        # Construct full diagonal S tensor
        bond_index = U.indices[1]  # Extract bond index from U
        
        S_diag_blocks: Dict[BlockKey, np.ndarray] = {}
        for key, s_array in S_blocks.items():
            # Convert 1D singular values to diagonal matrix
            S_diag_blocks[key] = np.diag(s_array)
        
        # Natural S has indices matching the natural flow from svd
        S_tensor = Tensor(
            indices=(bond_index.flip(), bond_index),
            itags=(bond_tag_left, bond_tag_right),
            data=S_diag_blocks,
            dtype=np.result_type(T.dtype, float),
            label="Diagonal"
        )
        
        # Apply tensor flip to convert from natural_flow to desired flow
        # When we flip S, we must also flip the corresponding index in U or Vh
        if natural_flow == ">>":
            # Natural for S: (IN, OUT)
            if flow == "><":
                # Desired: (IN, IN) - flip S's right index and Vh's bond index
                S_tensor.flip(1)
                Vh.flip(0)
            elif flow == "<<":
                # Desired: (OUT, IN) - flip both S indices and both U's bond and Vh's bond
                S_tensor.flip([0, 1])
                U.flip(1)
                Vh.flip(0)
            # else flow == ">>": natural, no flip needed
        else:  # natural_flow == "<<"
            # Natural for S: (OUT, IN)
            if flow == "><":
                # Desired: (IN, IN) - flip S's left index and U's bond index
                S_tensor.flip(0)
                U.flip(1)
            elif flow == ">>":
                # Desired: (IN, OUT) - flip both S indices and both U's bond and Vh's bond
                S_tensor.flip([0, 1])
                U.flip(1)
                Vh.flip(0)
            # else flow == "<<": natural, no flip needed
        
        return U, S_tensor, Vh
    
    elif mode == "UR":
        # Multiply singular values into Vh to get R = S*Vh
        R_blocks: Dict[BlockKey, np.ndarray] = {}
        
        for key, vh_block in Vh.data.items():
            # key = (q_bond, *q_right)
            # S_blocks key = (q_bond, q_bond)
            q_bond = key[0]
            s_key = (q_bond, q_bond)
            
            if s_key in S_blocks:
                s_array = S_blocks[s_key]
                # Multiply: R = diag(s) @ Vh = s[:, None, ...] * Vh
                # vh_block shape: (rank, *right_dims)
                # Broadcast multiplication along first axis
                rank = len(s_array)
                s_broadcasted = s_array.reshape((rank,) + (1,) * (vh_block.ndim - 1))
                R_blocks[key] = (s_broadcasted * vh_block).astype(T.dtype)
            else:
                # No singular values for this block (shouldn't happen normally)
                R_blocks[key] = vh_block
        
        # Change bond tag to match U's bond tag for easier contraction
        R_itags = (bond_tag_left,) + Vh.itags[1:]
        
        # R inherits Vh's bond index structure
        R_tensor = Tensor(
            indices=Vh.indices,
            itags=R_itags,
            data=R_blocks,
            dtype=T.dtype
        )
        
        # For UR mode: normalize flow (both ">>" and "><" mean ">>")
        normalized_flow = ">>" if flow in (">>", "><") else "<<"
        
        # Flip if normalized flow differs from natural flow
        if normalized_flow != natural_flow:
            U.flip(1)  # Flip U's bond index (position 1)
            R_tensor.flip(0)  # Flip R's bond index (position 0)
        
        return U, R_tensor
    
    else:  # mode == "LV"
        # Multiply singular values into U to get L = U*S
        L_blocks: Dict[BlockKey, np.ndarray] = {}
        
        for key, u_block in U.data.items():
            # key = (q_left, q_bond)
            # S_blocks key = (q_bond, q_bond)
            q_bond = key[1]
            s_key = (q_bond, q_bond)
            
            if s_key in S_blocks:
                s_array = S_blocks[s_key]
                # Multiply: L = U @ diag(s) = U * s[None, :]
                # u_block shape: (dim_left, rank)
                # Broadcast multiplication along second axis
                L_blocks[key] = (u_block * s_array[None, :]).astype(T.dtype)
            else:
                # No singular values for this block (shouldn't happen normally)
                L_blocks[key] = u_block
        
        # Change bond tag to match Vh's bond tag for easier contraction
        L_itags = (U.itags[0], bond_tag_right)
        
        # L inherits U's bond index structure
        L_tensor = Tensor(
            indices=U.indices,
            itags=L_itags,
            data=L_blocks,
            dtype=T.dtype
        )
        
        # For LV mode: normalize flow (both "<<" and "><" mean "<<")
        normalized_flow = "<<" if flow in ("<<", "><") else ">>"
        
        # Flip if normalized flow differs from natural flow
        if normalized_flow != natural_flow:
            L_tensor.flip(1)  # Flip L's bond index (position 1)
            Vh.flip(0)  # Flip Vh's bond index (position 0)
        
        return L_tensor, Vh
