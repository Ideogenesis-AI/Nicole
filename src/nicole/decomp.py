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

"""Decomposition utilities for symmetry-aware Nicole tensors.

This module provides functions for decomposing tensors into their singular
value decomposition (SVD) components, QR decomposition, and eigen-decomposition.

Functions
---------
svd(T, axis, trunc=None)
    Low-level SVD returning U tensor, singular values dict, and Vh tensor.
    Returns singular values as 1D arrays for memory efficiency.

qr(T, axis)
    QR decomposition returning Q (orthogonal) and R (upper triangular) tensors.
    Separates specified axis into Q, all other axes go to R. No truncation applied.

eig(T, itag=None, order="ascend", trunc=None)
    Eigenvalue decomposition of square matrix returning U tensor and eigenvalues dict.
    Returns eigenvalues as 1D arrays for memory efficiency. Supports sorting eigenvalues
    in ascending or descending order (by value for real eigenvalues, by real part for complex).

decomp(T, axis, mode="SVD", flow="><", itag=None, trunc=None)
    High-level decomposition with four modes:
    - "SVD": Returns (U, S, Vh) with S as diagonal matrix tensor
    - "UR": Returns (U, R) where R = S*Vh
    - "LV": Returns (L, V) where L = U*S
    - "QR": Returns (Q, R) where Q is orthogonal and R is upper triangular
    The flow parameter controls arrow directions. The itag parameter customizes bond tags.
"""

from typing import Dict, List, Literal, MutableMapping, Optional, Sequence, Tuple, Union
import math

import torch

from .blocks import BlockKey
from .index import Index
from .tensor import Tensor
from .typing import Direction, Sector
from .symmetry import delegate as dg


def _axes_from_names(itags: Sequence[str], names: Sequence[str]) -> List[int]:
    """Translate itags into integer axes."""
    # Build a lookup once to avoid repeated linear searches.
    name_to_axis = {tag: i for i, tag in enumerate(itags)}
    return [name_to_axis[n] for n in names]


def svd(
    T: Tensor, 
    axis: int | str,
    trunc: Optional[Dict[str, Union[int, float]]] = None
) -> Tuple[Tensor, MutableMapping[BlockKey, torch.Tensor], Tensor]:
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
    tuple[Tensor, MutableMapping[BlockKey, torch.Tensor], Tensor]
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
    
    # Index-level permutation: moves left_axis to position 0.
    # Used both for block-key reordering and as the R-symbol argument for intw.
    perm = [left_axis] + right_axes
    # For non-Abelian tensors each data block has a trailing OM axis that must
    # be kept last.  The array-level permutation appends that axis index.
    perm_with_om = perm + [len(T.indices)]  # only used when intw is not None
    
    # Get indices
    left_index = T.indices[left_axis]
    right_indices = tuple(T.indices[i] for i in right_axes)
    right_itags = tuple(T.itags[i] for i in right_axes)
    
    # Group blocks by left charge for proper general-purpose SVD
    # Structure: q_left -> list of (key, arr_perm, dims_right, mat)
    blocks_by_left_charge: Dict[tuple, List[Tuple[BlockKey, torch.Tensor, Tuple[int, ...], torch.Tensor]]] = {}
    
    for key, arr in T.data.items():
        # Permute array to [left_axis] + right_axes (+ trailing OM axis for SU(2))
        if T.intw is not None:
            arr_perm = torch.permute(arr, perm_with_om)
        else:
            arr_perm = torch.permute(arr, perm)
        
        # Get left charge and right charges
        q_left = key[left_axis]
        
        # Reshape to matrix: (dim_left, prod(dims_right))
        dim_left = arr_perm.shape[0]
        dims_right = arr_perm.shape[1:]
        dim_right_prod = math.prod(dims_right)
        mat = arr_perm.reshape(dim_left, dim_right_prod)
        
        # Group by left charge
        if q_left not in blocks_by_left_charge:
            blocks_by_left_charge[q_left] = []
        blocks_by_left_charge[q_left].append((key, arr_perm, dims_right, mat))
    
    # Perform SVD for each left charge sector by concatenating all blocks with same q_left
    svd_results: Dict[tuple, Tuple[torch.Tensor, torch.Tensor, Dict[BlockKey, torch.Tensor]]] = {}
    bond_charge_dims: Dict[tuple, int] = {}
    
    for q_left, block_list in blocks_by_left_charge.items():
        # Concatenate all matrices with the same left charge horizontally
        mats = [mat for _, _, _, mat in block_list]
        concatenated_mat = torch.cat(mats, dim=1)
        
        # Perform single SVD on concatenated matrix
        U, s, Vh = torch.linalg.svd(concatenated_mat, full_matrices=False)
        
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
        Vh_dict: Dict[BlockKey, torch.Tensor] = {}
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
    U_blocks: Dict[BlockKey, torch.Tensor] = {}
    S_blocks: Dict[BlockKey, torch.Tensor] = {}
    Vh_blocks: Dict[BlockKey, torch.Tensor] = {}
    
    # Promote to appropriate dtype (float for real input, stays as-is for complex)
    target_dtype = torch.promote_types(T.dtype, torch.float32) if T.dtype in [torch.float32, torch.complex64] \
        else torch.promote_types(T.dtype, torch.float64)
    
    for q_left, (U, s, Vh_dict) in svd_results.items():
        # For U tensor: indices (left_index, bond_index)
        # Block key: (q_left, q_left) since bond charge equals left charge
        U_key = (q_left, q_left)
        if T.group.is_abelian:
            U_blocks[U_key] = U
        else:
            # Non-Abelian: add trailing component dimension of 1
            U_blocks[U_key] = U.unsqueeze(-1)
        
        # For S: store singular values as 1D array (memory efficient)
        # Block key: (q_left, q_left)
        S_key = (q_left, q_left)
        S_blocks[S_key] = s.to(dtype=target_dtype)
        
        # For Vh tensor: indices (bond_index.flip(), *right_indices)
        # Each block gets its corresponding Vh from the dictionary
        for key, Vh_reshaped in Vh_dict.items():
            q_right = tuple(key[i] for i in right_axes)
            Vh_key = (q_left,) + q_right
            Vh_blocks[Vh_key] = Vh_reshaped
    
    # Build intertwiners for non-Abelian (SU(2)) tensors
    U_intw: Optional[Dict[BlockKey, dg.Bridge]] = None
    Vh_intw: Optional[Dict[BlockKey, dg.Bridge]] = None
    
    if not T.group.is_abelian:
        # U intertwiner: identity-like Bridge for each (q_left, q_left) block.
        # Same convention as identity() in identity.py: weights[0, 0] = sqrt(irrep_dim).
        U_intw = {}
        for q_left in bond_charge_dims:
            bridge = dg.Bridge.from_block(
                T.group, (q_left, q_left),
                [left_index.direction, bond_index.direction],
                dtype=T.dtype
            )
            bridge.weights[0, 0] = math.sqrt(T.group.irrep_dim(q_left))
            U_intw[(q_left, q_left)] = bridge
        
        # Vd intertwiner: T's intertwiner permuted by perm = [left_axis] + right_axes.
        # Vd's index order is [bond, right_indices_in_original_order], which equals T's
        # index order permuted by perm.  Applying compute_rsymbol(bridge, perm) gives the
        # correctly recoupled Bridge for Vd's edge ordering.
        # For left_axis = 0, perm is the identity and the R-symbol is the identity matrix,
        # so the weights are copied unchanged.
        Vh_intw = {}
        for key, bridge in T.intw.items():
            r_symbol, spec_permuted = dg.compute_rsymbol(bridge, perm)
            new_weights = bridge.weights @ r_symbol.to(dtype=bridge.weights.dtype)
            new_key = tuple(key[i] for i in perm)
            Vh_intw[new_key] = dg.Bridge(cgspec=spec_permuted, weights=new_weights)
    
    # Construct output tensors
    U_tensor = Tensor(
        indices=(left_index, bond_index),
        itags=(T.itags[left_axis], "_bond_L"),
        data=U_blocks, intw=U_intw, dtype=T.dtype
    )
    
    Vh_tensor = Tensor(
        indices=(bond_index.flip(),) + right_indices,
        itags=("_bond_R",) + right_itags,
        data=Vh_blocks, intw=Vh_intw, dtype=T.dtype
    )
    
    return U_tensor, S_blocks, Vh_tensor


def qr(
    T: Tensor,
    axis: int | str
) -> Tuple[Tensor, Tensor]:
    """Perform a symmetry-preserving QR decomposition separating one axis from all others.

    Parameters
    ----------
    T:
        Tensor to be decomposed.
    axis:
        Axis to separate from all others. Can be an integer axis or itag.
        This axis forms the left partition (Q), all others form the right partition (R).

    Returns
    -------
    tuple[Tensor, Tensor]
        Pair `(Q, R)` where:
        - Q has indices (left_index, bond_index) and is orthogonal
        - R has indices (bond_index.flip(), *right_indices) and is upper triangular
    
    Raises
    ------
    ValueError
        If axis specification is invalid or ambiguous.
    
    Notes
    -----
    QR decomposition factors a tensor into an orthogonal matrix Q and an upper triangular
    matrix R such that T = Q @ R. Unlike SVD, no truncation is applied.
    
    The decomposition is performed block-wise, preserving symmetry structure. For each
    charge sector, blocks with the same left charge are concatenated horizontally,
    QR decomposed together, and then R is split back into individual blocks.
    
    Examples
    --------
    >>> # Basic QR decomposition
    >>> Q, R = qr(T, axis=0)
    >>> 
    >>> # Using itag
    >>> Q, R = qr(T, axis="physical")
    """
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
    
    # Group blocks by left charge for proper QR decomposition
    # Structure: q_left -> list of (key, arr_perm, dims_right, mat)
    blocks_by_left_charge: Dict[tuple, List[Tuple[BlockKey, torch.Tensor, Tuple[int, ...], torch.Tensor]]] = {}
    
    for key, arr in T.data.items():
        # Permute array to [left_axis] + right_axes
        arr_perm = torch.permute(arr, perm)
        
        # Get left charge
        q_left = key[left_axis]
        
        # Reshape to matrix: (dim_left, prod(dims_right))
        dim_left = arr_perm.shape[0]
        dims_right = arr_perm.shape[1:]
        dim_right_prod = math.prod(dims_right)
        mat = arr_perm.reshape(dim_left, dim_right_prod)
        
        # Group by left charge
        if q_left not in blocks_by_left_charge:
            blocks_by_left_charge[q_left] = []
        blocks_by_left_charge[q_left].append((key, arr_perm, dims_right, mat))
    
    # Perform QR for each left charge sector by concatenating all blocks with same q_left
    qr_results: Dict[tuple, Tuple[torch.Tensor, Dict[BlockKey, torch.Tensor]]] = {}
    bond_charge_dims: Dict[tuple, int] = {}
    
    for q_left, block_list in blocks_by_left_charge.items():
        # Concatenate all matrices with the same left charge horizontally
        mats = [mat for _, _, _, mat in block_list]
        concatenated_mat = torch.cat(mats, dim=1)
        
        # Perform QR decomposition on concatenated matrix
        Q, R = torch.linalg.qr(concatenated_mat, mode='reduced')
        
        # Split R back to individual blocks
        R_dict: Dict[BlockKey, torch.Tensor] = {}
        col_offset = 0
        for key, arr_perm, dims_right, mat in block_list:
            n_cols = mat.shape[1]
            R_block = R[:, col_offset:col_offset+n_cols]
            # Reshape back to original right dimensions
            rank = R_block.shape[0]
            R_reshaped = R_block.reshape((rank,) + dims_right)
            R_dict[key] = R_reshaped
            col_offset += n_cols
        
        # Store results grouped by left charge
        qr_results[q_left] = (Q, R_dict)
        bond_charge_dims[q_left] = Q.shape[1]
    
    # Build bond index with sectors from left charges
    bond_sectors = tuple(Sector(q, d) for q, d in sorted(bond_charge_dims.items(), key=lambda x: str(x[0])))
    bond_direction = left_index.direction.reverse()
    bond_index = Index(direction=bond_direction, group=left_index.group, sectors=bond_sectors)
    
    # Construct output blocks from grouped QR results
    Q_blocks: Dict[BlockKey, torch.Tensor] = {}
    R_blocks: Dict[BlockKey, torch.Tensor] = {}
    
    for q_left, (Q, R_dict) in qr_results.items():
        # For Q tensor: indices (left_index, bond_index)
        # Block key: (q_left, q_left) since bond charge equals left charge
        Q_key = (q_left, q_left)
        Q_blocks[Q_key] = Q
        
        # For R tensor: indices (bond_index.flip(), *right_indices)
        # Each block gets its corresponding R from the dictionary
        for key, R_reshaped in R_dict.items():
            q_right = tuple(key[i] for i in right_axes)
            R_key = (q_left,) + q_right
            R_blocks[R_key] = R_reshaped
    
    # Construct output tensors
    Q_tensor = Tensor(
        indices=(left_index, bond_index),
        itags=(T.itags[left_axis], "_bond"),
        data=Q_blocks,
        dtype=T.dtype
    )
    
    R_tensor = Tensor(
        indices=(bond_index.flip(),) + right_indices,
        itags=("_bond",) + right_itags,
        data=R_blocks,
        dtype=T.dtype
    )
    
    return Q_tensor, R_tensor


def eig(
    T: Tensor,
    itag: Optional[str] = None,
    order: Literal["ascend", "descend"] = "ascend",
    trunc: Optional[Dict[str, Union[int, float]]] = None
) -> Tuple[Tensor, MutableMapping[BlockKey, torch.Tensor]]:
    """Perform eigenvalue decomposition of a square matrix tensor.

    Parameters
    ----------
    T:
        Square matrix tensor to be decomposed. Must have exactly 2 indices
        with matching charge structure (opposite directions).
    itag:
        Index tag for the bond dimension. If None, uses default tag "_bond_eig".
    order:
        Sorting order for eigenvalues. Either "ascend" for ascending order (smallest
        to largest) or "descend" for descending order (largest to smallest). Default
        is "ascend". For real eigenvalues, sorts by value (e.g., -5 < -3 < 1 < 2).
        For complex eigenvalues, sorts by real part. Sorting is applied per block and
        affects both truncation modes.
    trunc:
        Truncation specification as a dict. If None, no truncation. Supported keys:
        - "nkeep": Keep at most n eigenvalues globally. With order="descend", keeps
          the n largest (most positive) eigenvalues. With order="ascend", keeps the
          n smallest (most negative) eigenvalues.
        - "thresh": Keep eigenvalues relative to threshold t per block. With
          order="descend", keeps eigenvalues >= t. With order="ascend", keeps
          eigenvalues <= t.
        Both can be specified together: thresh is applied first, then nkeep.

    Returns
    -------
    tuple[Tensor, MutableMapping[BlockKey, torch.Tensor]]
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
    
    Sorting behavior:
    - For real eigenvalues: sorts by actual value (e.g., -5 < -3 < 1 < 2)
    - For complex eigenvalues: sorts by real part
    - Use order="ascend" to get smallest/most negative eigenvalues first (e.g., ground states)
    - Use order="descend" to get largest/most positive eigenvalues first
    
    For "nkeep" mode, truncation respects the order parameter: with order="descend",
    keeps the n largest eigenvalues; with order="ascend", keeps the n smallest.
    For "thresh" mode, truncation is also order-aware: with order="descend", keeps
    eigenvalues >= threshold (most positive); with order="ascend", keeps eigenvalues
    <= threshold (most negative). Threshold filtering is applied per block.
    
    When both modes are specified, "thresh" is applied first (per-block filtering),
    then "nkeep" is applied globally to the remaining eigenvalues.
    
    The eigenvectors are stored in columns of U, normalized such that U is unitary
    (or as close as the eigendecomposition provides).
    
    Examples
    --------
    >>> # No truncation, ascending order (default - smallest eigenvalues first)
    >>> U, D_blocks = eig(T)
    >>> 
    >>> # Descending order (largest eigenvalues first)
    >>> U, D_blocks = eig(T, order="descend")
    >>> 
    >>> # Keep 5 smallest (most negative) eigenvalues - useful for ground states
    >>> U, D_blocks = eig(T, order="ascend", trunc={"nkeep": 5})
    >>> 
    >>> # Keep 5 largest (most positive) eigenvalues
    >>> U, D_blocks = eig(T, order="descend", trunc={"nkeep": 5})
    >>> 
    >>> # Keep eigenvalues >= 0.1 (positive eigenvalues above threshold)
    >>> U, D_blocks = eig(T, order="descend", trunc={"thresh": 0.1})
    >>> 
    >>> # Keep eigenvalues <= -0.5 (negative eigenvalues below threshold)
    >>> U, D_blocks = eig(T, order="ascend", trunc={"thresh": -0.5})
    >>> 
    >>> # Apply both: keep eigenvalues >= 0.1, then keep top 5
    >>> U, D_blocks = eig(T, order="descend", trunc={"thresh": 0.1, "nkeep": 5})
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
    eig_results: Dict[tuple, Tuple[torch.Tensor, torch.Tensor]] = {}
    bond_charge_dims: Dict[tuple, int] = {}
    
    for key, arr in T.data.items():
        q_row = key[0]
        # Note: charge conservation ensures q_row == key[1] for square matrices
        
        # Perform eigendecomposition
        # torch.linalg.eig returns (eigenvalues, eigenvectors)
        # eigenvectors[:, i] is the eigenvector for eigenvalues[i]
        eigenvalues, eigenvectors = torch.linalg.eig(arr)
        
        # Sort eigenvalues according to order parameter
        # For real eigenvalues, sorts by value; for complex, sorts by real part
        sort_indices = torch.argsort(eigenvalues.real)
        if order == "descend":
            sort_indices = torch.flip(sort_indices, dims=[0])
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # Apply per-block truncation for thresh mode
        if trunc is not None and "thresh" in trunc:
            # thresh behavior depends on order parameter:
            # - order="descend": keep eigenvalues >= thresh (largest/most positive)
            # - order="ascend": keep eigenvalues <= thresh (smallest/most negative)
            if order == "descend":
                keep_mask = eigenvalues.real >= trunc["thresh"]
            else:  # order == "ascend"
                keep_mask = eigenvalues.real <= trunc["thresh"]
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
                all_eigenvalues.append((val.real.item(), q, i))
        
        # Keep nkeep eigenvalues according to order parameter
        # For "descend": keep largest (most positive) eigenvalues
        # For "ascend": keep smallest (most negative) eigenvalues
        all_eigenvalues.sort(key=lambda x: x[0], reverse=(order == "descend"))
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
    U_blocks: Dict[BlockKey, torch.Tensor] = {}
    D_blocks: Dict[BlockKey, torch.Tensor] = {}
    
    # Determine if eigenvalues and eigenvectors are actually complex
    # If all eigenvalues and eigenvectors are real, we can use real dtype
    all_real = all(
        (torch.allclose(eigvals.imag, torch.zeros_like(eigvals.imag)) if eigvals.is_complex() else True) and 
        (torch.allclose(eigvecs.imag, torch.zeros_like(eigvecs.imag)) if eigvecs.is_complex() else True)
        for eigvecs, eigvals in eig_results.values()
    )
    
    # Choose appropriate dtype
    if all_real:
        # Eigenvalues and eigenvectors are real
        D_dtype = torch.promote_types(T.dtype, torch.float32) if T.dtype == torch.float32 \
            else torch.promote_types(T.dtype, torch.float64)
        U_dtype = D_dtype
    else:
        # Complex eigenvalues/eigenvectors
        D_dtype = torch.promote_types(T.dtype, torch.complex128)
        U_dtype = D_dtype
    
    for q, (eigvecs, eigvals) in eig_results.items():
        # For U tensor: indices (row_index, bond_index)
        # Block key: (q, q) since bond charge equals row charge
        U_key = (q, q)
        # When all_real is True, explicitly take real part to avoid casting warning
        U_blocks[U_key] = eigvecs.real.to(dtype=U_dtype) if all_real else eigvecs.to(dtype=U_dtype)
        
        # For D: store eigenvalues as 1D array (memory efficient)
        # Block key: (q, q)
        D_key = (q, q)
        # When all_real is True, explicitly take real part to avoid casting warning
        D_blocks[D_key] = eigvals.real.to(dtype=D_dtype) if all_real else eigvals.to(dtype=D_dtype)
    
    # Construct output tensor
    U_tensor = Tensor(
        indices=(row_index, bond_index),
        itags=(T.itags[0], bond_tag),
        data=U_blocks,
        dtype=U_dtype
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
        - "SVD": Returns (U, S, Vh) where S is diagonal matrix tensor (full SVD)
        - "UR": Returns (U, R) where R = S*Vh (singular values multiplied into Vh)
        - "LV": Returns (L, V) where L = U*S (singular values multiplied into U)
        - "QR": Returns (Q, R) where Q is orthogonal and R is upper triangular
    flow:
        Arrow direction control. Default is "><" (both arrows incoming).
        - For SVD mode: Controls S matrix arrow directions ("><", ">>", or "<<")
        - For UR mode: Both ">>" and "><" normalize to ">>" (outward bonds); "<<" is also accepted
        - For LV mode: Both "<<" and "><" normalize to "<<" (inward bonds); ">>" is also accepted
        - For QR mode: Controls bond arrow directions
        Note: The underlying svd naturally produces ">>" or "<<" depending on left_index.direction.
        This parameter uses capcup() to adjust from the natural flow to the desired flow.
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
        - "SVD" mode: (U, S, Vh) with S as diagonal matrix tensor
        - "UR" mode: (U, R) where R incorporates singular values
        - "LV" mode: (L, V) where L incorporates singular values
        - "QR" mode: (Q, R) where Q is orthogonal and R is upper triangular
    
    Raises
    ------
    ValueError
        If mode is not one of "SVD", "UR", "LV", or "QR",
        or if flow is not ">>", "<<", or "><"
    
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
    # Import here to avoid circular dependency
    from .maneuver import capcup, merge_axes
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
            U_unmerged.trim_zero_blocks()
            return U_unmerged, S, Vh
        else:  # mode == "UR", "LV", or "QR"
            first, second = result
            # For UR mode, first is U; for LV mode, first is L; for QR mode, first is Q
            # All have the merged index at position 0
            # The merged index is at position 0 of first, and at last position of iso_conj
            iso_conj_last_idx = len(iso_conj.indices) - 1
            first_unmerged = contract(iso_conj, first, axes=(iso_conj_last_idx, 0))
            first_unmerged.trim_zero_blocks()
            return first_unmerged, second
    
    # Single axis: original behavior
    # Validate mode
    mode = mode.upper()
    if mode not in ("SVD", "UR", "LV", "QR"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'SVD', 'UR', 'LV', or 'QR'")
    
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
        
        target_dtype = torch.promote_types(T.dtype, torch.float32) if T.dtype in [torch.float32, torch.complex64] else torch.promote_types(T.dtype, torch.float64)
        
        S_diag_blocks: Dict[BlockKey, torch.Tensor] = {}
        S_intw: Optional[Dict[BlockKey, dg.Bridge]] = None
        
        if T.group.is_abelian:
            for key, s_array in S_blocks.items():
                S_diag_blocks[key] = torch.diag(s_array)
        else:
            # Non-Abelian: diagonal matrix with trailing component dimension of 1.
            # Intertwiner is identity-like: weights[0, 0] = sqrt(irrep_dim(q)).
            S_intw = {}
            bond_flip_index = bond_index.flip()
            for key, s_array in S_blocks.items():
                S_diag_blocks[key] = torch.diag(s_array).unsqueeze(-1)
                q = key[0]
                bridge = dg.Bridge.from_block(
                    T.group, key, [bond_flip_index.direction, bond_index.direction], dtype=target_dtype
                )
                bridge.weights[0, 0] = math.sqrt(T.group.irrep_dim(q))
                S_intw[key] = bridge
        
        # Natural S has indices matching the natural flow from svd
        S_tensor = Tensor(
            indices=(bond_index.flip(), bond_index),
            itags=(bond_tag_left, bond_tag_right),
            data=S_diag_blocks, intw=S_intw, dtype=target_dtype, label="Diagonal"
        )
        
        # Convert from natural_flow to desired flow by inverting contraction bonds.
        # capcup preserves U⊗S⊗Vh = T across the flow change.
        if natural_flow == ">>":
            # Natural for S: (IN, OUT)
            if flow == "><":
                # Desired: (IN, IN) - invert S-Vh bond
                capcup(S_tensor, 1, Vh, 0)
            elif flow == "<<":
                # Desired: (OUT, IN) - invert U-S bond then S-Vh bond
                capcup(S_tensor, 0, U, 1)
                capcup(S_tensor, 1, Vh, 0)
            # else flow == ">>": natural, no change needed
        else:  # natural_flow == "<<"
            # Natural for S: (OUT, IN)
            if flow == "><":
                # Desired: (IN, IN) - invert U-S bond
                capcup(S_tensor, 0, U, 1)
            elif flow == ">>":
                # Desired: (IN, OUT) - invert U-S bond then S-Vh bond
                capcup(S_tensor, 0, U, 1)
                capcup(S_tensor, 1, Vh, 0)
            # else flow == "<<": natural, no change needed
        
        return U, S_tensor, Vh
    
    elif mode == "UR":
        # Multiply singular values into Vh to get R = S*Vh
        R_blocks: Dict[BlockKey, torch.Tensor] = {}
        
        for key, vh_block in Vh.data.items():
            # key = (q_bond, *q_right)
            # S_blocks key = (q_bond, q_bond)
            q_bond = key[0]
            s_key = (q_bond, q_bond)
            
            if s_key in S_blocks:
                s_array = S_blocks[s_key]
                # Multiply: R = diag(s) @ Vh = s[:, None, ...] * Vh
                # Abelian vh_block shape: (rank, *right_dims)
                # SU(2) vh_block shape:  (rank, *right_dims, k)  — trailing k is treated
                # the same way: reshape s to (rank, 1, ..., 1) and broadcast.
                rank = len(s_array)
                s_broadcasted = s_array.reshape((rank,) + (1,) * (vh_block.ndim - 1))
                R_blocks[key] = (s_broadcasted * vh_block).to(dtype=T.dtype)
            else:
                # No singular values for this block (shouldn't happen normally)
                R_blocks[key] = vh_block
        
        # Change bond tag to match U's bond tag for easier contraction
        R_itags = (bond_tag_left,) + Vh.itags[1:]
        
        # R inherits Vh's bond index structure and intertwiner
        R_tensor = Tensor(
            indices=Vh.indices, itags=R_itags, data=R_blocks,
            intw=Vh.intw, dtype=T.dtype
        )
        
        # For UR mode: normalize flow (both ">>" and "><" mean ">>")
        normalized_flow = ">>" if flow in (">>", "><") else "<<"
        
        # Invert U-R contraction bond if normalized flow differs from natural flow
        if normalized_flow != natural_flow:
            capcup(U, 1, R_tensor, 0)
        
        return U, R_tensor
    
    elif mode == "LV":
        # Multiply singular values into U to get L = U*S
        L_blocks: Dict[BlockKey, torch.Tensor] = {}
        
        for key, u_block in U.data.items():
            # key = (q_left, q_bond)
            # S_blocks key = (q_bond, q_bond)
            q_bond = key[1]
            s_key = (q_bond, q_bond)
            
            if s_key in S_blocks:
                s_array = S_blocks[s_key]
                # Multiply: L = U @ diag(s) = U * s[None, :]
                # Abelian u_block shape: (dim_left, rank)       — s_array[None, :] broadcasts
                # SU(2) u_block shape:   (dim_left, rank, 1)    — need (1, rank, 1) to match
                # General: reshape s to (1, rank, 1, ..., 1) with u_block.ndim - 2 trailing ones.
                rank = len(s_array)
                s_broadcasted = s_array.reshape((1, rank) + (1,) * (u_block.ndim - 2))
                L_blocks[key] = (u_block * s_broadcasted).to(dtype=T.dtype)
            else:
                # No singular values for this block (shouldn't happen normally)
                L_blocks[key] = u_block
        
        # Change bond tag to match Vh's bond tag for easier contraction
        L_itags = (U.itags[0], bond_tag_right)
        
        # L inherits U's bond index structure and intertwiner
        L_tensor = Tensor(
            indices=U.indices, itags=L_itags, data=L_blocks,
            intw=U.intw, dtype=T.dtype
        )
        
        # For LV mode: normalize flow (both "<<" and "><" mean "<<")
        normalized_flow = "<<" if flow in ("<<", "><") else ">>"
        
        # Invert L-Vh contraction bond if normalized flow differs from natural flow
        if normalized_flow != natural_flow:
            capcup(L_tensor, 1, Vh, 0)
        
        return L_tensor, Vh
    
    elif mode == "QR":
        # Use the qr function directly
        # axes go into Q, remaining axes go into R
        Q, R = qr(T, axis_idx)
        
        # Update bond tags to use unified tag (or custom tags if specified)
        # If itag is a string (same tag for both), use it; otherwise use separate tags
        if isinstance(itag, str):
            unified_tag = itag
        else:
            # For QR, we use bond_tag_left for both to allow automatic contraction
            unified_tag = bond_tag_left
        
        Q.retag({Q.itags[1]: unified_tag})
        R.retag({R.itags[0]: unified_tag})
        
        # For QR mode: natural flow depends on left_index direction
        # Similar to UR mode, normalize flow (both ">>" and "><" mean ">>")
        normalized_flow = ">>" if flow in (">>", "><") else "<<"
        
        # Determine natural flow from Q's left index (same as SVD logic)
        q_left_direction = Q.indices[0].direction
        # If Q's left index is OUT, natural flow is "<<"; if IN, natural flow is ">>"
        qr_natural_flow = "<<" if q_left_direction == Direction.OUT else ">>"
        
        # Invert Q-R contraction bond if normalized flow differs from natural flow
        if normalized_flow != qr_natural_flow:
            capcup(Q, 1, R, 0)
        
        return Q, R
