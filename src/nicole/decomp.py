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

"""Decomposition utilities for symmetry-aware Nicole (TN) tensors.

This module provides functions for decomposing tensors into their singular
value decomposition (SVD) components.

Functions
---------
svd(T, axis)
    Low-level SVD returning U tensor, singular values dict, and Vh tensor.
    Returns singular values as 1D arrays for memory efficiency.

decomp(T, axis, mode)
    High-level decomposition with three modes:
    - "UR": Returns (U, R) where R = S*Vh
    - "SVD": Returns (U, S, Vh) with S as diagonal matrix tensor
    - "LV": Returns (L, V) where L = U*S
"""

from typing import Dict, List, MutableMapping, Sequence, Tuple, Union

import numpy as np

from .blocks import BlockKey
from .index import Index
from .tensor import Tensor
from .typing import Sector


def _axes_from_names(itags: Sequence[str], names: Sequence[str]) -> List[int]:
    """Translate index tags into positional axis indices."""
    # Build a lookup once to avoid repeated linear searches.
    name_to_axis = {tag: i for i, tag in enumerate(itags)}
    return [name_to_axis[n] for n in names]


def svd(T: Tensor, axis: int | str) -> Tuple[Tensor, MutableMapping[BlockKey, np.ndarray], Tensor]:
    """Perform a symmetry-preserving SVD separating one axis from all others.

    Parameters
    ----------
    T:
        Tensor to be decomposed.
    axis:
        Index to separate from all others. Can be an integer position or index tag.
        This axis forms the left partition, all others form the right partition.

    Returns
    -------
    tuple[Tensor, MutableMapping[BlockKey, np.ndarray], Tensor]
        Triplet `(U, S_blocks, Vh)` where:
        - U has indices (left_index, bond_index)
        - S_blocks is a Dict mapping block keys to 1D arrays of singular values
        - Vh has indices (bond_index.flip(), *right_indices)
    
    Notes
    -----
    The singular values are returned as 1D arrays for memory efficiency.
    Use the `decomp()` function with mode="SVD" if you need S as a diagonal matrix tensor.
    """
    # Parse axis parameter into integer index
    if isinstance(axis, str):
        try:
            axis_idx = T.itags.index(axis)
        except ValueError:
            raise ValueError(f"Index tag '{axis}' not found in tensor")
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


def decomp(
    T: Tensor,
    axis: int | str,
    mode: str = "SVD"
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
    """Perform tensor decomposition with flexible output modes.
    
    Parameters
    ----------
    T:
        Tensor to be decomposed.
    axis:
        Index to separate from all others. Can be integer position or index tag.
    mode:
        Decomposition mode:
        - "UR": Returns (U, R) where R = S*Vh (singular values multiplied into Vh)
        - "SVD": Returns (U, S, Vh) where S is diagonal matrix tensor (full SVD)
        - "LV": Returns (L, V) where L = U*S (singular values multiplied into U)
    
    Returns
    -------
    tuple[Tensor, Tensor] or tuple[Tensor, Tensor, Tensor]
        - "UR" mode: (U, R) where R incorporates singular values
        - "SVD" mode: (U, S, Vh) with S as diagonal matrix tensor
        - "LV" mode: (L, V) where L incorporates singular values
    
    Raises
    ------
    ValueError
        If mode is not one of "UR", "SVD", or "LV"
    
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
    >>> U, R = decomp(T, axis=0, mode="UR")
    >>> 
    >>> # SVD mode: Get full SVD with diagonal S
    >>> U, S, Vh = decomp(T, axis=0, mode="SVD")
    >>> 
    >>> # LV mode: Get L=U*S and V
    >>> L, V = decomp(T, axis=0, mode="LV")
    
    Notes
    -----
    - UR and LV modes are more memory and computationally efficient than SVD mode
    - SVD mode constructs a full diagonal matrix tensor for S
    - All modes produce mathematically equivalent decompositions
    """
    # Validate mode
    mode = mode.upper()
    if mode not in ("UR", "SVD", "LV"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'UR', 'SVD', or 'LV'")
    
    # Perform SVD to get U, singular values dict, and Vh
    U, S_blocks, Vh = svd(T, axis)
    
    if mode == "SVD":
        # Construct full diagonal S tensor
        bond_index = U.indices[1]  # Extract bond index from U
        
        S_diag_blocks: Dict[BlockKey, np.ndarray] = {}
        for key, s_array in S_blocks.items():
            # Convert 1D singular values to diagonal matrix
            S_diag_blocks[key] = np.diag(s_array)
        
        S_tensor = Tensor(
            indices=(bond_index.flip(), bond_index),
            itags=("_bond_L", "_bond_R"),
            data=S_diag_blocks,
            dtype=np.result_type(T.dtype, float)
        )
        
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
        R_itags = ("_bond_L",) + Vh.itags[1:]
        
        R_tensor = Tensor(
            indices=Vh.indices,
            itags=R_itags,
            data=R_blocks,
            dtype=T.dtype
        )
        
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
        L_itags = (U.itags[0], "_bond_R")
        
        L_tensor = Tensor(
            indices=U.indices,
            itags=L_itags,
            data=L_blocks,
            dtype=T.dtype
        )
        
        return L_tensor, Vh
