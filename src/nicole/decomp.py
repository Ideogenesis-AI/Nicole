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
value decomposition (SVD) components. The `svd` function implements a
general-purpose symmetry-preserving SVD that separates a single tensor axis
from all others, returning properly structured U, S, and Vh tensors with
charge-conserving bond indices.
"""

from typing import Dict, List, Sequence, Tuple

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


def svd(T: Tensor, axis: int | str) -> Tuple[Tensor, Tensor, Tensor]:
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
    tuple[Tensor, Tensor, Tensor]
        Triplet `(U, S, Vh)` where:
        - U has indices (left_index, bond_index)
        - S has indices (bond_index.flip(), bond_index) with diagonal singular value matrices
        - Vh has indices (bond_index.flip(), *right_indices)
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
        
        # For S tensor: indices (bond_index.flip(), bond_index)
        # Block key: (q_left, q_left) as diagonal matrix
        S_key = (q_left, q_left)
        S_blocks[S_key] = np.diag(s).astype(np.result_type(T.dtype, float))
        
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
    
    S_tensor = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("_bond_L", "_bond_R"),
        data=S_blocks,
        dtype=np.result_type(T.dtype, float)
    )
    
    Vh_tensor = Tensor(
        indices=(bond_index.flip(),) + right_indices,
        itags=("_bond_R",) + right_itags,
        data=Vh_blocks,
        dtype=T.dtype
    )
    
    return U_tensor, S_tensor, Vh_tensor
