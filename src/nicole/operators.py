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
merge_axes(tensor, axes, merged_tag=None, direction=OUT)
    Merge multiple tensor axes into one using isometry fusion, returning both
    the merged tensor and conjugate isometry for potential unfusing.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .blocks import BlockKey
from .index import Index
from .tensor import Tensor
from .typing import Charge, Direction, Sector


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
        Sequence of integer axes specifying the new ordering. Must be a
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
    >>> # Assuming t is a 3-index tensor with itags [a, b, c]
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
        Optional integer axes specifying the new ordering. If not provided,
        defaults to reversing the axis order.
    
    Returns
    -------
    Tensor
        A new tensor instance with transposed axes.
    
    Examples
    --------
    >>> from nicole import transpose, Tensor
    >>> # Assuming t is a 3-index tensor with itags [a, b, c]
    >>> t_T = transpose(t)  # Reverse order to [c, b, a]
    >>> t_T2 = transpose(t, 1, 0, 2)  # Swap first two to [b, a, c]
    """
    if not order:
        order = tuple(reversed(range(len(tensor.indices))))
    return permute(tensor, order)


def subsector(tensor: Tensor, block_indices: Union[int, Sequence[int]]) -> Tensor:
    """Return a new tensor containing only the specified blocks with pruned sectors.
    
    Parameters
    ----------
    tensor:
        The input tensor to extract blocks from.
    block_indices:
        Block index or sequence of block indices (1-indexed, matching display numbering)
        specifying which blocks to include in the new tensor. Can be a single integer
        or a sequence of integers.
    
    Returns
    -------
    Tensor
        A new tensor instance containing only the specified blocks with unused sectors
        removed from the indices. Other attributes (itags, dtype, label) are preserved.
    
    Raises
    ------
    IndexError
        If any block index is out of range.
    
    Examples
    --------
    >>> from nicole import subsector, Tensor
    >>> # Assuming t has 5 blocks numbered 1-5 in display
    >>> t_sub = subsector(t, [1, 3, 5])  # Extract blocks 1, 3, and 5
    >>> t_single = subsector(t, 2)  # Extract just block 2
    """
    # Convert single integer to sequence
    if isinstance(block_indices, int):
        block_indices = [block_indices]
    
    num_blocks = len(tensor.data)
    for i in block_indices:
        if i < 1 or i > num_blocks:
            raise IndexError(f"Block index {i} out of range [1, {num_blocks}]")
    
    new_data = {tensor.key(i): tensor.block(i).copy() for i in block_indices}
    
    # Prune unused sectors from indices
    pruned_indices = Tensor._prune_unused_sectors(tensor.indices, new_data)
    
    return Tensor(
        indices=pruned_indices,
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


def diag(
    S_blocks: Dict[BlockKey, np.ndarray],
    bond_index: Index,
    itags: Optional[Tuple[str, str]] = None,
    dtype: Optional[np.dtype] = None
) -> Tensor:
    """Convert diagonal blocks (from SVD or eig) into a diagonal matrix tensor.
    
    Takes a dictionary of 1D arrays (such as singular values from SVD or eigenvalues
    from eig) and creates a diagonal matrix tensor where each 1D array becomes a
    diagonal matrix block.
    
    Parameters
    ----------
    S_blocks : dict[BlockKey, np.ndarray]
        Dictionary mapping block keys to 1D arrays. Each array contains the diagonal
        elements for that block. Typically from the S output of svd() or D from eig().
    bond_index : Index
        The bond index defining the sectors and dimensions. Both output indices will
        be based on this index (one normal, one flipped).
    itags : tuple of two str, optional
        Custom itags for the two output indices. If None, uses ("_bond_L", "_bond_R").
        Default: None.
    dtype : np.dtype, optional
        Data type for the output tensor. If None, inferred from input arrays.
        Default: None.
    
    Returns
    -------
    Tensor
        Diagonal matrix tensor with two indices (bond_index.flip(), bond_index).
        Label is set to "Diagonal".
    
    Raises
    ------
    ValueError
        If any data block is not 1-dimensional.
    
    Examples
    --------
    >>> from nicole import Tensor, Index, Sector, Direction, U1Group, decomp
    >>> import numpy as np
    >>> # Perform SVD
    >>> T = Tensor.random([idx_i, idx_j], itags=["i", "j"])
    >>> U, S_blocks, Vh = decomp(T, axes=0, mode="UR")  # Get S as dict
    >>> 
    >>> # Convert S_blocks to diagonal matrix
    >>> from nicole import diag
    >>> S_diag = diag(S_blocks, U.indices[1], itags=("left", "right"))
    >>> S_diag.itags
    ('left', 'right')
    >>> S_diag.label
    'Diagonal'
    
    >>> # Can now use S_diag in contractions
    >>> result = contract(U, S_diag)  # Equivalent to U @ S
    
    Notes
    -----
    This function is useful for converting the singular values dict S from svd() or
    eigenvalues dict D from eig() into full diagonal matrix form for explicit matrix
    operations like contraction.
    
    The output tensor will have:
    - Two indices: (bond_index.flip(), bond_index)
    - Block keys (q, q) for each charge q in S_blocks
    - Diagonal matrices as data blocks
    - Label "Diagonal" (overriding default "Tensor")
    """
    # Validate all blocks are 1D
    for key, arr in S_blocks.items():
        if arr.ndim != 1:
            raise ValueError(
                f"diag requires all data blocks to be 1-dimensional, "
                f"but block {key} has shape {arr.shape}"
            )
    
    # Determine output itags
    if itags is None:
        out_itags = ("_bond_L", "_bond_R")
    else:
        if not isinstance(itags, tuple) or len(itags) != 2:
            length = len(itags) if isinstance(itags, (tuple, list)) else 'N/A'
            raise ValueError(
                f"itags must be a tuple of two strings, got {type(itags)} with length {length}"
            )
        out_itags = itags
    
    # Determine dtype
    if dtype is None:
        # Infer from first block
        if S_blocks:
            sample_arr = next(iter(S_blocks.values()))
            dtype = sample_arr.dtype
        else:
            dtype = np.float64
    
    # Convert each 1D block to diagonal matrix
    diag_blocks: Dict[BlockKey, np.ndarray] = {}
    for key, vec_array in S_blocks.items():
        # Create diagonal matrix from 1D array
        diag_matrix = np.diag(vec_array)
        diag_blocks[key] = diag_matrix
    
    # Create output tensor with two indices
    return Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=out_itags,
        data=diag_blocks,
        dtype=dtype,
        label="Diagonal"
    )


def inv(tensor: Tensor) -> Tensor:
    """Invert a diagonal matrix tensor.
    
    Takes a diagonal matrix tensor and returns its inverse by inverting each
    diagonal element. For charge conservation, the input tensor must have
    opposite index directions.
    
    Parameters
    ----------
    tensor : Tensor
        Input diagonal matrix tensor with exactly 2 indices. If both indices have
        the same direction, they will be flipped automatically. If labeled "Diagonal",
        the diagonal structure check is skipped.
    
    Returns
    -------
    Tensor
        Inverted diagonal matrix with the same structure as input.
    
    Raises
    ------
    ValueError
        If tensor does not have exactly 2 indices, or if tensor is not diagonal
        (when label != "Diagonal").
    ZeroDivisionError
        If any diagonal element is zero (within machine epsilon).
    
    Examples
    --------
    >>> from nicole import Tensor, Index, Sector, Direction, U1Group, diag
    >>> from nicole.decomp import svd
    >>> import numpy as np
    >>> # Create a diagonal tensor from SVD
    >>> T = Tensor.random([idx_i, idx_j], itags=["i", "j"])
    >>> U, S_blocks, Vh = svd(T, axis=0)
    >>> S_diag = diag(S_blocks, U.indices[1])
    >>> 
    >>> # Invert the diagonal matrix
    >>> S_inv = inv(S_diag)
    >>> 
    >>> # Verify: S @ S_inv should give identity
    >>> from nicole import contract
    >>> result = contract(S_diag, S_inv)
    
    >>> # Manual diagonal tensor
    >>> idx = Index(Direction.IN, U1Group(), (Sector(0, 2),))
    >>> D = Tensor(
    ...     indices=(idx.flip(), idx),
    ...     itags=("i", "j"),
    ...     data={(0, 0): np.diag([2.0, 4.0])},
    ...     label="Diagonal"
    ... )
    >>> D_inv = inv(D)
    >>> D_inv.data[(0, 0)]
    array([[0.5 , 0.  ],
           [0.  , 0.25]])
    
    Notes
    -----
    The function inverts each diagonal matrix block independently by computing
    1/x for each diagonal element. Off-diagonal elements are assumed to be zero
    (only checked if label != "Diagonal").
    
    For numerical stability, elements with absolute value below machine epsilon
    for float64 will raise ZeroDivisionError.
    """
    # Validate tensor has exactly 2 indices
    if len(tensor.indices) != 2:
        raise ValueError(
            f"inv requires a tensor with exactly 2 indices, got {len(tensor.indices)}"
        )
    
    # Check diagonal structure (skip if labeled "Diagonal")
    eps = np.finfo(np.float64).eps
    if tensor.label != "Diagonal":
        for key, block in tensor.data.items():
            if block.ndim != 2 or block.shape[0] != block.shape[1]:
                raise ValueError(
                    f"inv requires square matrix blocks, but block {key} has shape {block.shape}"
                )
            # Check off-diagonal elements are zero
            off_diag = block - np.diag(np.diag(block))
            if np.max(np.abs(off_diag)) > eps:
                raise ValueError(
                    f"inv requires diagonal matrices, but block {key} has non-zero "
                    f"off-diagonal elements (max: {np.max(np.abs(off_diag))})"
                )
    
    # Invert each diagonal block and transpose by swapping block keys
    inv_blocks: Dict[BlockKey, np.ndarray] = {}
    
    for key, block in tensor.data.items():
        # Extract diagonal elements
        diag_elements = np.diag(block)
        
        # Check for zeros
        if np.any(np.abs(diag_elements) < eps):
            zero_indices = np.where(np.abs(diag_elements) < eps)[0]
            raise ZeroDivisionError(
                f"Cannot invert diagonal matrix: block {key} has zero elements "
                f"at diagonal positions {zero_indices.tolist()}"
            )
        
        # Invert diagonal elements
        inv_diag = 1.0 / diag_elements
        
        # Swap block keys for transpose
        swapped_key = (key[1], key[0])
        inv_blocks[swapped_key] = np.diag(inv_diag)
    
    # Always swap and flip indices (transpose)
    result_indices = (tensor.indices[1].flip(), tensor.indices[0].flip())
    result_itags = (tensor.itags[1], tensor.itags[0])
    
    # Create inverted tensor
    return Tensor(
        indices=result_indices,
        itags=result_itags,
        data=inv_blocks,
        dtype=tensor.dtype,
        label=tensor.label
    )


def merge_axes(
    tensor: Tensor,
    axes: Sequence[Union[int, str]],
    *,
    merged_tag: Optional[str] = None,
    direction: Direction = Direction.OUT,
) -> Tuple[Tensor, Tensor]:
    """Merge multiple tensor axes into a single axis using isometry fusion.

    This function creates an n-to-1 isometry that fuses the specified axes,
    contracts it with the input tensor to perform the merging, and returns
    both the merged tensor and the conjugate of the isometry (which can be
    used to unfuse the axis later).

    Parameters
    ----------
    tensor:
        Input tensor whose axes should be merged.
    axes:
        Sequence of axes to merge. Each element can be either an integer
        position (0-indexed) or a string itag. Must specify at least 2 axes.
    merged_tag:
        Optional tag for the merged axis in the result. If None, uses "_merged_".
    direction:
        Direction for the merged axis. Defaults to Direction.OUT.

    Returns
    -------
    merged_tensor:
        Tensor with the specified axes merged into a single axis. The merged
        axis appears first, followed by the remaining unmerged axes in their
        original order.
    isometry_conj:
        Conjugate of the isometry used for merging. Can be contracted with
        the merged tensor to unfuse the axis back to the original indices.

    Raises
    ------
    ValueError:
        If fewer than 2 axes are specified, if axis specifications are invalid,
        or if axes don't exist in the tensor.
    TypeError:
        If axis specifications are neither int nor str.

    Examples
    --------
    Merge three axes of a tensor:

    >>> tensor = Tensor.random([idx1, idx2, idx3, idx4], itags=['a', 'b', 'c', 'd'])
    >>> merged, iso_conj = merge_axes(tensor, ['a', 'b', 'c'], merged_tag='abc')
    >>> merged.itags  # Merged index 'abc' appears first
    ('abc', 'd')

    Merge using integer positions:

    >>> merged, iso_conj = merge_axes(tensor, [0, 1, 2], merged_tag='merged')

    Unfuse the merged axis (approximately recover original):

    >>> from nicole import contract
    >>> unmerged = contract(merged, iso_conj)  # Should match original structure

    Notes
    -----
    - The merged axis will have direction opposite to what natural fusion produces
      unless specified otherwise via the `direction` parameter
    - The isometry conjugate has all its indices flipped, making it suitable for
      contracting with the merged tensor to reverse the operation
    - Axes are merged in the order they appear in the tensor, not the order
      specified in the `axes` parameter
    """
    # Import here to avoid circular dependency
    from .contract import contract
    from .identity import isometry_n

    # Validate input
    if len(axes) < 2:
        raise ValueError(f"Need at least 2 axes to merge, got {len(axes)}")

    # Check for duplicates in input
    if len(axes) != len(set(axes)):
        raise ValueError("Duplicate axes specified - there are ambiguities in the provided itags")

    # Convert axes to integer positions
    positions = []
    for ax in axes:
        if isinstance(ax, int):
            if ax < 0 or ax >= len(tensor.indices):
                raise ValueError(
                    f"Axis position {ax} out of range [0, {len(tensor.indices)})"
                )
            positions.append(ax)
        elif isinstance(ax, str):
            if ax not in tensor.itags:
                raise ValueError(f"Axis tag '{ax}' not found in tensor tags {tensor.itags}")
            positions.append(tensor.itags.index(ax))
        else:
            raise TypeError(f"Axis must be int or str, got {type(ax)}")


    # Sort positions to maintain tensor axis order
    sorted_positions = sorted(positions)

    # Extract indices and tags to merge
    indices_to_merge = [tensor.indices[i] for i in sorted_positions]
    tags_to_merge = [tensor.itags[i] for i in sorted_positions]

    # Determine merged tag
    if merged_tag is None:
        merged_tag = "_merged_"

    # Create isometry for fusion
    # The isometry will have opposite directions to enable contraction
    iso = isometry_n(
        indices_to_merge,
        itags=tuple(tags_to_merge) + (merged_tag,),
        direction=direction,
        dtype=tensor.dtype,
    )

    # Contract isometry with tensor to merge axes
    # The isometry indices are opposite to tensor indices, so they'll contract
    # Order: iso first, tensor second -> merged index comes first
    # Contract: first N indices of iso with the sorted_positions axes of tensor
    iso_axes = list(range(len(sorted_positions)))
    merged = contract(iso, tensor, axes=(iso_axes, sorted_positions))

    # Create conjugate of isometry for potential unfusing
    # This flips all directions and conjugates data
    iso_conj = conj(iso)

    return merged, iso_conj

