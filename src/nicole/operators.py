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
"""

from typing import Sequence

import numpy as np

from .tensor import Tensor


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

