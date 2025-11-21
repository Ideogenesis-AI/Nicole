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
value decomposition (SVD) components. The `svd` function implements the
general SVD decomposition operation, while `svd_left` and `svd_right`
provide specialised variants for decomposing tensors into left/right singular
vectors and singular values, respectively.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .blocks import BlockKey
from .index import Index
from .tensor import Tensor


def _axes_from_names(itags: Sequence[str], names: Sequence[str]) -> List[int]:
    """Translate index tags into positional axis indices."""
    # Build a lookup once to avoid repeated linear searches.
    name_to_axis = {tag: i for i, tag in enumerate(itags)}
    return [name_to_axis[n] for n in names]


def svd(T: Tensor, split: Tuple[Sequence[int] | Sequence[str], Sequence[int] | Sequence[str]]):
    """Perform an SVD over a bipartition of tensor axes.

    Parameters
    ----------
    T:
        Tensor to be decomposed.
    split:
        Two-element tuple describing the axes belonging to the left and right
        partitions. Entries can be integer axis positions or index tags.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Triplet `(U, S, Vh)` where `U` and `Vh` carry the left/right legs and
        `S` stores the singular values per block.
    """
    # Normalise axis descriptors into lists of integer positions.
    left_axes, right_axes = split
    if left_axes and isinstance(left_axes[0], str):  # type: ignore[index]
        left = _axes_from_names(T.itags, left_axes)  # type: ignore[arg-type]
    else:
        left = list(left_axes)  # type: ignore[assignment]
    if right_axes and isinstance(right_axes[0], str):  # type: ignore[index]
        right = _axes_from_names(T.itags, right_axes)  # type: ignore[arg-type]
    else:
        right = list(right_axes)  # type: ignore[assignment]
    if set(left) & set(right):
        raise ValueError("Left and right axes overlap")
    if set(left + right) != set(range(len(T.indices))):
        raise ValueError("Left/right axes must cover all indices")

    # Names for the resulting left/right indices (caller may retag later).
    left_name = "_L"
    right_name = "_R"
    U_blocks: Dict[BlockKey, np.ndarray] = {}
    S_blocks: Dict[BlockKey, np.ndarray] = {}
    Vh_blocks: Dict[BlockKey, np.ndarray] = {}

    for key, arr in T.data.items():
        # Move selected axes to matrix form and compute standard SVD.
        left_shape = [arr.shape[i] for i in left]
        right_shape = [arr.shape[i] for i in right]
        mat = np.transpose(arr, axes=left + right).reshape(int(np.prod(left_shape)), int(np.prod(right_shape)))
        U, s, Vh = np.linalg.svd(mat, full_matrices=False)
        out_key = tuple(key)
        U_blocks[out_key] = U
        S_blocks[out_key] = s
        Vh_blocks[out_key] = Vh

    # Construct placeholder indices for the left/right singular vector legs.
    left_index = Index(T.indices[left[0]].direction, T.indices[left[0]].group, sectors=())
    right_index = Index(T.indices[right[0]].direction, T.indices[right[0]].group, sectors=())
    U_tensor = Tensor(indices=(left_index,), itags=(left_name,), data=U_blocks, dtype=T.dtype)
    S_tensor = Tensor(indices=(), itags=(), data=S_blocks, dtype=np.result_type(T.dtype, float))
    Vh_tensor = Tensor(indices=(right_index,), itags=(right_name,), data=Vh_blocks, dtype=T.dtype)
    return U_tensor, S_tensor, Vh_tensor
