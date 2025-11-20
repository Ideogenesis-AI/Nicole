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

"""Utilities for constructing canonical identity and fusion tensors.

This module provides helpers that build symmetry-aware tensors commonly used in
tensor network algorithms: a two-leg identity and a three-leg fusion isometry.
Both routines respect the block structure defined by Nicole indices and ensure
charge conservation across all generated blocks.
"""

from typing import Dict, Optional

import numpy as np

from .index import Index, combine_indices
from .symmetry.base import AbelianGroup
from .tensor import Tensor
from .typing import Charge, Direction


def identity(index: Index, *, dtype=np.float64, new_itag: Optional[str] = None) -> Tensor:
    """Return a 2-leg identity tensor between `index` and its conjugate leg.

    Parameters
    ----------
    index:
        The index to be paired with its flipped counterpart.
    dtype:
        Data type for the identity matrices stored in each block.
    new_itag:
        Optional replacement tag applied to the flipped leg before tensor creation.

    Returns
    -------
    Tensor
        Tensor with two indices (original and flipped) whose blocks encode the
        identity matrices for each sector.
    """

    # Prepare the left leg and its flipped partner.
    left = index
    right = index.flip()
    if new_itag is not None:
        right = right.retag(new_itag)

    blocks: Dict[tuple[Charge, Charge], np.ndarray] = {}
    # Populate diagonal blocks keyed by identical charges.
    for sector in left.sectors:
        q = sector.charge
        dim = sector.dim
        blocks[(q, q)] = np.eye(dim, dtype=dtype)

    return Tensor(indices=(left, right), data=blocks, dtype=dtype)


def isometry(
    first: Index,
    second: Index,
    *,
    dtype=np.float64,
    new_itag: Optional[str] = None,
    fused_direction: Optional[Direction] = None,
) -> Tensor:
    """Return a 3-leg tensor that fuses ``first ⊗ second`` into a fused leg.

    Parameters
    ----------
    first, second:
        Input indices to be fused. They must share a symmetry group.
    dtype:
        Data type for the emitted fusion blocks.
    new_itag:
        Optional tag applied to the fused leg.
    fused_direction:
        Optional direction for the fused leg. Defaults to the dual of `first`.

    Returns
    -------
    Tensor
        Three-leg tensor whose third index represents the fusion of the first two.

    Raises
    ------
    ValueError
        If the incoming indices do not belong to the same symmetry group.
    NotImplementedError
        When attempting to fuse non-Abelian indices (not yet supported).
    RuntimeError
        If internal bookkeeping detects a fusion shape mismatch.
    """
    if first.group != second.group:
        raise ValueError("Both indices must share the same symmetry group")
    group = first.group
    if not isinstance(group, AbelianGroup):
        raise NotImplementedError("Fusion currently supports only Abelian groups")

    # Determine orientation of the fused leg; default to the dual of `first`.
    default_dir = first.direction.reverse()
    direction = fused_direction if fused_direction is not None else default_dir
    fused = combine_indices(new_itag or f"{first.itag}{second.itag}", direction, first, second)

    # Track how many columns have been written per fused charge.
    offsets: Dict[Charge, int] = {sector.charge: 0 for sector in fused.sectors}
    blocks: Dict[tuple[Charge, Charge, Charge], np.ndarray] = {}
    dim_fused_map = fused.sector_dim_map()

    for sa in first.sectors:
        qa = sa.charge
        da = sa.dim
        for sb in second.sectors:
            qb = sb.charge
            db = sb.dim
            qf = group.fuse(qa, qb)
            fused_dim = dim_fused_map[qf]
            offset = offsets[qf]
            arr = np.zeros((da, db, fused_dim), dtype=dtype)
            # Fill a set of identity matrices at appropriate column offsets.
            for i in range(da):
                base = offset + i * db
                cols = slice(base, base + db)
                arr[i, :, cols] = np.eye(db, dtype=dtype)
            blocks[(qa, qb, qf)] = arr
            offsets[qf] = offset + da * db

    # Ensure each fused sector is completely populated.
    for q, offset in offsets.items():
        if offset != dim_fused_map[q]:
            raise RuntimeError("Fusion tensor construction mismatch")

    return Tensor(indices=(first, second, fused), data=blocks, dtype=dtype)

