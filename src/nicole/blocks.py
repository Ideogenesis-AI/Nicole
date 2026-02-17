# Copyright (C) 2025 Changkai Zhang.
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

"""Block schema utilities for symmetry-aware tensors.

This module collects helper routines that describe and validate the block
structure induced by a set of symmetry-labelled indices. The central
`BlockSchema` class offers static helpers for iterating admissible charge
combinations, deriving dense shapes, allocating zero blocks, and checking
charge conservation for a given block key.
"""

from itertools import product
from typing import Iterable, List, Mapping, Sequence, Tuple

import torch

from .index import Index
from .typing import Charge, Direction

BlockKey = Tuple[Charge, ...]


class BlockSchema:
    """Utility namespace for reasoning about tensor block dictionaries.

    Responsibilities
    ----------------
    - Generate all admissible charge keys permitted by a collection of indices.
    - Convert a block key into explicit dense shapes and allocate zero blocks.
    - Validate that supplied data matches the expected shapes and charge rules.

    Methods
    -------
    iter_admissible_keys()
        Yield the cartesian product of available charges for each index.
    shape_for_key()
        Translate a block key into the per-leg dense dimensions.
    validate_blocks()
        Ensure blocks are torch tensors of the correct shape.
    charge_totals()
        Compute the net fused charge for a block key.
    charges_conserved()
        Check whether a block key respects charge conservation.
    """

    @staticmethod
    def iter_admissible_keys(indices: Iterable[Index]) -> Iterable[BlockKey]:
        """Yield all charge combinations compatible with the provided indices."""
        # Collect the charge list for each leg and build the cartesian product.
        charges_per_leg = [idx.charges() for idx in indices]
        return product(*charges_per_leg)

    @staticmethod
    def shape_for_key(indices: Sequence[Index], key: BlockKey) -> Tuple[int, ...]:
        """Return the dense tensor shape associated with a block key."""
        if len(key) != len(indices):
            raise ValueError("Key length does not match number of indices")
        shape: List[int] = []
        # Pair each charge with its index and look up the dimensionality.
        for i, (idx, charge) in enumerate(zip(indices, key)):
            dim_map = idx.sector_dim_map()
            if charge not in dim_map:
                raise KeyError(f"Charge {charge} not present in index at position {i}")
            shape.append(dim_map[charge])
        return tuple(shape)

    @staticmethod
    def validate_blocks(indices: Sequence[Index], blocks: Mapping[BlockKey, torch.Tensor]) -> None:
        """Verify that blocks are PyTorch tensors with shapes consistent with the indices."""
        for arr in blocks.values():
            if not isinstance(arr, torch.Tensor):
                raise TypeError("Blocks must be torch tensors")
        for key, arr in blocks.items():
            expected = BlockSchema.shape_for_key(indices, key)
            if arr.shape != expected:
                raise ValueError(f"Block {key} has shape {arr.shape}, expected {expected}")

    @staticmethod
    def charge_totals(indices: Sequence[Index], key: BlockKey) -> Charge:
        """Compute net charge for a given block key.
        
        All indices must share the same symmetry group. Returns the fused
        total charge with direction-aware contributions (OUT charges contribute
        as-is, IN charges contribute as dual).
        """
        if not indices:
            raise ValueError("Cannot compute charge totals for empty indices")
        
        group = indices[0].group
        total = group.neutral
        
        # Traverse each leg, fusing charges with appropriate direction adjustments.
        for idx, charge in zip(indices, key):
            contribution = charge if idx.direction == Direction.OUT else group.dual(charge)
            total = group.fuse(total, contribution)
        return total

    @staticmethod
    def charges_conserved(indices: Sequence[Index], key: BlockKey) -> bool:
        """Return True if the block key satisfies charge conservation."""
        if not indices:
            return True
        group = indices[0].group
        total = BlockSchema.charge_totals(indices, key)
        return group.equal(total, group.neutral)


