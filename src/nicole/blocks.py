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

"""Block schema utilities for symmetry-aware tensors.

This module collects helper routines that describe and validate the block
structure induced by a set of symmetry-labelled indices. The central
`BlockSchema` class offers static helpers for iterating admissible charge
combinations, deriving dense shapes, allocating zero blocks, and checking
charge conservation for a given block key.
"""

from itertools import product
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple
from typing import TYPE_CHECKING

import torch

from .index import Index
from .typing import Charge, Direction

if TYPE_CHECKING:
    from .symmetry.delegate import Bridge

BlockKey = Tuple[Charge, ...]


class BlockSchema:
    """Utility namespace for reasoning about tensor block dictionaries.

    Responsibilities
    ----------------
    - Generate all admissible charge keys permitted by a collection of indices.
    - Convert a block key into explicit dense shapes and allocate zero blocks.
    - Validate that supplied data matches the expected shapes and charge rules.
    - Handle both Abelian (unique fusion) and non-Abelian (multi-channel) symmetries.

    Methods
    -------
    iter_admissible_keys()
        Yield the cartesian product of available charges for each index.
    shape_for_key()
        Translate a block key into the per-index dense dimensions.
    validate_blocks()
        Ensure blocks are torch tensors of the correct shape.
    charge_totals()
        Compute the net fused charge for a block key (Abelian only).
    charge_avail()
        Compute all achievable total charges for a block key (non-Abelian only).
    charges_conserved()
        Check whether a block key respects charge conservation (both types).
    """

    @staticmethod
    def iter_admissible_keys(indices: Iterable[Index]) -> Iterable[BlockKey]:
        """Yield all charge combinations compatible with the provided indices."""
        # Collect the charge list for each index and build the cartesian product.
        charges_per_index = [idx.charges() for idx in indices]
        return product(*charges_per_index)

    @staticmethod
    def shape_for_key(
        indices: Sequence[Index], key: BlockKey, num_components: Optional[int] = None
    ) -> Tuple[int, ...]:
        """Return the dense tensor shape associated with a block key.
        
        Parameters
        ----------
        indices : Sequence[Index]
            Tensor indices
        key : BlockKey
            Block key (one charge per index)
        num_components : int, optional
            If provided, appends this as a trailing reduced multiplicity dimension.
            Used for non-Abelian groups to store multiple reduced tensor components.
        
        Returns
        -------
        Tuple[int, ...]
            Shape tuple. For Abelian groups: (d1, d2, ..., dn).
            For non-Abelian with num_components: (d1, d2, ..., dn, num_components).
        """
        if len(key) != len(indices):
            raise ValueError("Key length does not match number of indices")
        shape: List[int] = []
        # Pair each charge with its index and look up the dimensionality.
        for i, (idx, charge) in enumerate(zip(indices, key)):
            dim_map = idx.sector_dim_map()
            if charge not in dim_map:
                raise KeyError(f"Charge {charge} not present in index at position {i}")
            shape.append(dim_map[charge])
        
        # Append reduced multiplicity dimension for non-Abelian groups
        if num_components is not None:
            if num_components < 1:
                raise ValueError(f"num_components must be at least 1, got {num_components}")
            shape.append(num_components)
        
        return tuple(shape)

    @staticmethod
    def validate_blocks(
        indices: Sequence[Index], blocks: Mapping[BlockKey, torch.Tensor],
        intw: Optional[Mapping[BlockKey, Bridge]] = None
    ) -> None:
        """Verify that blocks are PyTorch tensors with shapes consistent with the indices.
        
        Parameters
        ----------
        indices : Sequence[Index]
            Tensor indices
        blocks : Mapping[BlockKey, torch.Tensor]
            Block data tensors
        intw : Mapping[BlockKey, Bridge], optional
            Intertwiner mapping for non-Abelian groups. If provided, validates
            that block shapes include trailing reduced multiplicity dimension
            matching Bridge.num_components.
        
        Raises
        ------
        TypeError
            If blocks are not torch.Tensor instances
        ValueError
            If block shapes don't match expected dimensions, or if intw keys
            don't match block keys for non-Abelian groups
        """
        for arr in blocks.values():
            if not isinstance(arr, torch.Tensor):
                raise TypeError("Blocks must be torch tensors")
        
        # Validate intw keys match block keys if provided
        if intw is not None:
            if set(intw.keys()) != set(blocks.keys()):
                raise ValueError("Intertwiner (intw) keys must match block keys for non-Abelian groups")
        
        # Validate block shapes
        for key, arr in blocks.items():
            if intw is not None and key in intw:
                # Non-Abelian: expect trailing reduced multiplicity dimension
                bridge = intw[key]
                base_shape = BlockSchema.shape_for_key(indices, key)
                expected = base_shape + (bridge.num_components,)
                if arr.shape != expected:
                    raise ValueError(
                        f"Block {key} has shape {arr.shape}, expected {expected} "
                        f"(base shape {base_shape} + trailing reduced multiplicity {bridge.num_components})"
                    )
            else:
                # Abelian: no trailing dimension
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
        
        # Traverse each index, fusing charges with appropriate direction adjustments.
        for idx, charge in zip(indices, key):
            contribution = charge if idx.direction == Direction.OUT else group.dual(charge)
            total = group.fuse_unique(total, contribution)
        return total

    @staticmethod
    def charge_avail(indices: Sequence[Index], key: BlockKey) -> Tuple[Charge, ...]:
        """Compute all achievable total charges for a given block key.
        
        Returns all achievable total charges from any fusion tree for non-Abelian groups.
        All indices must share the same symmetry group. Charges are fused with
        direction-aware contributions (OUT charges contribute as-is, IN charges
        contribute as dual).
        """
        if not indices:
            raise ValueError("Cannot compute charge totals for empty indices")
        
        group = indices[0].group
        
        # Collect all contributions with direction adjustments
        contributions = []
        for idx, charge in zip(indices, key):
            contribution = charge if idx.direction == Direction.OUT else group.dual(charge)
            contributions.append(contribution)
        
        # Non-Abelian: multiple fusion channels
        return group.fuse_channels(*contributions)

    @staticmethod
    def charges_conserved(indices: Sequence[Index], key: BlockKey) -> bool:
        """Return True if the block key satisfies charge conservation.
        
        For Abelian groups, checks if the unique total charge equals neutral.
        For non-Abelian groups, checks if neutral is among the achievable channels.
        """
        if not indices:
            return True
        
        group = indices[0].group
        
        # Switch logic based on whether group is Abelian or non-Abelian
        if group.is_abelian:
            # Abelian: check if unique total equals neutral
            total = BlockSchema.charge_totals(indices, key)
            return group.equal(total, group.neutral)
        else:
            # Non-Abelian: check if neutral is among available channels
            avail_charges = BlockSchema.charge_avail(indices, key)
            return any(group.equal(charge, group.neutral) for charge in avail_charges)


