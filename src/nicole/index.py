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

"""Index utilities for symmetry-aware tensor networks.

The `Index` dataclass models a single tensor leg annotated with symmetry
information. Each instance records whether the leg is incoming or outgoing,
the associated symmetry group, and the available sectors (charge, dimension
pairs) on that leg. The helper functions `combine_indices` and `split_index`
encapsulate a consistent way to fuse or validate indices while respecting the
charge rules enforced by the symmetry group.

Key responsibilities
--------------------
- Validate sectors when constructing indices so charge metadata stays sane.
- Provide convenience helpers for retagging and flipping indices during tensor
  manipulations.
- Combine multiple indices into a single fused index, accumulating sector
  dimensions, and perform the inverse consistency check when splitting.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Sequence, Tuple

from .typing import Charge, Direction, Sector
from .symmetry.base import AbelianGroup, SymmetryGroup



@dataclass(frozen=True)
class Index:
    """Symmetry-aware tensor index capturing direction, group, and charge sectors.
    Keeps tensor legs self-consistent so fusion and splitting utilities can rely
    on validated charges. Convenience helpers such as `retag` and the flipping
    pair streamline common tensor network rewrites.

    Attributes
    ----------
    itag:
        Human-readable label used when printing or logging tensors.
    direction:
        Orientation of the index (e.g. bra vs ket leg). Flips determine how
        charge conjugation is applied.
    group:
        Symmetry group object responsible for validating and fusing charges.
    sectors:
        Tuple of `(charge, dim)` pairs describing the block structure available
        on this leg.
    dim (property):
        Property returning the total dimension derived from `sectors`.

    Methods
    -------
    retag()
        Relabel the index without altering direction, group, or sectors.
    dual() / flip():
        Reverse orientation with or without charge conjugation to suit diagram
        manipulations.
    """

    itag: str
    direction: Direction
    group: SymmetryGroup
    sectors: Tuple[Sector, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate sector charges and dimensions at construction time."""
        seen: Dict[Charge, int] = {}
        for s in self.sectors:
            self.group.validate_charge(s.charge)
            if s.dim <= 0:
                raise ValueError("Sector dim must be positive")
            if s.charge in seen:
                raise ValueError(f"Duplicate sector charge {s.charge} in index {self.itag}")
            seen[s.charge] = s.dim

    @property
    def dim(self) -> int:
        """Total dimension of the index after summing over all sectors."""
        return sum(s.dim for s in self.sectors)

    def retag(self, new_itag: str) -> Index:
        """Return a copy of the index with a different identifier."""
        return Index(new_itag, self.direction, self.group, self.sectors)

    def flip(self) -> Index:
        """Return a copy with direction flipped but raw charge sectors untouched."""

        return Index(self.itag, self.direction.reverse(), self.group, self.sectors)

    def dual(self) -> Index:
        """Return the dual index with direction reversed and conjugated charges."""

        new_direction = self.direction.reverse()
        new_sectors = tuple(
            Sector(self.group.dual(sector.charge), sector.dim) for sector in self.sectors
        )
        return Index(self.itag, new_direction, self.group, new_sectors)

    def sector_dim_map(self) -> Dict[Charge, int]:
        """Map each sector's charge to its dimension for quick lookups."""
        return {s.charge: s.dim for s in self.sectors}

    def charges(self) -> Tuple[Charge, ...]:
        """Return the immutable sequence of charges carried by this index."""
        return tuple(s.charge for s in self.sectors)



def combine_indices(itag: str, direction: Direction, *inds: Index) -> Index:
    """Fuse multiple indices into one, accumulating sector dimensions.

    Parameters
    ----------
    itag:
        Identifier assigned to the fused index.
    direction:
        Direction applied to the resulting index. This does not need to match
        any individual input index directions because the caller typically
        controls the orientation of the fused leg.
    *inds:
        Component indices, each using the same symmetry group.

    Returns
    -------
    Index
        A new index whose sectors reflect the combined charge content of the
        inputs, summed across all compatible charge tuples.
    """

    # Sanity checks
    if not inds:
        raise ValueError("No indices to combine")
    group = inds[0].group
    if not isinstance(group, AbelianGroup):
        raise NotImplementedError("Only Abelian combine supported initially")
    if any(ind.group != group for ind in inds):
        raise ValueError("All indices must share the same group to combine")

    # Fuse the charges from each component index using the group's fusion rule,
    # keeping track of the cumulative dimension contributed by the block tuple.
    charge_to_dim: Dict[Charge, int] = {}
    for sectors in product(*(ind.sectors for ind in inds)):
        fused = group.neutral
        dim = 1
        for s in sectors:
            fused = group.fuse(fused, s.charge)
            dim *= s.dim
        charge_to_dim[fused] = charge_to_dim.get(fused, 0) + dim

    # Build the fused index by sorting charges for deterministic ordering.
    sectors = tuple(Sector(q, d) for q, d in sorted(charge_to_dim.items(), key=lambda x: str(x[0])))
    # Return the fused index with the accumulated sectors and direction.
    return Index(itag=itag, direction=direction, group=group, sectors=sectors)


def split_index(parent: Index, parts: Sequence[Index]) -> Tuple[Index, ...]:
    """Validate that `parts` can be combined back into `parent`.

    This helper mirrors `combine_indices` by fusing the proposed child indices
    and confirming that the resulting sector map matches the parent. No new
    indices are created; the original `parts` are returned once validated.

    Parameters
    ----------
    parent:
        The index expected to be reconstructed from `parts`.
    parts:
        Sequence of indices that should collectively match the parent's sectors.

    Returns
    -------
    tuple[Index, ...]
        The validated input sequence for ergonomic chaining.
    """

    # Sanity checks
    group = parent.group
    if not isinstance(group, AbelianGroup):
        raise NotImplementedError("Only Abelian split supported initially")

    # Reuse `combine_indices` to ensure the proposed parts reproduce the parent.
    fused = combine_indices("__tmp__", parent.direction, *parts)
    if fused.sector_dim_map() != parent.sector_dim_map():
        # Any mismatch implies the supplied indices do not faithfully represent
        # the original parent's charge structure.
        raise ValueError("Split parts do not match parent index sectors")

    # Return the original parts as a validated sequence.
    return tuple(parts)


