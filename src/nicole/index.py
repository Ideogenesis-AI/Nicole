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
- Provide convenience helpers for flipping indices during tensor
  manipulations.
- Combine multiple indices into a single fused index, accumulating sector
  dimensions, and perform the inverse consistency check when splitting.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Sequence, Tuple

from .typing import Charge, Direction, Sector
from .symmetry.base import AbelianGroup, SymmetryGroup
from .symmetry.product import ProductGroup



@dataclass(frozen=True)
class Index:
    """Symmetry-aware tensor index capturing direction, group, and charge sectors.
    Keeps tensor legs self-consistent so fusion and splitting utilities can rely
    on validated charges. The flipping pair streamline common tensor network rewrites.

    Attributes
    ----------
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
    dual() / flip():
        Reverse orientation with or without charge conjugation to suit diagram
        manipulations.
    """

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
                raise ValueError(f"Duplicate sector charge {s.charge} in index")
            seen[s.charge] = s.dim

    @property
    def dim(self) -> int:
        """Total dimension of the index after summing over all sectors."""
        return sum(s.dim for s in self.sectors)

    def __str__(self) -> str:
        """Return a formatted multiline summary of the Index."""
        from .display import index_summary
        return index_summary(self)
    
    __repr__ = __str__

    def flip(self) -> Index:
        """Return a copy with direction flipped but raw charge sectors untouched."""

        return Index(self.direction.reverse(), self.group, self.sectors)

    def dual(self) -> Index:
        """Return the dual index with direction reversed and conjugated charges."""

        new_direction = self.direction.reverse()
        new_sectors = tuple(
            Sector(self.group.dual(sector.charge), sector.dim) for sector in self.sectors
        )
        return Index(new_direction, self.group, new_sectors)

    def sector_dim_map(self) -> Dict[Charge, int]:
        """Map each sector's charge to its dimension for quick lookups."""
        return {s.charge: s.dim for s in self.sectors}

    def charges(self) -> Tuple[Charge, ...]:
        """Return the immutable sequence of charges carried by this index."""
        return tuple(s.charge for s in self.sectors)



def combine_indices(direction: Direction, *inds: Index) -> Index:
    """Fuse multiple indices into one, accumulating sector dimensions.

    Parameters
    ----------
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
    if not isinstance(group, (AbelianGroup, ProductGroup)):
        raise NotImplementedError("Only Abelian/Product combine supported")
    if any(ind.group != group for ind in inds):
        raise ValueError("All indices must share the same group to combine")

    # Fuse the charges from each component index using the group's fusion rule,
    # keeping track of the cumulative dimension contributed by the block tuple.
    # Apply direction-aware charge contributions: OUT uses charge as-is, IN uses inverse.
    charge_to_dim: Dict[Charge, int] = {}
    for sectors_tuple in product(*(ind.sectors for ind in inds)):
        # Compute the sum of direction-aware contributions from input indices
        # Charges being fused (IN) contribute as-is, charges fused (OUT) contribute inverse
        total_contrib = group.neutral
        dim = 1
        for ind, sector in zip(inds, sectors_tuple):
            contrib = sector.charge if ind.direction == Direction.IN else group.inverse(sector.charge)
            total_contrib = group.fuse(total_contrib, contrib)
            dim *= sector.dim
        
        # The fused index contains the result (outgoing)
        # Inverse when direction is IN
        fused_charge = group.inverse(total_contrib) if direction == Direction.IN else total_contrib
        charge_to_dim[fused_charge] = charge_to_dim.get(fused_charge, 0) + dim

    # Build the fused index by sorting charges for deterministic ordering.
    sectors = tuple(Sector(q, d) for q, d in sorted(charge_to_dim.items(), key=lambda x: str(x[0])))
    # Return the fused index with the accumulated sectors and direction.
    return Index(direction=direction, group=group, sectors=sectors)


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
    if not isinstance(group, (AbelianGroup, ProductGroup)):
        raise NotImplementedError("Only Abelian/Product split supported")

    # Reuse `combine_indices` to ensure the proposed parts reproduce the parent.
    fused = combine_indices(parent.direction, *parts)
    if fused.sector_dim_map() != parent.sector_dim_map():
        # Any mismatch implies the supplied indices do not faithfully represent
        # the original parent's charge structure.
        raise ValueError("Split parts do not match parent index sectors")

    # Return the original parts as a validated sequence.
    return tuple(parts)


def union_indices(idx_a: Index, idx_b: Index) -> Index:
    """Create an index containing the union of sectors from two indices.
    
    The two indices must have the same symmetry group and direction.
    If a charge appears in both indices, it must have the same dimension.
    
    Parameters
    ----------
    idx_a : Index
        First index
    idx_b : Index
        Second index
    
    Returns
    -------
    Index
        New index containing all sectors from both inputs, sorted by charge
    
    Raises
    ------
    ValueError
        If indices have different groups, directions, or overlapping sectors
        with mismatched dimensions
    
    Examples
    --------
    >>> from nicole.symmetry import U1Group
    >>> idx1 = Index(Direction.OUT, U1Group(), (Sector(0, 2), Sector(1, 2)))
    >>> idx2 = Index(Direction.OUT, U1Group(), (Sector(0, 2), Sector(-1, 2)))
    >>> idx_union = union_indices(idx1, idx2)
    >>> [s.charge for s in idx_union.sectors]
    [-1, 0, 1]
    """
    if idx_a.group != idx_b.group:
        raise ValueError("Indices must have the same symmetry group")
    if idx_a.direction != idx_b.direction:
        raise ValueError("Indices must have the same direction")
    
    # Combine sectors: charge -> dim
    sector_map = {}
    for sector in idx_a.sectors:
        sector_map[sector.charge] = sector.dim
    
    for sector in idx_b.sectors:
        if sector.charge in sector_map:
            # Charge appears in both - dimensions must match
            if sector_map[sector.charge] != sector.dim:
                raise ValueError(
                    f"Sector with charge {sector.charge} has dimension "
                    f"{sector_map[sector.charge]} in first index but {sector.dim} in second"
                )
        else:
            sector_map[sector.charge] = sector.dim
    
    # Create new index with merged sectors, sorted by charge
    merged_sectors = tuple(Sector(charge, dim) for charge, dim in sorted(sector_map.items()))
    return Index(
        direction=idx_a.direction,
        group=idx_a.group,
        sectors=merged_sectors
    )


