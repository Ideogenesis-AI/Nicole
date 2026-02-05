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


"""Tests for Index class and related functions."""

import pytest

from nicole import Direction, Index, Sector, U1Group, Z2Group
from nicole.index import combine_indices, split_index, union_indices
from nicole.symmetry.product import ProductGroup


def test_index_construction():
    """Test basic Index construction."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 3))
    idx = Index(Direction.OUT, group, sectors)
    
    assert idx.direction == Direction.OUT
    assert idx.group == group
    assert idx.sectors == sectors


def test_index_empty_sectors():
    """Test Index with empty sectors."""
    group = U1Group()
    idx = Index(Direction.IN, group, sectors=())
    
    assert idx.direction == Direction.IN
    assert idx.group == group
    assert idx.sectors == ()
    assert idx.dim == 0


def test_index_dim_property():
    """Test Index.dim property."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 3), Sector(-1, 5))
    idx = Index(Direction.OUT, group, sectors)
    
    assert idx.dim == 2 + 3 + 5


def test_index_rejects_duplicate_charges():
    """Test that Index rejects duplicate charges."""
    group = U1Group()
    with pytest.raises(ValueError, match="Duplicate sector charge"):
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(0, 2)))


def test_index_rejects_zero_dimension():
    """Test that Index rejects zero-dimension sectors."""
    group = U1Group()
    with pytest.raises(ValueError, match="must be positive"):
        Index(Direction.OUT, group, sectors=(Sector(0, 0),))


def test_index_rejects_negative_dimension():
    """Test that Index rejects negative-dimension sectors."""
    group = U1Group()
    with pytest.raises(ValueError, match="must be positive"):
        Index(Direction.OUT, group, sectors=(Sector(1, -3),))


def test_index_flip():
    """Test Index.flip() method."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 3))
    idx = Index(Direction.OUT, group, sectors)
    
    flipped = idx.flip()
    
    assert flipped.direction == Direction.IN
    assert flipped.group == group
    assert flipped.sectors == sectors  # Charges unchanged


def test_index_flip_preserves_original():
    """Test that flip() doesn't modify original."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 3))
    idx = Index(Direction.OUT, group, sectors)
    
    flipped = idx.flip()
    
    assert idx.direction == Direction.OUT  # Original unchanged
    assert flipped.direction == Direction.IN


def test_index_dual():
    """Test Index.dual() method."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 3), Sector(-1, 4))
    idx = Index(Direction.OUT, group, sectors)
    
    dual = idx.dual()
    
    assert dual.direction == Direction.IN
    assert dual.group == group
    # Charges should be conjugated
    assert dual.sectors == (Sector(0, 2), Sector(-1, 3), Sector(1, 4))


def test_index_dual_z2():
    """Test Index.dual() with Z2 group."""
    group = Z2Group()
    sectors = (Sector(0, 2), Sector(1, 3))
    idx = Index(Direction.IN, group, sectors)
    
    dual = idx.dual()
    
    assert dual.direction == Direction.OUT
    # Z2 dual: 0->0, 1->1
    assert dual.sectors == (Sector(0, 2), Sector(1, 3))


def test_index_sector_dim_map():
    """Test Index.sector_dim_map() method."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 3), Sector(-1, 5))
    idx = Index(Direction.OUT, group, sectors)
    
    dim_map = idx.sector_dim_map()
    
    assert dim_map == {0: 2, 1: 3, -1: 5}


def test_index_charges():
    """Test Index.charges() method."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 3), Sector(-1, 5))
    idx = Index(Direction.OUT, group, sectors)
    
    charges = idx.charges()
    
    assert charges == (0, 1, -1)


def test_index_frozen():
    """Test that Index is immutable (frozen)."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    with pytest.raises(AttributeError):
        idx.direction = Direction.IN  # type: ignore
    with pytest.raises(AttributeError):
        idx.group = Z2Group()  # type: ignore


# combine_indices tests

def test_combine_indices_simple():
    """Test basic combine_indices operation."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(-1, 2)))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    assert combined.direction == Direction.OUT
    assert combined.group == group
    # Expected fused charges and dimensions (with direction-aware fusion):
    # idx1 OUT charges: 0, 1 (contribute as inverse since OUT)
    # idx2 IN charges: 0, -1 (contribute as-is since IN)
    # Output direction is OUT, so qf = total_contrib (no inverse)
    # (OUT:0, IN:0) -> total: inv(0)+0=0, qf=0, dim 2*3=6
    # (OUT:0, IN:-1) -> total: inv(0)+(-1)=-1, qf=-1, dim 2*2=4
    # (OUT:1, IN:0) -> total: inv(1)+0=-1, qf=-1, dim 1*3=3
    # (OUT:1, IN:-1) -> total: inv(1)+(-1)=-2, qf=-2, dim 1*2=2
    # So charge 0 has dim 6, charge -1 has dim 4+3=7, charge -2 has dim 2
    dim_map = combined.sector_dim_map()
    assert dim_map[0] == 6
    assert dim_map[-1] == 7
    assert dim_map[-2] == 2


def test_combine_indices_single():
    """Test combine_indices with single index."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    combined = combine_indices(Direction.IN, idx)
    
    # Single OUT index contributes inverse (since OUT)
    # Output dir=IN, so qf = inv(total_contrib)
    # OUT:0 -> total=inv(0)=0, qf=inv(0)=0, dim 2
    # OUT:1 -> total=inv(1)=-1, qf=inv(-1)=1, dim 3
    assert combined.direction == Direction.IN
    assert combined.sector_dim_map() == {0: 2, 1: 3}


def test_combine_indices_three():
    """Test combine_indices with three indices."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))
    
    combined = combine_indices(Direction.OUT, idx1, idx2, idx3)
    
    assert combined.direction == Direction.OUT
    # All combinations that fuse to neutral should appear
    assert combined.group == group


def test_combine_indices_empty_raises():
    """Test that combine_indices raises on empty input."""
    with pytest.raises(ValueError, match="No indices to combine"):
        combine_indices(Direction.OUT)


def test_combine_indices_different_groups_raises():
    """Test that combine_indices raises on different groups."""
    u1 = U1Group()
    z2 = Z2Group()
    idx1 = Index(Direction.OUT, u1, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.OUT, z2, sectors=(Sector(0, 2),))
    
    with pytest.raises(ValueError, match="same group"):
        combine_indices(Direction.OUT, idx1, idx2)


def test_combine_indices_z2():
    """Test combine_indices with Z2 group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 3)))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    dim_map = combined.sector_dim_map()
    # (0,0)->0: dim 2*1=2
    # (0,1)->1: dim 2*3=6
    # (1,0)->1: dim 1*1=1
    # (1,1)->0: dim 1*3=3
    assert dim_map[0] == 2 + 3
    assert dim_map[1] == 6 + 1


# split_index tests

def test_split_index_valid():
    """Test split_index with valid parts."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(-1, 2)))
    
    parent = combine_indices(Direction.OUT, idx1, idx2)
    parts = split_index(parent, [idx1, idx2])
    
    assert parts == (idx1, idx2)


def test_split_index_invalid_raises():
    """Test split_index with mismatched parts."""
    group = U1Group()
    parent = Index(Direction.OUT, group, sectors=(Sector(0, 5), Sector(1, 3)))
    wrong_parts = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 1),))
    ]
    
    with pytest.raises(ValueError, match="do not match parent"):
        split_index(parent, wrong_parts)


def test_split_index_single_part():
    """Test split_index with single part."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    parent = combine_indices(Direction.IN, idx)
    parts = split_index(parent, [idx])
    
    assert len(parts) == 1
    assert parts[0] == idx


def test_split_index_z2():
    """Test split_index with Z2 group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 3)))
    
    parent = combine_indices(Direction.OUT, idx1, idx2)
    parts = split_index(parent, [idx1, idx2])
    
    assert parts == (idx1, idx2)


def test_combine_split_roundtrip():
    """Test that combine and split are inverses."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 4)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    combined = combine_indices(Direction.OUT, idx1, idx2, idx3)
    recovered = split_index(combined, [idx1, idx2, idx3])
    
    assert recovered == (idx1, idx2, idx3)


# ProductGroup integration tests for combine/split

def test_combine_indices_product_group():
    """Test combining indices with ProductGroup."""
    group = ProductGroup([U1Group(), Z2Group()])
    
    idx1 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2), Sector((1, 1), 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 1), Sector((1, 0), 2)))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # Check that combined index has correct sectors
    assert combined.direction == Direction.OUT
    assert combined.group == group
    
    # Expected combined sectors (both indices OUT, output direction OUT):
    # Both OUT, so contribute inverse; output OUT, so qf = total (no inverse)
    # (OUT:(0,0), OUT:(0,0)) -> total=inv((0,0))=(0,0), qf=(0,0) with dim 2*1=2
    # (OUT:(0,0), OUT:(1,0)) -> total=inv((0,0))+inv((1,0))=(-1,0), qf=(-1,0) with dim 2*2=4
    # (OUT:(1,1), OUT:(0,0)) -> total=inv((1,1))+inv((0,0))=(-1,1), qf=(-1,1) with dim 1*1=1
    # (OUT:(1,1), OUT:(1,0)) -> total=inv((1,1))+inv((1,0))=(-2,1), qf=(-2,1) with dim 1*2=2
    expected_charges = {(0, 0), (-1, 0), (-1, 1), (-2, 1)}
    actual_charges = {s.charge for s in combined.sectors}
    assert actual_charges == expected_charges


def test_split_index_product_group():
    """Test splitting an index with ProductGroup."""
    group = ProductGroup([U1Group(), Z2Group()])
    
    idx1 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2), Sector((1, 1), 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 1), Sector((1, 0), 2)))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    recovered = split_index(combined, [idx1, idx2])
    
    assert recovered == (idx1, idx2)


# union_indices tests

def test_union_indices_non_overlapping():
    """Test union_indices with completely non-overlapping sectors."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(2, 3)))
    
    union = union_indices(idx1, idx2)
    
    # Should contain all sectors, sorted by charge
    assert union.direction == Direction.OUT
    assert union.group == group
    charges = [s.charge for s in union.sectors]
    assert charges == [-1, 0, 1, 2]
    
    # Check dimensions
    dim_map = union.sector_dim_map()
    assert dim_map == {-1: 2, 0: 2, 1: 2, 2: 3}


def test_union_indices_overlapping_matching():
    """Test union_indices with overlapping sectors that have matching dimensions."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    
    union = union_indices(idx1, idx2)
    
    # Overlapping charge 0 should appear once with dim 2
    charges = [s.charge for s in union.sectors]
    assert charges == [-1, 0, 1]
    
    dim_map = union.sector_dim_map()
    assert dim_map == {-1: 2, 0: 2, 1: 3}


def test_union_indices_identical():
    """Test union_indices with identical indices."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    union = union_indices(idx, idx)
    
    # Should be equivalent to the original
    assert union.direction == idx.direction
    assert union.group == idx.group
    assert union.sector_dim_map() == idx.sector_dim_map()


def test_union_indices_fully_overlapping():
    """Test union_indices where all sectors overlap with matching dimensions."""
    group = U1Group()
    # Same charges, same dimensions, but different order
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(-1, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 3), Sector(-1, 1), Sector(0, 2)))
    
    union = union_indices(idx1, idx2)
    
    # Should have all sectors, sorted by charge
    charges = [s.charge for s in union.sectors]
    assert charges == [-1, 0, 1]
    assert union.sector_dim_map() == {-1: 1, 0: 2, 1: 3}


def test_union_indices_mismatched_dimensions_raises():
    """Test union_indices raises when overlapping sectors have different dimensions."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(-1, 2)))
    
    with pytest.raises(ValueError, match="Sector with charge 0 has dimension 2.*but 3"):
        union_indices(idx1, idx2)


def test_union_indices_different_groups_raises():
    """Test union_indices raises when indices have different groups."""
    idx1 = Index(Direction.OUT, U1Group(), sectors=(Sector(0, 2),))
    idx2 = Index(Direction.OUT, Z2Group(), sectors=(Sector(0, 2),))
    
    with pytest.raises(ValueError, match="same symmetry group"):
        union_indices(idx1, idx2)


def test_union_indices_different_directions_raises():
    """Test union_indices raises when indices have different directions."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    with pytest.raises(ValueError, match="same direction"):
        union_indices(idx1, idx2)


def test_union_indices_z2():
    """Test union_indices with Z2 group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 3), Sector(0, 1)))
    
    # Charge 1 appears in both with dim 3 -> OK
    # Charge 0 appears in both with different dims (2 vs 1) -> should raise
    with pytest.raises(ValueError, match="Sector with charge 0 has dimension 2.*but 1"):
        union_indices(idx1, idx2)


def test_union_indices_product_group():
    """Test union_indices with ProductGroup."""
    group = ProductGroup([U1Group(), Z2Group()])
    idx1 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2), Sector((1, 1), 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2), Sector((-1, 0), 1)))
    
    union = union_indices(idx1, idx2)
    
    # Charge (0, 0) overlaps with matching dim 2
    # Other charges are disjoint
    charges = [s.charge for s in union.sectors]
    assert sorted(charges) == sorted([(-1, 0), (0, 0), (1, 1)])
    assert union.sector_dim_map() == {(-1, 0): 1, (0, 0): 2, (1, 1): 3}


def test_union_indices_preserves_order():
    """Test that union_indices returns sectors sorted by charge."""
    group = U1Group()
    # Deliberately create indices with unsorted charges
    idx1 = Index(Direction.OUT, group, sectors=(Sector(5, 1), Sector(1, 2)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(-3, 3), Sector(0, 1)))
    
    union = union_indices(idx1, idx2)
    
    charges = [s.charge for s in union.sectors]
    assert charges == sorted(charges)  # Should be [-3, 0, 1, 5]
    assert charges == [-3, 0, 1, 5]

