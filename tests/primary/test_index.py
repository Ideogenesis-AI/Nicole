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


"""Tests for Index class and related functions."""

import pytest

from nicole import Direction, Index, Sector, U1Group, Z2Group, SU2Group
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


def test_index_num_states_abelian():
    """Test Index.num_states for Abelian group."""
    group = U1Group()
    sectors = (Sector(0, 2), Sector(1, 3), Sector(-1, 5))
    idx = Index(Direction.OUT, group, sectors)
    
    # For Abelian groups, num_states equals dim (irrep_dim = 1)
    assert idx.num_states == idx.dim
    assert idx.num_states == 2 + 3 + 5


def test_index_num_states_su2():
    """Test Index.num_states for SU2 group."""
    group = SU2Group()
    # spin-0 (two_j=0): irrep_dim = 1, dim = 2
    # spin-1 (two_j=2): irrep_dim = 3, dim = 4
    # spin-3/2 (two_j=3): irrep_dim = 4, dim = 5
    sectors = (Sector(0, 2), Sector(2, 4), Sector(3, 5))
    idx = Index(Direction.OUT, group, sectors)
    
    # num_states = Σ (sector.dim × irrep_dim)
    expected = 2 * 1 + 4 * 3 + 5 * 4
    assert idx.num_states == expected
    assert idx.num_states == 34
    assert idx.dim == 2 + 4 + 5
    assert idx.num_states != idx.dim


def test_index_num_states_product_group_abelian():
    """Test Index.num_states for ProductGroup with only Abelian components."""
    group = ProductGroup([U1Group(), Z2Group()])
    sectors = (Sector((0, 0), 2), Sector((1, 1), 3))
    idx = Index(Direction.OUT, group, sectors)
    
    # For Abelian ProductGroup, num_states equals dim
    assert idx.num_states == idx.dim
    assert idx.num_states == 2 + 3


def test_index_num_states_product_group_with_su2():
    """Test Index.num_states for ProductGroup with SU2."""
    u1 = U1Group()
    su2 = SU2Group()
    group = ProductGroup([u1, su2])
    # (U1 charge, SU2 two_j)
    # (0, 0): irrep_dim = 1*1 = 1
    # (1, 2): irrep_dim = 1*3 = 3
    # (-1, 1): irrep_dim = 1*2 = 2
    sectors = (Sector((0, 0), 2), Sector((1, 2), 4), Sector((-1, 1), 3))
    idx = Index(Direction.OUT, group, sectors)
    
    # num_states = Σ (sector.dim × irrep_dim)
    expected = 2 * 1 + 4 * 3 + 3 * 2
    assert idx.num_states == expected
    assert idx.num_states == 20
    assert idx.dim == 2 + 4 + 3
    assert idx.num_states != idx.dim


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


def test_combine_indices_product_group_dimension_check():
    """Test combine_indices with ProductGroup verifies dimension accumulation."""
    group = ProductGroup([U1Group(), Z2Group()])
    
    # First index: charges (1,0) dim 3, (2,1) dim 5
    idx1 = Index(Direction.OUT, group, sectors=(Sector((1, 0), 3), Sector((2, 1), 5)))
    # Second index: charges (0,1) dim 7, (1,0) dim 11
    idx2 = Index(Direction.OUT, group, sectors=(Sector((0, 1), 7), Sector((1, 0), 11)))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # Both OUT, so use dual; output OUT, so no dual on result
    # (1,0)_OUT + (0,1)_OUT → dual((1,0)) + dual((0,1)) = (-1,0) + (0,1) = (-1,1)
    # (1,0)_OUT + (1,0)_OUT → dual((1,0)) + dual((1,0)) = (-1,0) + (-1,0) = (-2,0)
    # (2,1)_OUT + (0,1)_OUT → dual((2,1)) + dual((0,1)) = (-2,1) + (0,1) = (-2,0)
    # (2,1)_OUT + (1,0)_OUT → dual((2,1)) + dual((1,0)) = (-2,1) + (-1,0) = (-3,1)
    
    dim_map = combined.sector_dim_map()
    
    # Check dimension accumulation
    assert dim_map[(-1, 1)] == 3 * 7  # One path
    assert dim_map[(-2, 0)] == 3 * 11 + 5 * 7  # Two paths: accumulate!
    assert dim_map[(-3, 1)] == 5 * 11  # One path
    
    # Verify total dimension
    assert combined.dim == 3*7 + 3*11 + 5*7 + 5*11


def test_combine_indices_product_group_stress():
    """Stress test combine_indices with ProductGroup having many sectors."""
    group = ProductGroup([U1Group(), Z2Group()])
    
    # First index: 8 sectors with varying U1 charges and Z2 parities
    sectors1 = tuple(Sector((i, i % 2), i+1) for i in range(8))
    idx1 = Index(Direction.OUT, group, sectors=sectors1)
    
    # Second index: 6 sectors
    sectors2 = tuple(Sector((2*i, i % 2), i+2) for i in range(6))
    idx2 = Index(Direction.OUT, group, sectors=sectors2)
    
    # 8*6 = 48 fusion operations
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # Verify basic properties
    assert combined.group == group
    assert combined.direction == Direction.OUT
    assert len(combined.sectors) > 0
    
    # All dimensions should be positive
    for sector in combined.sectors:
        assert sector.dim > 0
    
    # Total dimension should match sum of all products
    expected_total = sum((i1+1)*(i2+2) for i1 in range(8) for i2 in range(6))
    assert combined.dim == expected_total
    
    # Verify some charges are shared (dimension accumulation)
    # With 48 combinations and limited charge range, many will overlap
    assert len(combined.sectors) < 48


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


# Non-Abelian (SU2) tests for combine_indices and split_index

def test_combine_indices_su2_pairwise():
    """Test combine_indices with SU2 group (pairwise fusion)."""
    group = SU2Group()
    # Two spin-1/2 indices (2j=1)
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Combine: spin-1/2 ⊗ spin-1/2 → spin-0 or spin-1 (2j: 0, 2)
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # Should have two sectors: charge 0 and charge 2
    assert len(combined.sectors) == 2
    charges = {s.charge for s in combined.sectors}
    assert charges == {0, 2}
    
    # Each sector has dimension 2*2 = 4 (product of input dims)
    for sector in combined.sectors:
        assert sector.dim == 4


def test_combine_indices_su2_multiple_sectors():
    """Test combine_indices with SU2 group and multiple input sectors."""
    group = SU2Group()
    # Index with spin-1/2 and spin-1 (2j: 1, 2)
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    # Index with spin-1/2 (2j: 1)
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # 1⊗1 → {0, 2}, 2⊗1 → {1, 3}
    # Combined charges: {0, 1, 2, 3}
    charges = {s.charge for s in combined.sectors}
    assert charges == {0, 1, 2, 3}
    
    # Check dimensions
    dim_map = combined.sector_dim_map()
    assert dim_map[0] == 2 * 2  # from 1⊗1
    assert dim_map[1] == 3 * 2  # from 2⊗1
    assert dim_map[2] == 2 * 2  # from 1⊗1
    assert dim_map[3] == 3 * 2  # from 2⊗1


def test_combine_indices_su2_with_directions():
    """Test combine_indices with SU2 respects IN/OUT directions."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    
    # OUT: 1, IN: dual(1) = 1 (SU2 is self-dual)
    # 1⊗1 → {0, 2}
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    charges = {s.charge for s in combined.sectors}
    assert charges == {0, 2}


def test_combine_indices_su2_more_than_two_raises():
    """Test that combine_indices raises for SU2 with more than 2 indices."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    with pytest.raises(ValueError, match="only supports pairwise fusion"):
        combine_indices(Direction.OUT, idx1, idx2, idx3)


# Tests for multiple fusion paths to same output charge

def test_combine_indices_su2_same_output_charge():
    """Test that different sector pairs fusing to same charge accumulate dimensions."""
    group = SU2Group()
    
    # Index with spin-0 and spin-1 (2j: 0, 2)
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(2, 5)))
    # Index with spin-1/2 (2j: 1)
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 7),))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # Fusion channels:
    # 0⊗1 → {1} with dim 3*7 = 21
    # 2⊗1 → {1, 3} with dim 5*7 = 35 for each
    # Output charge 1 gets contributions from BOTH: 21 + 35 = 56
    # Output charge 3 gets contribution from 2⊗1: 35
    
    dim_map = combined.sector_dim_map()
    assert dim_map[1] == 21 + 35  # Two paths to charge 1
    assert dim_map[3] == 35       # One path to charge 3


def test_combine_indices_su2_many_paths_same_charge():
    """Test multiple sector pairs all fusing to same output charges."""
    group = SU2Group()
    
    # Index with spin-1/2, spin-3/2, spin-5/2 (2j: 1, 3, 5)
    idx1 = Index(Direction.OUT, group, sectors=(
        Sector(1, 2),  # spin-1/2, dim 2
        Sector(3, 4),  # spin-3/2, dim 4
        Sector(5, 6),  # spin-5/2, dim 6
    ))
    # Index with spin-1/2 (2j: 1)
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 3),))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # All three input sectors (1, 3, 5) fused with 1:
    # 1⊗1 → {0, 2} with dims 2*3=6 each
    # 3⊗1 → {2, 4} with dims 4*3=12 each
    # 5⊗1 → {4, 6} with dims 6*3=18 each
    # Charge 0: 6 (from 1⊗1)
    # Charge 2: 6 (from 1⊗1) + 12 (from 3⊗1) = 18
    # Charge 4: 12 (from 3⊗1) + 18 (from 5⊗1) = 30
    # Charge 6: 18 (from 5⊗1)
    
    dim_map = combined.sector_dim_map()
    assert dim_map[0] == 6
    assert dim_map[2] == 6 + 12
    assert dim_map[4] == 12 + 18
    assert dim_map[6] == 18


def test_combine_indices_su2_symmetric_multiple_paths():
    """Test symmetric case where both indices have same sectors."""
    group = SU2Group()
    
    # Both indices have spin-1/2 and spin-1 (2j: 1, 2)
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 5), Sector(2, 7)))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # Fusion channels (all four combinations):
    # 1⊗1 → {0, 2} with dims 2*5=10 each
    # 1⊗2 → {1, 3} with dims 2*7=14 each
    # 2⊗1 → {1, 3} with dims 3*5=15 each
    # 2⊗2 → {0, 2, 4} with dims 3*7=21 each
    
    # Accumulation:
    # Charge 0: 10 (1⊗1) + 21 (2⊗2) = 31
    # Charge 1: 14 (1⊗2) + 15 (2⊗1) = 29
    # Charge 2: 10 (1⊗1) + 21 (2⊗2) = 31
    # Charge 3: 14 (1⊗2) + 15 (2⊗1) = 29
    # Charge 4: 21 (2⊗2)
    
    dim_map = combined.sector_dim_map()
    assert dim_map[0] == 31
    assert dim_map[1] == 29
    assert dim_map[2] == 31
    assert dim_map[3] == 29
    assert dim_map[4] == 21


def test_combine_indices_su2_all_to_neutral():
    """Test case where multiple paths lead to neutral charge."""
    group = SU2Group()
    
    # spin-1, spin-2 (2j: 2, 4)
    idx1 = Index(Direction.OUT, group, sectors=(Sector(2, 3), Sector(4, 5)))
    # spin-1, spin-2 (2j: 2, 4) with IN direction (dual)
    idx2 = Index(Direction.IN, group, sectors=(Sector(2, 7), Sector(4, 11)))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # OUT: 2, IN: dual(2)=2 → 2⊗2 → {0,2,4}
    # OUT: 4, IN: dual(4)=4 → 4⊗4 → {0,2,4,6,8}
    # Both contribute to charge 0 (neutral)
    
    dim_map = combined.sector_dim_map()
    # Charge 0: 3*7=21 (2⊗2) + 5*11=55 (4⊗4) = 76
    assert dim_map[0] == 21 + 55
    assert 0 in dim_map  # Neutral is present


def test_combine_indices_product_su2_basic():
    """Test combine_indices with ProductGroup containing SU(2)."""
    group = ProductGroup([U1Group(), SU2Group()])
    
    # First index: U1 charge 0 with SU(2) spins 0, 1 (2j: 0, 2)
    idx1 = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2),
        Sector((0, 2), 3),
    ))
    # Second index: U1 charge 1 with SU(2) spins 1, 2 (2j: 2, 4)
    idx2 = Index(Direction.OUT, group, sectors=(
        Sector((1, 2), 5),
        Sector((1, 4), 7),
    ))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    assert combined.group == group
    assert combined.direction == Direction.OUT
    
    # Both OUT, so use dual on inputs; output OUT, so no dual on result
    # Fusion details:
    # (0, 0) × (1, 2): dual(0,0)⊗dual(1,2) = (0,0)⊗(-1,2)
    #   U1: 0+(-1)=-1, SU(2): 0⊗1 → {1} → 2j:{2} → charge (-1, 2) dim 2*5=10
    # (0, 0) × (1, 4): dual(0,0)⊗dual(1,4) = (0,0)⊗(-1,4)
    #   U1: 0+(-1)=-1, SU(2): 0⊗2 → {2} → 2j:{4} → charge (-1, 4) dim 2*7=14
    # (0, 2) × (1, 2): dual(0,2)⊗dual(1,2) = (0,2)⊗(-1,2)
    #   U1: 0+(-1)=-1, SU(2): 1⊗1 → {0,1,2} → 2j:{0,2,4} → charges (-1,0), (-1,2), (-1,4), each dim 3*5=15
    # (0, 2) × (1, 4): dual(0,2)⊗dual(1,4) = (0,2)⊗(-1,4)
    #   U1: 0+(-1)=-1, SU(2): 1⊗2 → {1,2,3} → 2j:{2,4,6} → charges (-1,2), (-1,4), (-1,6), each dim 3*7=21
    
    dim_map = combined.sector_dim_map()
    assert dim_map[(-1, 0)] == 15                    # From (0,2)⊗(1,2)
    assert dim_map[(-1, 2)] == 10 + 15 + 21          # From all three fusions
    assert dim_map[(-1, 4)] == 14 + 15 + 21          # From all three fusions
    assert dim_map[(-1, 6)] == 21                    # From (0,2)⊗(1,4)


def test_combine_indices_product_su2_with_directions():
    """Test combine_indices with ProductGroup(U1×SU2) using different directions."""
    group = ProductGroup([U1Group(), SU2Group()])
    
    # First index OUT: (1, 1) with dim 4, (2, 1) with dim 6
    idx1 = Index(Direction.OUT, group, sectors=(
        Sector((1, 1), 4),
        Sector((2, 1), 6),
    ))
    # Second index IN: (0, 2) with dim 5, (1, 0) with dim 7
    idx2 = Index(Direction.IN, group, sectors=(
        Sector((0, 2), 5),
        Sector((1, 0), 7),
    ))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # Fusion calculations (OUT uses dual, IN does not):
    # (1,1)_OUT × (0,2)_IN: dual(1,1)⊗(0,2) = (-1,1)⊗(0,2)
    #   U1: -1+0=-1, SU(2): 1/2⊗1 → {1/2, 3/2} → 2j: {1,3}
    #   Charges: (-1,1) dim 4*5=20, (-1,3) dim 4*5=20
    # (1,1)_OUT × (1,0)_IN: dual(1,1)⊗(1,0) = (-1,1)⊗(1,0)
    #   U1: -1+1=0, SU(2): 1/2⊗0 → {1/2} → 2j: {1}
    #   Charge: (0,1) dim 4*7=28
    # (2,1)_OUT × (0,2)_IN: dual(2,1)⊗(0,2) = (-2,1)⊗(0,2)
    #   U1: -2+0=-2, SU(2): 1/2⊗1 → {1/2, 3/2} → 2j: {1,3}
    #   Charges: (-2,1) dim 6*5=30, (-2,3) dim 6*5=30
    # (2,1)_OUT × (1,0)_IN: dual(2,1)⊗(1,0) = (-2,1)⊗(1,0)
    #   U1: -2+1=-1, SU(2): 1/2⊗0 → {1/2} → 2j: {1}
    #   Charge: (-1,1) dim 6*7=42
    
    dim_map = combined.sector_dim_map()
    
    # Check dimension accumulation
    assert dim_map[(-1, 1)] == 20 + 42     # From (1,1)×(0,2) and (2,1)×(1,0)
    assert dim_map[(-1, 3)] == 20          # From (1,1)×(0,2)
    assert dim_map[(0, 1)] == 28           # From (1,1)×(1,0)
    assert dim_map[(-2, 1)] == 30          # From (2,1)×(0,2)
    assert dim_map[(-2, 3)] == 30          # From (2,1)×(0,2)
    
    # Verify total dim
    expected_total = 20 + 20 + 28 + 30 + 30 + 42
    assert combined.dim == expected_total


def test_combine_indices_product_su2_more_than_two_raises():
    """Test that combining more than two indices with ProductGroup(SU2) raises."""
    group = ProductGroup([U1Group(), SU2Group()])
    
    idx1 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2),))
    
    with pytest.raises(ValueError, match="Non-Abelian.*pairwise"):
        combine_indices(Direction.OUT, idx1, idx2, idx3)


# Stress tests for robustness

def test_combine_indices_su2_stress_many_sectors():
    """Stress test with many sectors creating complex overlaps."""
    group = SU2Group()
    
    # Index with 10 different spin sectors (2j: 0, 1, 2, ..., 9)
    sectors1 = tuple(Sector(i, i+1) for i in range(10))
    idx1 = Index(Direction.OUT, group, sectors=sectors1)
    
    # Index with 5 different spin sectors (2j: 0, 2, 4, 6, 8)
    sectors2 = tuple(Sector(2*i, i+1) for i in range(5))
    idx2 = Index(Direction.OUT, group, sectors=sectors2)
    
    # This creates 10*5 = 50 fusion operations with many overlapping outputs
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # Should have no errors and produce valid index
    assert combined.group == group
    assert combined.direction == Direction.OUT
    assert len(combined.sectors) > 0
    
    # All dimensions should be positive
    for sector in combined.sectors:
        assert sector.dim > 0
    
    # Total dimension will be larger than simple products due to multi-channel fusion
    # Each fusion creates multiple channels, all contributing to the total
    # Just verify it's larger than the minimum (single channel per fusion)
    min_expected = sum((i1+1)*(i2+1) for i1 in range(10) for i2 in range(5))
    assert combined.dim >= min_expected


def test_combine_indices_su2_stress_all_same_output():
    """Stress test where all inputs fuse to same few outputs."""
    group = SU2Group()
    
    # Many high-spin sectors that all fuse with spin-1/2 to similar outputs
    # spin-9/2, spin-11/2, spin-13/2, spin-15/2 (2j: 9, 11, 13, 15)
    idx1 = Index(Direction.OUT, group, sectors=(
        Sector(9, 2),
        Sector(11, 3),
        Sector(13, 4),
        Sector(15, 5),
    ))
    # spin-1/2 (2j: 1)
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 7),))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # 9⊗1 → {8, 10}
    # 11⊗1 → {10, 12}
    # 13⊗1 → {12, 14}
    # 15⊗1 → {14, 16}
    # Charges 10, 12, 14 each get two contributions
    
    dim_map = combined.sector_dim_map()
    assert dim_map[8] == 2*7
    assert dim_map[10] == 2*7 + 3*7  # Two paths
    assert dim_map[12] == 3*7 + 4*7  # Two paths
    assert dim_map[14] == 4*7 + 5*7  # Two paths
    assert dim_map[16] == 5*7


def test_combine_indices_su2_mixed_directions_complex():
    """Complex test with mixed directions and multiple overlap."""
    group = SU2Group()
    
    # OUT index with multiple sectors
    idx1 = Index(Direction.OUT, group, sectors=(
        Sector(0, 1),
        Sector(1, 2),
        Sector(2, 3),
    ))
    # IN index with multiple sectors (will use dual)
    idx2 = Index(Direction.IN, group, sectors=(
        Sector(1, 4),
        Sector(2, 5),
    ))
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # OUT: 0, IN: dual(1)=1 → 0⊗1 → {1} with dim 1*4=4
    # OUT: 0, IN: dual(2)=2 → 0⊗2 → {2} with dim 1*5=5
    # OUT: 1, IN: dual(1)=1 → 1⊗1 → {0,2} with dim 2*4=8 each
    # OUT: 1, IN: dual(2)=2 → 1⊗2 → {1,3} with dim 2*5=10 each
    # OUT: 2, IN: dual(1)=1 → 2⊗1 → {1,3} with dim 3*4=12 each
    # OUT: 2, IN: dual(2)=2 → 2⊗2 → {0,2,4} with dim 3*5=15 each
    
    dim_map = combined.sector_dim_map()
    assert dim_map[0] == 8 + 15         # From 1⊗1 and 2⊗2
    assert dim_map[1] == 4 + 10 + 12    # From 0⊗1, 1⊗2, 2⊗1
    assert dim_map[2] == 5 + 8 + 15     # From 0⊗2, 1⊗1, 2⊗2
    assert dim_map[3] == 10 + 12        # From 1⊗2, 2⊗1
    assert dim_map[4] == 15             # From 2⊗2


def test_combine_indices_product_su2_stress_many_sectors():
    """Stress test combine_indices with ProductGroup(U1×SU2) having many sectors."""
    group = ProductGroup([U1Group(), SU2Group()])
    
    # First index: 5 U1 charges × 4 SU(2) spins = 20 sectors
    sectors1 = tuple(
        Sector((u1, two_j), u1 + two_j + 1)
        for u1 in range(5)
        for two_j in [0, 2, 4, 6]
    )
    idx1 = Index(Direction.OUT, group, sectors=sectors1)
    
    # Second index: 3 U1 charges × 3 SU(2) spins = 9 sectors
    sectors2 = tuple(
        Sector((u1, two_j), u1 * 2 + two_j + 2)
        for u1 in range(3)
        for two_j in [0, 2, 4]
    )
    idx2 = Index(Direction.OUT, group, sectors=sectors2)
    
    # 20 × 9 = 180 input sector pairs
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # Verify basic properties
    assert combined.group == group
    assert combined.direction == Direction.OUT
    assert len(combined.sectors) > 0
    
    # All dimensions should be positive
    for sector in combined.sectors:
        assert sector.dim > 0
    
    # Verify there's significant dimension accumulation
    # (many input pairs fuse to same output charge)
    total_combinations = 20 * 9
    # Each SU(2) fusion can produce multiple channels,
    # so output sectors > input combinations, but many U1 charges overlap
    assert len(combined.sectors) < total_combinations
    
    # Verify total dimension matches expected
    expected_total = 0
    for s1 in sectors1:
        u1_1, two_j_1 = s1.charge
        for s2 in sectors2:
            u1_2, two_j_2 = s2.charge
            # Count fusion channels
            two_j_max = two_j_1 + two_j_2
            two_j_min = abs(two_j_1 - two_j_2)
            num_channels = (two_j_max - two_j_min) // 2 + 1
            expected_total += s1.dim * s2.dim * num_channels
    
    assert combined.dim == expected_total


def test_combine_indices_product_su2_stress_z2():
    """Stress test with ProductGroup(Z2×SU2)."""
    group = ProductGroup([Z2Group(), SU2Group()])
    
    # First index: 2 Z2 values × 6 SU(2) spins = 12 sectors
    sectors1 = tuple(
        Sector((z2, two_j), z2 * 3 + two_j + 1)
        for z2 in [0, 1]
        for two_j in [0, 1, 2, 3, 4, 5]
    )
    idx1 = Index(Direction.OUT, group, sectors=sectors1)
    
    # Second index: 2 Z2 values × 5 SU(2) spins = 10 sectors
    sectors2 = tuple(
        Sector((z2, two_j), (1 - z2) * 2 + two_j + 2)
        for z2 in [0, 1]
        for two_j in [1, 2, 3, 4, 5]
    )
    idx2 = Index(Direction.OUT, group, sectors=sectors2)
    
    combined = combine_indices(Direction.OUT, idx1, idx2)
    
    # Verify properties
    assert combined.group == group
    assert combined.direction == Direction.OUT
    
    # All dimensions should be positive
    for sector in combined.sectors:
        assert sector.dim > 0
    
    # Z2 part has only 2 possible values, so significant overlap expected
    # SU(2) fusions produce multiple channels
    assert len(combined.sectors) > 0
    
    # Verify some dimension accumulation occurred
    # Total possible Z2×SU(2) combinations is limited
    max_possible_sectors = 2 * (max(5, 5+5) + 1)  # 2 Z2 values, max spin range
    assert len(combined.sectors) <= max_possible_sectors
    
    # Sanity check on total dimension
    assert combined.dim > 0


# split_index tests for SU2

def test_split_index_su2_pairwise():
    """Test split_index with SU2 group (pairwise)."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Combine and then split
    combined = combine_indices(Direction.OUT, idx1, idx2)
    parts = split_index(combined, [idx1, idx2])
    
    # Should return the original parts
    assert len(parts) == 2
    assert parts[0] == idx1
    assert parts[1] == idx2


def test_split_index_su2_more_than_two_raises():
    """Test that split_index raises for SU2 with more than 2 parts."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Create a fake parent (doesn't matter, will fail on count)
    parent = Index(Direction.OUT, group, sectors=(Sector(1, 4),))
    
    with pytest.raises(ValueError, match="only supports pairwise fusion"):
        split_index(parent, [idx1, idx2, idx3])

