# Copyright (C) 2026 Changkai Zhang.
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


"""Tests for SU2Group: SU(2) symmetry with multi-channel fusion."""

import pytest

from nicole.symmetry import SU2Group


# Basic properties tests

def test_su2_name():
    """Test SU2Group name property."""
    group = SU2Group()
    assert group.name == "SU2"


def test_su2_neutral():
    """Test SU2Group neutral element (spin-0, 2j=0)."""
    group = SU2Group()
    assert group.neutral == 0


def test_su2_dual():
    """Test SU2Group dual operation (self-dual)."""
    group = SU2Group()
    # SU(2) representations are self-dual
    assert group.dual(0) == 0    # spin-0
    assert group.dual(1) == 1    # spin-1/2
    assert group.dual(2) == 2    # spin-1
    assert group.dual(3) == 3    # spin-3/2
    assert group.dual(10) == 10  # spin-5


# Fusion channels tests

def test_su2_fuse_channels_spin_half_spin_half():
    """Test spin-1/2 ⊗ spin-1/2 (1 ⊗ 1 in 2j notation) → 0, 2."""
    group = SU2Group()
    channels = group.fuse_channels(1, 1)
    assert channels == (0, 2)


def test_su2_fuse_channels_spin_1_spin_1():
    """Test spin-1 ⊗ spin-1 (2 ⊗ 2) → 0, 2, 4."""
    group = SU2Group()
    channels = group.fuse_channels(2, 2)
    assert channels == (0, 2, 4)


def test_su2_fuse_channels_spin_2_spin_1():
    """Test spin-2 ⊗ spin-1 (4 ⊗ 2) → 2, 4, 6."""
    group = SU2Group()
    channels = group.fuse_channels(4, 2)
    assert channels == (2, 4, 6)


def test_su2_fuse_channels_spin_3half_spin_half():
    """Test spin-3/2 ⊗ spin-1/2 (3 ⊗ 1) → 2, 4."""
    group = SU2Group()
    channels = group.fuse_channels(3, 1)
    assert channels == (2, 4)


def test_su2_fuse_channels_spin_5half_spin_3half():
    """Test spin-5/2 ⊗ spin-3/2 (5 ⊗ 3) → 2, 4, 6, 8."""
    group = SU2Group()
    channels = group.fuse_channels(5, 3)
    assert channels == (2, 4, 6, 8)


def test_su2_fuse_channels_with_zero():
    """Test fusion with spin-0 (2j=0) gives single channel."""
    group = SU2Group()
    # 0 ⊗ 2j → 2j
    assert group.fuse_channels(0, 1) == (1,)   # spin-0 ⊗ spin-1/2
    assert group.fuse_channels(0, 2) == (2,)   # spin-0 ⊗ spin-1
    assert group.fuse_channels(0, 6) == (6,)   # spin-0 ⊗ spin-3
    
    # 2j ⊗ 0 → 2j
    assert group.fuse_channels(1, 0) == (1,)   # spin-1/2 ⊗ spin-0
    assert group.fuse_channels(4, 0) == (4,)   # spin-2 ⊗ spin-0


def test_su2_fuse_channels_symmetry():
    """Test that fusion is symmetric: 2j1 ⊗ 2j2 = 2j2 ⊗ 2j1."""
    group = SU2Group()
    assert group.fuse_channels(2, 4) == group.fuse_channels(4, 2)  # spin-1 ⊗ spin-2
    assert group.fuse_channels(1, 3) == group.fuse_channels(3, 1)  # spin-1/2 ⊗ spin-3/2


def test_su2_fuse_channels_count():
    """Test that the number of channels is min(2j1, 2j2) + 1."""
    group = SU2Group()
    
    # For 2j1 ⊗ 2j2, number of channels is min(2j1, 2j2) + 1
    channels = group.fuse_channels(6, 2)  # spin-3 ⊗ spin-1
    assert len(channels) == 2 + 1  # 3 channels (min is 2)
    
    channels = group.fuse_channels(5, 3)  # spin-5/2 ⊗ spin-3/2
    assert len(channels) == 3 + 1  # 4 channels (min is 3)
    
    channels = group.fuse_channels(0, 10)  # spin-0 ⊗ spin-5
    assert len(channels) == 1  # min is 0


def test_su2_fuse_channels_triangular_inequality():
    """Test triangular inequality: |2j1 - 2j2| ≤ 2j ≤ 2j1 + 2j2."""
    group = SU2Group()
    two_j1, two_j2 = 4, 3  # spin-2 and spin-3/2
    channels = group.fuse_channels(two_j1, two_j2)
    
    two_j_min = abs(two_j1 - two_j2)
    two_j_max = two_j1 + two_j2
    
    assert min(channels) == two_j_min
    assert max(channels) == two_j_max
    
    # Check all channels are in range
    for two_j in channels:
        assert two_j_min <= two_j <= two_j_max


# Equality tests

def test_su2_equal():
    """Test SU2Group equality comparison."""
    group = SU2Group()
    assert group.equal(0, 0)
    assert group.equal(1, 1)
    assert group.equal(2, 2)
    assert not group.equal(0, 1)
    assert not group.equal(1, 3)


# Charge validation tests

def test_su2_validate_charge_valid():
    """Test SU2Group validate_charge with valid charges."""
    group = SU2Group()
    # All non-negative integers are valid (representing 2j)
    group.validate_charge(0)   # spin-0
    group.validate_charge(1)   # spin-1/2
    group.validate_charge(2)   # spin-1
    group.validate_charge(3)   # spin-3/2
    group.validate_charge(10)  # spin-5
    group.validate_charge(99)  # spin-99/2


def test_su2_validate_charge_invalid_type():
    """Test SU2Group validate_charge with invalid types."""
    group = SU2Group()
    with pytest.raises(TypeError, match="must be int"):
        group.validate_charge(1.5)
    with pytest.raises(TypeError, match="must be int"):
        group.validate_charge("1")
    with pytest.raises(TypeError, match="must be int"):
        group.validate_charge([1, 2])


def test_su2_validate_charge_negative():
    """Test SU2Group validate_charge rejects negative quantum numbers."""
    group = SU2Group()
    with pytest.raises(ValueError, match="must be non-negative"):
        group.validate_charge(-1)
    with pytest.raises(ValueError, match="must be non-negative"):
        group.validate_charge(-5)


# Instance equality tests

def test_su2_group_instances_equal():
    """Test that multiple instances of SU2Group are equal."""
    group1 = SU2Group()
    group2 = SU2Group()
    assert group1 == group2


def test_su2_group_hashable():
    """Test that SU2Group is hashable."""
    group1 = SU2Group()
    group2 = SU2Group()
    
    # Can use as dict keys
    d = {group1: "value"}
    assert d[group2] == "value"  # group1 == group2


# Integration with ProductGroup tests

def test_su2_in_product_group():
    """Test SU2Group as last component in ProductGroup."""
    from nicole.symmetry import U1Group, ProductGroup
    
    # U1 × SU2 should work
    group = ProductGroup([U1Group(), SU2Group()])
    assert group.num_components == 2
    assert group.name == "U1×SU2"
    
    # Test fusion channels
    # Charges: (particle_number, 2j)
    channels = group.fuse_channels((1, 1), (2, 1))
    # U1: 1+2=3, SU2: 1⊗1 → 0,2 (spin-1/2 ⊗ spin-1/2 → spin-0, spin-1)
    assert channels == ((3, 0), (3, 2))


def test_su2_not_at_end_in_product_group_fails():
    """Test that SU2Group not at end of ProductGroup is rejected."""
    from nicole.symmetry import U1Group, ProductGroup
    
    with pytest.raises(ValueError, match="must be the last component"):
        ProductGroup([SU2Group(), U1Group()])
