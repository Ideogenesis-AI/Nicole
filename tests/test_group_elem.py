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


"""Tests for symmetry groups: U1Group and Z2Group."""

import pytest

from nicole import U1Group, Z2Group


# U1Group tests

def test_u1_neutral():
    """Test U1Group neutral element."""
    group = U1Group()
    assert group.neutral == 0


def test_u1_inverse():
    """Test U1Group inverse operation."""
    group = U1Group()
    assert group.inverse(4) == -4
    assert group.inverse(-3) == 3
    assert group.inverse(0) == 0


def test_u1_fuse_two():
    """Test U1Group fuse with two charges."""
    group = U1Group()
    assert group.fuse(2, 3) == 5
    assert group.fuse(-1, 4) == 3
    assert group.fuse(0, 0) == 0


def test_u1_fuse_many():
    """Test U1Group fuse with multiple charges."""
    group = U1Group()
    assert group.fuse(2, 3, -5) == 0
    assert group.fuse(1, 2, 3, 4) == 10
    assert group.fuse() == 0  # Empty fusion gives neutral


def test_u1_fuse_many_method():
    """Test U1Group fuse_many method."""
    group = U1Group()
    assert group.fuse_many([2, 3, -5]) == 0
    assert group.fuse_many([1, 2, 3, 4]) == 10
    assert group.fuse_many([]) == 0


def test_u1_equal():
    """Test U1Group equality comparison."""
    group = U1Group()
    assert group.equal(7, 7)
    assert group.equal(0, 0)
    assert not group.equal(7, 8)
    assert not group.equal(-1, 1)


def test_u1_validate_charge_valid():
    """Test U1Group validate_charge with valid charges."""
    group = U1Group()
    group.validate_charge(0)  # Should not raise
    group.validate_charge(5)
    group.validate_charge(-10)


def test_u1_validate_charge_invalid():
    """Test U1Group validate_charge with invalid charges."""
    group = U1Group()
    with pytest.raises(TypeError, match="must be an int"):
        group.validate_charge(3.5)
    with pytest.raises(TypeError, match="must be an int"):
        group.validate_charge("string")
    with pytest.raises(TypeError, match="must be an int"):
        group.validate_charge([1, 2])


def test_u1_dual():
    """Test U1Group dual (should be same as inverse for Abelian)."""
    group = U1Group()
    assert group.dual(5) == -5
    assert group.dual(-3) == 3
    assert group.dual(0) == 0


def test_u1_name():
    """Test U1Group name property."""
    group = U1Group()
    assert group.name == "U1"


# Z2Group tests

def test_z2_neutral():
    """Test Z2Group neutral element."""
    group = Z2Group()
    assert group.neutral == 0


def test_z2_inverse():
    """Test Z2Group inverse operation."""
    group = Z2Group()
    assert group.inverse(0) == 0
    assert group.inverse(1) == 1


def test_z2_fuse_two():
    """Test Z2Group fuse with two charges."""
    group = Z2Group()
    assert group.fuse(0, 0) == 0
    assert group.fuse(0, 1) == 1
    assert group.fuse(1, 0) == 1
    assert group.fuse(1, 1) == 0


def test_z2_fuse_many():
    """Test Z2Group fuse with multiple charges."""
    group = Z2Group()
    assert group.fuse(1, 1) == 0
    assert group.fuse(1, 0, 1) == 0
    assert group.fuse(1, 1, 1) == 1
    assert group.fuse(0, 0, 0) == 0
    assert group.fuse() == 0


def test_z2_fuse_many_method():
    """Test Z2Group fuse_many method."""
    group = Z2Group()
    assert group.fuse_many([1, 1]) == 0
    assert group.fuse_many([1, 0, 1]) == 0
    assert group.fuse_many([1, 1, 1]) == 1
    assert group.fuse_many([]) == 0


def test_z2_equal():
    """Test Z2Group equality comparison."""
    group = Z2Group()
    assert group.equal(0, 0)
    assert group.equal(1, 1)
    assert not group.equal(0, 1)
    assert not group.equal(1, 0)


def test_z2_validate_charge_valid():
    """Test Z2Group validate_charge with valid charges."""
    group = Z2Group()
    group.validate_charge(0)  # Should not raise
    group.validate_charge(1)


def test_z2_validate_charge_invalid_type():
    """Test Z2Group validate_charge with invalid type."""
    group = Z2Group()
    with pytest.raises(TypeError, match="must be an int"):
        group.validate_charge("1")
    with pytest.raises(TypeError, match="must be an int"):
        group.validate_charge(1.0)


def test_z2_validate_charge_invalid_value():
    """Test Z2Group validate_charge with invalid value."""
    group = Z2Group()
    with pytest.raises(ValueError, match="must be 0 or 1"):
        group.validate_charge(2)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        group.validate_charge(-1)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        group.validate_charge(10)


def test_z2_dual():
    """Test Z2Group dual (should be same as inverse)."""
    group = Z2Group()
    assert group.dual(0) == 0
    assert group.dual(1) == 1


def test_z2_name():
    """Test Z2Group name property."""
    group = Z2Group()
    assert group.name == "Z2"


# Cross-group tests

def test_u1_z2_are_different():
    """Test that U1Group and Z2Group are distinct objects."""
    u1 = U1Group()
    z2 = Z2Group()
    assert u1 != z2
    assert u1.name != z2.name


def test_group_instances_are_equal():
    """Test that multiple instances of the same group are equal."""
    u1a = U1Group()
    u1b = U1Group()
    assert u1a == u1b
    
    z2a = Z2Group()
    z2b = Z2Group()
    assert z2a == z2b

