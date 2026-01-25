# Copyright (C) 2026 Changkai Zhang.
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


"""Tests for ProductGroup: multiple independent symmetries."""

import pytest

from nicole.symmetry.abelian import U1Group, Z2Group
from nicole.symmetry.product import ProductGroup


# Basic ProductGroup creation tests

def test_product_group_creation_u1_u1():
    """Test creating U1×U1 product group."""
    group = ProductGroup([U1Group(), U1Group()])
    assert group.num_components == 2
    assert group.neutral == (0, 0)
    assert group.name == "U1×U1"


def test_product_group_creation_u1_z2():
    """Test creating U1×Z2 product group."""
    group = ProductGroup([U1Group(), Z2Group()])
    assert group.num_components == 2
    assert group.neutral == (0, 0)
    assert group.name == "U1×Z2"


def test_product_group_creation_three_components():
    """Test creating U1×U1×Z2 product group."""
    group = ProductGroup([U1Group(), U1Group(), Z2Group()])
    assert group.num_components == 3
    assert group.neutral == (0, 0, 0)
    assert group.name == "U1×U1×Z2"


def test_product_group_creation_empty_fails():
    """Test that creating ProductGroup with no components fails."""
    with pytest.raises(ValueError, match="at least one component"):
        ProductGroup([])


def test_product_group_creation_non_abelian_fails():
    """Test that non-Abelian groups are rejected (for now)."""
    # Create a mock non-Abelian group
    from nicole.symmetry.base import SymmetryGroup
    
    class MockNonAbelian(SymmetryGroup):
        @property
        def name(self):
            return "SU2"
        
        @property
        def neutral(self):
            return 0
        
        def inverse(self, q):
            return -q
        
        def fuse(self, *qs):
            return sum(qs)
        
        def equal(self, a, b):
            return a == b
        
        def validate_charge(self, q):
            pass
    
    mock_group = MockNonAbelian()
    with pytest.raises(ValueError, match="not an AbelianGroup"):
        ProductGroup([mock_group])


# Neutral element tests

def test_product_group_neutral_u1_u1():
    """Test neutral element for U1×U1."""
    group = ProductGroup([U1Group(), U1Group()])
    assert group.neutral == (0, 0)


def test_product_group_neutral_u1_z2():
    """Test neutral element for U1×Z2."""
    group = ProductGroup([U1Group(), Z2Group()])
    assert group.neutral == (0, 0)


# Inverse tests

def test_product_group_inverse_u1_u1():
    """Test inverse for U1×U1."""
    group = ProductGroup([U1Group(), U1Group()])
    assert group.inverse((2, 3)) == (-2, -3)
    assert group.inverse((-1, 5)) == (1, -5)
    assert group.inverse((0, 0)) == (0, 0)


def test_product_group_inverse_u1_z2():
    """Test inverse for U1×Z2."""
    group = ProductGroup([U1Group(), Z2Group()])
    assert group.inverse((3, 1)) == (-3, 1)
    assert group.inverse((-2, 0)) == (2, 0)
    assert group.inverse((0, 1)) == (0, 1)


# Fuse tests

def test_product_group_fuse_two_u1_u1():
    """Test fusing two charges in U1×U1."""
    group = ProductGroup([U1Group(), U1Group()])
    assert group.fuse((2, 3), (1, -1)) == (3, 2)
    assert group.fuse((0, 0), (5, 7)) == (5, 7)
    assert group.fuse((-2, 4), (2, -4)) == (0, 0)


def test_product_group_fuse_many_u1_u1():
    """Test fusing multiple charges in U1×U1."""
    group = ProductGroup([U1Group(), U1Group()])
    assert group.fuse((1, 2), (3, 4), (5, 6)) == (9, 12)
    assert group.fuse((2, -1), (-1, 3), (-1, -2)) == (0, 0)


def test_product_group_fuse_empty_u1_u1():
    """Test fusing no charges returns neutral."""
    group = ProductGroup([U1Group(), U1Group()])
    assert group.fuse() == (0, 0)


def test_product_group_fuse_u1_z2():
    """Test fusing charges in U1×Z2."""
    group = ProductGroup([U1Group(), Z2Group()])
    assert group.fuse((2, 1), (3, 0)) == (5, 1)
    assert group.fuse((1, 1), (2, 1)) == (3, 0)
    assert group.fuse((-5, 0), (5, 1)) == (0, 1)


# Equal tests

def test_product_group_equal_u1_u1():
    """Test equality for U1×U1."""
    group = ProductGroup([U1Group(), U1Group()])
    assert group.equal((2, 3), (2, 3))
    assert group.equal((0, 0), (0, 0))
    assert not group.equal((2, 3), (3, 2))
    assert not group.equal((1, 0), (0, 1))


def test_product_group_equal_u1_z2():
    """Test equality for U1×Z2."""
    group = ProductGroup([U1Group(), Z2Group()])
    assert group.equal((5, 1), (5, 1))
    assert group.equal((0, 0), (0, 0))
    assert not group.equal((5, 1), (5, 0))
    assert not group.equal((1, 1), (2, 1))


# Charge validation tests

def test_product_group_validate_valid_charges():
    """Test validation of valid charges."""
    group = ProductGroup([U1Group(), U1Group()])
    group.validate_charge((0, 0))
    group.validate_charge((5, -3))
    group.validate_charge((-10, 100))


def test_product_group_validate_wrong_type():
    """Test that non-tuple charges are rejected."""
    group = ProductGroup([U1Group(), U1Group()])
    with pytest.raises(TypeError, match="must be a tuple"):
        group.validate_charge(5)
    with pytest.raises(TypeError, match="must be a tuple"):
        group.validate_charge([2, 3])


def test_product_group_validate_wrong_length():
    """Test that wrong-length tuples are rejected."""
    group = ProductGroup([U1Group(), U1Group()])
    with pytest.raises(ValueError, match="length.*does not match"):
        group.validate_charge((1,))
    with pytest.raises(ValueError, match="length.*does not match"):
        group.validate_charge((1, 2, 3))


def test_product_group_validate_invalid_component():
    """Test that invalid component charges are rejected."""
    group = ProductGroup([U1Group(), Z2Group()])
    with pytest.raises(ValueError, match="Invalid charge for component"):
        group.validate_charge((3, 2))  # Z2 charge must be 0 or 1
    with pytest.raises(ValueError, match="Invalid charge for component"):
        group.validate_charge((3.5, 0))  # U1 charge must be int


# Dual tests

def test_product_group_dual_u1_u1():
    """Test dual for U1×U1."""
    group = ProductGroup([U1Group(), U1Group()])
    assert group.dual((2, 3)) == (-2, -3)
    assert group.dual((-1, 5)) == (1, -5)
    assert group.dual((0, 0)) == (0, 0)


def test_product_group_dual_u1_z2():
    """Test dual for U1×Z2."""
    group = ProductGroup([U1Group(), Z2Group()])
    assert group.dual((3, 1)) == (-3, 1)
    assert group.dual((-2, 0)) == (2, 0)


# Component access tests

def test_product_group_get_component():
    """Test accessing individual components."""
    group = ProductGroup([U1Group(), Z2Group()])
    comp0 = group.get_component(0)
    comp1 = group.get_component(1)
    assert isinstance(comp0, U1Group)
    assert isinstance(comp1, Z2Group)


def test_product_group_get_component_out_of_range():
    """Test that out-of-range component access fails."""
    group = ProductGroup([U1Group(), Z2Group()])
    with pytest.raises(IndexError, match="out of range"):
        group.get_component(2)
    with pytest.raises(IndexError, match="out of range"):
        group.get_component(-1)


# Equality and hashing tests

def test_product_group_equality():
    """Test that ProductGroup instances are equal if components match."""
    group1 = ProductGroup([U1Group(), Z2Group()])
    group2 = ProductGroup([U1Group(), Z2Group()])
    assert group1 == group2


def test_product_group_inequality_different_components():
    """Test that ProductGroup instances with different components are not equal."""
    group1 = ProductGroup([U1Group(), Z2Group()])
    group2 = ProductGroup([U1Group(), U1Group()])
    assert group1 != group2


def test_product_group_inequality_different_order():
    """Test that ProductGroup instances with different order are not equal."""
    group1 = ProductGroup([U1Group(), Z2Group()])
    group2 = ProductGroup([Z2Group(), U1Group()])
    assert group1 != group2


def test_product_group_hashable():
    """Test that ProductGroup is hashable."""
    group1 = ProductGroup([U1Group(), Z2Group()])
    group2 = ProductGroup([U1Group(), Z2Group()])
    group3 = ProductGroup([U1Group(), U1Group()])
    
    # Can use as dict keys
    d = {group1: "value1", group3: "value3"}
    assert d[group2] == "value1"  # group1 == group2


# Name tests

def test_product_group_name_u1_z2():
    """Test name property for U1×Z2."""
    group = ProductGroup([U1Group(), Z2Group()])
    assert group.name == "U1×Z2"


def test_product_group_name_z2_u1():
    """Test name property for Z2×U1."""
    group = ProductGroup([Z2Group(), U1Group()])
    assert group.name == "Z2×U1"


def test_product_group_name_three_components():
    """Test name property for U1×U1×Z2."""
    group = ProductGroup([U1Group(), U1Group(), Z2Group()])
    assert group.name == "U1×U1×Z2"
