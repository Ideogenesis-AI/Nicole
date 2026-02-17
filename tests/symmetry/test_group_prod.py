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


def test_product_group_creation_non_abelian_at_end_allowed():
    """Test that UnitaryGroup is allowed at the end."""
    # Create a mock UnitaryGroup
    from nicole.symmetry.base import UnitaryGroup
    
    class MockUnitary(UnitaryGroup):
        @property
        def name(self):
            return "SU2"
        
        @property
        def neutral(self):
            return 0
        
        def dual(self, q):
            return q
        
        def fuse_channels(self, q1, q2):
            return (0, 1, 2)  # Mock: return some channels
        
        def equal(self, a, b):
            return a == b
        
        def validate_charge(self, q):
            pass
    
    # Should succeed: UnitaryGroup at end
    group = ProductGroup([U1Group(), MockUnitary()])
    assert group.num_components == 2


def test_product_group_creation_non_abelian_not_at_end_fails():
    """Test that UnitaryGroup not at end is rejected."""
    from nicole.symmetry.base import UnitaryGroup
    
    class MockUnitary(UnitaryGroup):
        @property
        def name(self):
            return "SU2"
        
        @property
        def neutral(self):
            return 0
        
        def dual(self, q):
            return q
        
        def fuse_channels(self, q1, q2):
            return (0, 1, 2)
        
        def equal(self, a, b):
            return a == b
        
        def validate_charge(self, q):
            pass
    
    # Should fail: UnitaryGroup not at end
    with pytest.raises(ValueError, match="must be the last component"):
        ProductGroup([MockUnitary(), U1Group()])


def test_product_group_creation_nested_fails():
    """Test that nested ProductGroups are rejected."""
    inner_group = ProductGroup([U1Group(), Z2Group()])
    
    # Should fail: nested ProductGroup not allowed
    with pytest.raises(TypeError, match="Nested ProductGroups are not allowed"):
        ProductGroup([U1Group(), inner_group])
    
    # Should also fail even if inner is first
    with pytest.raises(TypeError, match="Nested ProductGroups are not allowed"):
        ProductGroup([inner_group, U1Group()])


# Neutral element tests

def test_product_group_neutral_u1_u1():
    """Test neutral element for U1×U1."""
    group = ProductGroup([U1Group(), U1Group()])
    assert group.neutral == (0, 0)


def test_product_group_neutral_u1_z2():
    """Test neutral element for U1×Z2."""
    group = ProductGroup([U1Group(), Z2Group()])
    assert group.neutral == (0, 0)


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
    assert group.dual((0, 1)) == (0, 1)


# Fuse tests

def test_product_group_fuse_unique_two_u1_u1():
    """Test fusing two charges in U1×U1."""
    group = ProductGroup([U1Group(), U1Group()])
    assert group.fuse_unique((2, 3), (1, -1)) == (3, 2)
    assert group.fuse_unique((0, 0), (5, 7)) == (5, 7)
    assert group.fuse_unique((-2, 4), (2, -4)) == (0, 0)


def test_product_group_fuse_unique_many_u1_u1():
    """Test fusing multiple charges in U1×U1."""
    group = ProductGroup([U1Group(), U1Group()])
    assert group.fuse_unique((1, 2), (3, 4), (5, 6)) == (9, 12)
    assert group.fuse_unique((2, -1), (-1, 3), (-1, -2)) == (0, 0)


def test_product_group_fuse_unique_empty_u1_u1():
    """Test fusing no charges returns neutral."""
    group = ProductGroup([U1Group(), U1Group()])
    assert group.fuse_unique() == (0, 0)


def test_product_group_fuse_unique_u1_z2():
    """Test fusing charges in U1×Z2."""
    group = ProductGroup([U1Group(), Z2Group()])
    assert group.fuse_unique((2, 1), (3, 0)) == (5, 1)
    assert group.fuse_unique((1, 1), (2, 1)) == (3, 0)
    assert group.fuse_unique((-5, 0), (5, 1)) == (0, 1)


def test_product_group_fuse_channels_all_abelian():
    """Test fuse_channels returns single result for all-Abelian groups."""
    group = ProductGroup([U1Group(), Z2Group()])
    result = group.fuse_channels((2, 1), (3, 0))
    assert result == ((5, 1),)  # Single-element tuple
    
    result2 = group.fuse_channels((1, 1), (2, 1))
    assert result2 == ((3, 0),)


def test_product_group_fuse_channels_multi_abelian():
    """Test fuse_channels with multiple charges for all-Abelian groups."""
    group = ProductGroup([U1Group(), Z2Group()])
    
    # Three charges
    result = group.fuse_channels((2, 1), (3, 0), (1, 1))
    assert result == ((6, 0),)  # (2+3+1, (1+0+1)%2)
    
    # Four charges
    result2 = group.fuse_channels((1, 1), (2, 1), (-1, 0), (3, 1))
    assert result2 == ((5, 1),)  # (1+2-1+3, (1+1+0+1)%2)


def test_product_group_fuse_channels_with_su2():
    """Test fuse_channels with SU2Group component."""
    from nicole.symmetry.unitary import SU2Group
    
    group = ProductGroup([U1Group(), SU2Group()])
    
    # Pairwise: U1 charge 1, SU2 spins 1⊗1 → (1,0) and (1,2)
    result = group.fuse_channels((1, 1), (0, 1))
    assert result == ((1, 0), (1, 2))
    
    # Three charges: U1 charges sum to 3, SU2 spins 1⊗1⊗1 → 1,3
    result2 = group.fuse_channels((1, 1), (0, 1), (2, 1))
    assert result2 == ((3, 1), (3, 3))
    
    # Check U1 component sums correctly
    assert all(ch[0] == 3 for ch in result2)
    # Check SU2 component gives correct channels
    assert set(ch[1] for ch in result2) == {1, 3}


def test_product_group_fuse_channels_single_charge():
    """Test fuse_channels with single charge returns itself."""
    group = ProductGroup([U1Group(), Z2Group()])
    result = group.fuse_channels((2, 1))
    assert result == ((2, 1),)
    
    from nicole.symmetry.unitary import SU2Group
    group2 = ProductGroup([U1Group(), SU2Group()])
    result2 = group2.fuse_channels((3, 2))
    assert result2 == ((3, 2),)


def test_product_group_fuse_channels_empty():
    """Test fuse_channels with no charges returns neutral."""
    group = ProductGroup([U1Group(), Z2Group()])
    result = group.fuse_channels()
    assert result == ((0, 0),)


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
