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


"""Tests for core types: Direction, Sector, and Charge."""

import pytest

from nicole import Direction, Sector


def test_direction_values():
    """Test Direction enum values."""
    assert Direction.IN == -1
    assert Direction.OUT == 1


def test_direction_reverse():
    """Test Direction.reverse() method."""
    assert Direction.IN.reverse() == Direction.OUT
    assert Direction.OUT.reverse() == Direction.IN


def test_direction_double_reverse():
    """Test that reversing twice returns original direction."""
    assert Direction.IN.reverse().reverse() == Direction.IN
    assert Direction.OUT.reverse().reverse() == Direction.OUT


def test_sector_construction():
    """Test basic Sector construction."""
    s = Sector(charge=0, dim=5)
    assert s.charge == 0
    assert s.dim == 5


def test_sector_with_tuple_charge():
    """Test Sector with tuple-valued charge."""
    s = Sector(charge=(1, 2), dim=3)
    assert s.charge == (1, 2)
    assert s.dim == 3


def test_sector_with_string_charge():
    """Test Sector with string charge (hashable)."""
    s = Sector(charge="A", dim=2)
    assert s.charge == "A"
    assert s.dim == 2


def test_sector_invalid_dimension_zero():
    """Test that Sector rejects zero dimension."""
    with pytest.raises(ValueError, match="dimension must be positive"):
        Sector(charge=0, dim=0)


def test_sector_invalid_dimension_negative():
    """Test that Sector rejects negative dimension."""
    with pytest.raises(ValueError, match="dimension must be positive"):
        Sector(charge=1, dim=-5)


def test_sector_frozen():
    """Test that Sector is immutable (frozen dataclass)."""
    s = Sector(charge=0, dim=5)
    with pytest.raises(AttributeError):
        s.charge = 1  # type: ignore
    with pytest.raises(AttributeError):
        s.dim = 10  # type: ignore


def test_sector_equality():
    """Test Sector equality."""
    s1 = Sector(charge=0, dim=5)
    s2 = Sector(charge=0, dim=5)
    s3 = Sector(charge=1, dim=5)
    s4 = Sector(charge=0, dim=3)
    
    assert s1 == s2
    assert s1 != s3
    assert s1 != s4


def test_sector_hashable():
    """Test that Sector can be used in sets and as dict keys."""
    s1 = Sector(charge=0, dim=5)
    s2 = Sector(charge=1, dim=3)
    s3 = Sector(charge=0, dim=5)
    
    sector_set = {s1, s2, s3}
    assert len(sector_set) == 2  # s1 and s3 are equal
    
    sector_dict = {s1: "first", s2: "second"}
    assert sector_dict[s3] == "first"


def test_charge_type_hashable():
    """Test that various hashable types can be used as charges."""
    # Integer charge
    s1 = Sector(charge=42, dim=1)
    assert s1.charge == 42
    
    # Tuple charge
    s2 = Sector(charge=(1, 2, 3), dim=1)
    assert s2.charge == (1, 2, 3)
    
    # String charge
    s3 = Sector(charge="quantum", dim=1)
    assert s3.charge == "quantum"
    
    # Float charge (hashable but unusual)
    s4 = Sector(charge=3.14, dim=1)
    assert s4.charge == 3.14

