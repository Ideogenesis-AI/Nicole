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


"""Tests for BlockSchema utility class."""

import numpy as np
import pytest

from nicole import Direction, Index, Sector, U1Group, Z2Group
from nicole.blocks import BlockSchema


def test_iter_admissible_keys_simple():
    """Test BlockSchema.iter_admissible_keys with simple indices."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(-1, 2)))
    
    keys = list(BlockSchema.iter_admissible_keys([idx1, idx2]))
    
    # Cartesian product: (0,0), (0,-1), (1,0), (1,-1)
    assert set(keys) == {(0, 0), (0, -1), (1, 0), (1, -1)}


def test_iter_admissible_keys_single_index():
    """Test BlockSchema.iter_admissible_keys with single index."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    keys = list(BlockSchema.iter_admissible_keys([idx]))
    
    assert set(keys) == {(0,), (1,)}


def test_iter_admissible_keys_three_indices():
    """Test BlockSchema.iter_admissible_keys with three indices."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))
    
    keys = list(BlockSchema.iter_admissible_keys([idx1, idx2, idx3]))
    
    # All combinations: 2^3 = 8
    assert len(keys) == 8


def test_iter_admissible_keys_empty():
    """Test BlockSchema.iter_admissible_keys with empty indices."""
    keys = list(BlockSchema.iter_admissible_keys([]))
    
    # Empty cartesian product yields one empty tuple
    assert keys == [()]


def test_shape_for_key():
    """Test BlockSchema.shape_for_key."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(-1, 4)))
    
    shape1 = BlockSchema.shape_for_key([idx1, idx2], (0, 0))
    assert shape1 == (2, 5)
    
    shape2 = BlockSchema.shape_for_key([idx1, idx2], (1, -1))
    assert shape2 == (3, 4)


def test_shape_for_key_single():
    """Test BlockSchema.shape_for_key with single index."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    shape = BlockSchema.shape_for_key([idx], (1,))
    assert shape == (3,)


def test_shape_for_key_invalid_length():
    """Test BlockSchema.shape_for_key with mismatched key length."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    with pytest.raises(ValueError, match="length does not match"):
        BlockSchema.shape_for_key([idx], (0, 1))


def test_shape_for_key_missing_charge():
    """Test BlockSchema.shape_for_key with charge not in index."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    with pytest.raises(KeyError, match="not present"):
        BlockSchema.shape_for_key([idx], (1,))


def test_validate_blocks_valid():
    """Test BlockSchema.validate_blocks with valid blocks."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(-1, 4)))
    
    blocks = {
        (0, 0): np.zeros((2, 5)),
        (1, -1): np.zeros((3, 4))
    }
    
    # Should not raise
    BlockSchema.validate_blocks([idx1, idx2], blocks)


def test_validate_blocks_wrong_shape():
    """Test BlockSchema.validate_blocks with wrong shape."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5),))
    
    blocks = {
        (0, 0): np.zeros((2, 3))  # Wrong shape, should be (2, 5)
    }
    
    with pytest.raises(ValueError, match="expected"):
        BlockSchema.validate_blocks([idx1, idx2], blocks)


def test_validate_blocks_not_numpy():
    """Test BlockSchema.validate_blocks with non-numpy arrays."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    blocks = {
        (0,): [[1, 2], [3, 4]]  # List, not numpy array
    }
    
    with pytest.raises(TypeError, match="numpy arrays"):
        BlockSchema.validate_blocks([idx], blocks)


def test_charge_totals_neutral():
    """Test BlockSchema.charge_totals for neutral block."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(1, 4)))
    
    # Block (1, 1): OUT(1) + IN(1) = 1 + inverse(1) = 1 + (-1) = 0
    totals = BlockSchema.charge_totals([idx1, idx2], (1, 1))
    
    assert totals[group] == 0


def test_charge_totals_non_neutral():
    """Test BlockSchema.charge_totals for non-neutral block."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(-1, 4)))
    
    # Block (1, 0): OUT(1) + IN(0) = 1 + 0 = 1 (not neutral)
    totals = BlockSchema.charge_totals([idx1, idx2], (1, 0))
    
    assert totals[group] == 1


def test_charge_totals_multiple_out():
    """Test BlockSchema.charge_totals with multiple OUT indices."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(2, 3),))
    idx3 = Index(Direction.IN, group, sectors=(Sector(3, 4),))
    
    # OUT(1) + OUT(2) + IN(3) = 1 + 2 - 3 = 0
    totals = BlockSchema.charge_totals([idx1, idx2, idx3], (1, 2, 3))
    
    assert totals[group] == 0


def test_charge_totals_z2():
    """Test BlockSchema.charge_totals with Z2 group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))
    
    # OUT(1) + IN(1) = 1 XOR 1 = 0
    totals = BlockSchema.charge_totals([idx1, idx2], (1, 1))
    
    assert totals[group] == 0


def test_charges_conserved_true():
    """Test BlockSchema.charges_conserved returns True for neutral."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(1, 4)))
    
    assert BlockSchema.charges_conserved([idx1, idx2], (0, 0))
    assert BlockSchema.charges_conserved([idx1, idx2], (1, 1))


def test_charges_conserved_false():
    """Test BlockSchema.charges_conserved returns False for non-neutral."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(-1, 4)))
    
    assert not BlockSchema.charges_conserved([idx1, idx2], (1, 0))
    assert not BlockSchema.charges_conserved([idx1, idx2], (0, -1))


def test_charges_conserved_z2():
    """Test BlockSchema.charges_conserved with Z2 group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))
    
    # Conservation: q1 XOR q2 XOR q3 = 0
    assert BlockSchema.charges_conserved([idx1, idx2, idx3], (0, 0, 0))
    assert BlockSchema.charges_conserved([idx1, idx2, idx3], (1, 1, 0))
    assert BlockSchema.charges_conserved([idx1, idx2, idx3], (1, 0, 1))
    assert BlockSchema.charges_conserved([idx1, idx2, idx3], (0, 1, 1))
    
    assert not BlockSchema.charges_conserved([idx1, idx2, idx3], (1, 0, 0))
    assert not BlockSchema.charges_conserved([idx1, idx2, idx3], (0, 1, 0))


def test_block_schema_empty_indices():
    """Test BlockSchema methods with empty indices."""
    keys = list(BlockSchema.iter_admissible_keys([]))
    assert keys == [()]
    
    shape = BlockSchema.shape_for_key([], ())
    assert shape == ()
    
    # Empty blocks dict should be valid
    BlockSchema.validate_blocks([], {})


def test_validate_blocks_empty_dict():
    """Test BlockSchema.validate_blocks with empty blocks dict."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    # Should not raise
    BlockSchema.validate_blocks([idx], {})

