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


"""Tests for tensor copy and block access operations: copy, sorted_keys, key, block."""

import numpy as np
import pytest

from nicole import Direction, Tensor, U1Group
from .utils import make_u1_index


# Copy tests

def test_copy_returns_new_instance():
    """Test that copy returns a new tensor instance."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)
    tensor = Tensor.random([idx_a, idx_b], seed=123, itags=["A", "B"])

    copied = tensor.copy()

    assert copied is not tensor


def test_copy_has_identical_data():
    """Test that copy has identical data values."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)
    tensor = Tensor.random([idx_a, idx_b], seed=456, dtype=np.complex128, itags=["A", "B"])

    copied = tensor.copy()

    assert set(copied.data.keys()) == set(tensor.data.keys())
    for key in tensor.data:
        np.testing.assert_array_equal(copied.data[key], tensor.data[key])


def test_copy_creates_independent_data():
    """Test that modifying copy doesn't affect original."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx], seed=789, itags=["X"])

    original_data = {k: v.copy() for k, v in tensor.data.items()}
    copied = tensor.copy()

    # Modify the copy's data
    for key in copied.data:
        copied.data[key] *= 100.0

    # Original should be unchanged
    for key in original_data:
        np.testing.assert_array_equal(tensor.data[key], original_data[key])


def test_copy_preserves_metadata():
    """Test that copy preserves indices, itags, dtype, and label."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 3)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 1), (-1, 2)], group)
    tensor = Tensor.random([idx_a, idx_b], seed=111, dtype=np.complex128, itags=["left", "right"])
    tensor.label = "MyTensor"

    copied = tensor.copy()

    assert copied.indices == tensor.indices
    assert copied.itags == tensor.itags
    assert copied.dtype == tensor.dtype
    assert copied.label == tensor.label


def test_copy_shares_immutable_indices():
    """Test that copy shares the same Index objects (since they're immutable)."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx], seed=222, itags=["A"])

    copied = tensor.copy()

    # Index objects should be the same (shared) since they're immutable
    for orig_idx, copy_idx in zip(tensor.indices, copied.indices):
        assert orig_idx is copy_idx


def test_copy_data_arrays_are_independent():
    """Test that numpy arrays in copy are different objects."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx], seed=333, itags=["A"])

    copied = tensor.copy()

    # Each array should be a different object
    for key in tensor.data:
        assert copied.data[key] is not tensor.data[key]


# Block access tests (sorted_keys, key, block)

def test_sorted_keys_returns_tuple():
    """Test that sorted_keys returns a tuple of BlockKeys."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)
    tensor = Tensor.random([idx_a, idx_b], seed=100, itags=["A", "B"])

    keys = tensor.sorted_keys

    assert isinstance(keys, tuple)
    assert len(keys) == len(tensor.data)
    assert set(keys) == set(tensor.data.keys())


def test_sorted_keys_is_deterministic():
    """Test that sorted_keys returns keys in consistent order."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 2), (-1, 1)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 1), (-1, 1), (1, 1)], group)
    tensor = Tensor.random([idx_a, idx_b], seed=101, itags=["A", "B"])

    # Multiple calls should return same order
    keys1 = tensor.sorted_keys
    keys2 = tensor.sorted_keys
    
    assert keys1 == keys2
    # Should be sorted by string representation
    assert keys1 == tuple(sorted(tensor.data.keys(), key=str))


def test_sorted_keys_is_cached():
    """Test that sorted_keys property is cached."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx], seed=102, itags=["A"])

    keys1 = tensor.sorted_keys
    keys2 = tensor.sorted_keys

    # Should be the same object (cached)
    assert keys1 is keys2


def test_key_returns_correct_blockkey():
    """Test that key(i) returns the correct BlockKey."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)
    tensor = Tensor.random([idx_a, idx_b], seed=103, itags=["A", "B"])

    sorted_keys = tensor.sorted_keys
    for i, expected_key in enumerate(sorted_keys, start=1):
        assert tensor.key(i) == expected_key


def test_key_raises_on_invalid_index():
    """Test that key(i) raises IndexError for invalid indices."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx], seed=104, itags=["A"])
    
    num_blocks = len(tensor.data)
    
    with pytest.raises(IndexError):
        tensor.key(0)  # 0 is invalid (1-indexed)
    
    with pytest.raises(IndexError):
        tensor.key(num_blocks + 1)  # Out of range
    
    with pytest.raises(IndexError):
        tensor.key(-1)  # Negative


def test_block_returns_correct_data():
    """Test that block(i) returns the correct data array."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)
    tensor = Tensor.random([idx_a, idx_b], seed=105, itags=["A", "B"])

    for i, key in enumerate(tensor.sorted_keys, start=1):
        np.testing.assert_array_equal(tensor.block(i), tensor.data[key])


def test_block_returns_same_object_as_data():
    """Test that block(i) returns the same array object as data[key]."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx], seed=106, itags=["A"])

    for i, key in enumerate(tensor.sorted_keys, start=1):
        assert tensor.block(i) is tensor.data[key]


def test_permute_invalidates_sorted_keys():
    """Test that permute invalidates the sorted_keys cache."""
    group = U1Group()
    # Use indices that produce asymmetric keys (e.g., (1, -1) becomes (-1, 1) after permute)
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 2), (-1, 1)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 1), (-1, 1), (1, 1)], group)
    tensor = Tensor.random([idx_a, idx_b], seed=107, itags=["A", "B"])

    # Access sorted_keys to populate cache
    old_keys = tensor.sorted_keys
    old_cache = tensor._sorted_keys
    assert old_cache is not None  # Cache should be populated
    
    # Permute changes the keys
    tensor.permute([1, 0])
    
    # Cache should be invalidated
    assert tensor._sorted_keys is None
    
    # Access new keys (this repopulates the cache)
    new_keys = tensor.sorted_keys
    
    # New keys should be valid for the permuted data
    assert set(new_keys) == set(tensor.data.keys())
    # Cache should now be repopulated
    assert tensor._sorted_keys is not None


def test_display_numbering_matches_block_index():
    """Test that display numbering is consistent with block() indexing."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 1), (1, 1), (-1, 1)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 1), (-1, 1), (1, 1)], group)
    tensor = Tensor.random([idx_a, idx_b], seed=108, itags=["A", "B"])
    
    # Get display string
    display_str = str(tensor)
    
    # The display shows blocks numbered 1, 2, 3, etc.
    # block(1), block(2), block(3) should match
    for i, key in enumerate(tensor.sorted_keys, start=1):
        # Verify the block index matches
        assert tensor.key(i) == key
        np.testing.assert_array_equal(tensor.block(i), tensor.data[key])

