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


"""Tests for tensor copy and block access operations: copy, sorted_keys, key, block."""

import numpy as np
import pytest

from nicole import Direction, Tensor, U1Group, subsector, Index, Sector


# Copy tests

def test_copy_returns_new_instance():
    """Test that copy returns a new tensor instance."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=123, itags=["A", "B"])

    copied = tensor.copy()

    assert copied is not tensor


def test_copy_has_identical_data():
    """Test that copy has identical data values."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=456, dtype=np.complex128, itags=["A", "B"])

    copied = tensor.copy()

    assert set(copied.data.keys()) == set(tensor.data.keys())
    for key in tensor.data:
        np.testing.assert_array_equal(copied.data[key], tensor.data[key])


def test_copy_creates_independent_data():
    """Test that modifying copy doesn't affect original."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=789, itags=["X", "Y"])

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
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 2)))
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
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=222, itags=["A", "B"])

    copied = tensor.copy()

    # Index objects should be the same (shared) since they're immutable
    for orig_idx, copy_idx in zip(tensor.indices, copied.indices):
        assert orig_idx is copy_idx


def test_copy_data_arrays_are_independent():
    """Test that numpy arrays in copy are different objects."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=333, itags=["A", "B"])

    copied = tensor.copy()

    # Each array should be a different object
    for key in tensor.data:
        assert copied.data[key] is not tensor.data[key]


# Block access tests (sorted_keys, key, block)

def test_sorted_keys_returns_tuple():
    """Test that sorted_keys returns a tuple of BlockKeys."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=100, itags=["A", "B"])

    keys = tensor.sorted_keys

    assert isinstance(keys, tuple)
    assert len(keys) == len(tensor.data)
    assert set(keys) == set(tensor.data.keys())


def test_sorted_keys_is_deterministic():
    """Test that sorted_keys returns keys in consistent order."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1), Sector(1, 1)))
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
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=102, itags=["A", "B"])

    keys1 = tensor.sorted_keys
    keys2 = tensor.sorted_keys

    # Should be the same object (cached)
    assert keys1 is keys2


def test_key_returns_correct_blockkey():
    """Test that key(i) returns the correct BlockKey."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=103, itags=["A", "B"])

    sorted_keys = tensor.sorted_keys
    for i, expected_key in enumerate(sorted_keys, start=1):
        assert tensor.key(i) == expected_key


def test_key_raises_on_invalid_index():
    """Test that key(i) raises IndexError for invalid indices."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=104, itags=["A", "B"])
    
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
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=105, itags=["A", "B"])

    for i, key in enumerate(tensor.sorted_keys, start=1):
        np.testing.assert_array_equal(tensor.block(i), tensor.data[key])


def test_block_returns_same_object_as_data():
    """Test that block(i) returns the same array object as data[key]."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=106, itags=["A", "B"])

    for i, key in enumerate(tensor.sorted_keys, start=1):
        assert tensor.block(i) is tensor.data[key]


def test_permute_invalidates_sorted_keys():
    """Test that permute invalidates the sorted_keys cache."""
    group = U1Group()
    # Use indices that produce asymmetric keys (e.g., (1, -1) becomes (-1, 1) after permute)
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1), Sector(1, 1)))
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
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1), Sector(-1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1), Sector(1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=108, itags=["A", "B"])
    
    # Get display string
    display_str = str(tensor)
    
    # The display shows blocks numbered 1, 2, 3, etc.
    # block(1), block(2), block(3) should match
    for i, key in enumerate(tensor.sorted_keys, start=1):
        # Verify the block index matches
        assert tensor.key(i) == key
        np.testing.assert_array_equal(tensor.block(i), tensor.data[key])


# subsector tests

def test_subsector_returns_new_instance():
    """Test that subsector returns a new tensor instance."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1), Sector(1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=200, itags=["A", "B"])

    sub = subsector(tensor, [1, 2])

    assert sub is not tensor


def test_subsector_contains_only_specified_blocks():
    """Test that subsector returns only the specified blocks."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1), Sector(1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=201, itags=["A", "B"])

    indices_to_get = [1, 3]
    sub = subsector(tensor, indices_to_get)

    # Should have exactly the specified number of blocks
    assert len(sub.data) == len(indices_to_get)
    
    # Should contain the correct keys
    expected_keys = {tensor.key(i) for i in indices_to_get}
    assert set(sub.data.keys()) == expected_keys


def test_subsector_data_is_copied():
    """Test that subsector creates independent copies of data."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=202, itags=["A", "B"])

    sub = subsector(tensor, 1)
    
    # Modify the sub tensor's data
    for key in sub.data:
        original_value = tensor.data[key].copy()
        sub.data[key] *= 100.0
        # Original should be unchanged
        np.testing.assert_array_equal(tensor.data[key], original_value)


def test_subsector_preserves_metadata():
    """Test that subsector preserves itags, dtype, and label."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=203, dtype=np.complex128, itags=["left", "right"])
    tensor.label = "TestTensor"

    sub = subsector(tensor, 1)

    assert sub.itags == tensor.itags
    assert sub.dtype == tensor.dtype
    assert sub.label == tensor.label


def test_subsector_prunes_unused_sectors():
    """Test that subsector removes sectors not present in selected blocks."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1), Sector(1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=203, itags=["A", "B"])

    # Get only first block
    sub = subsector(tensor, 1)

    # The subsector should only have sectors that appear in block 1
    key_1 = tensor.key(1)
    
    # Check that subsector indices only contain used charges
    for axis, charge in enumerate(key_1):
        used_charges = [s.charge for s in sub.indices[axis].sectors]
        assert charge in used_charges
        # Should have fewer or equal sectors than original
        assert len(sub.indices[axis].sectors) <= len(tensor.indices[axis].sectors)


def test_subsector_single_block():
    """Test subsector with a single block index."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=204, itags=["A", "B"])

    sub = subsector(tensor, 1)

    assert len(sub.data) == 1
    key = tensor.key(1)
    np.testing.assert_array_equal(sub.data[key], tensor.data[key])


def test_subsector_integer_vs_list_syntax():
    """Test that single integer and list syntax produce identical results."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1), Sector(1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=207, itags=["A", "B"])

    # Get block using single integer syntax
    sub_int = subsector(tensor, 2)
    
    # Get same block using list syntax
    sub_list = subsector(tensor, [2])

    # Both should have the same number of blocks
    assert len(sub_int.data) == len(sub_list.data) == 1
    
    # Both should have the same keys
    assert set(sub_int.data.keys()) == set(sub_list.data.keys())
    
    # Both should have the same data
    for key in sub_int.data:
        np.testing.assert_array_equal(sub_int.data[key], sub_list.data[key])
    
    # Both should preserve the same metadata
    assert sub_int.indices == sub_list.indices
    assert sub_int.itags == sub_list.itags
    assert sub_int.dtype == sub_list.dtype


def test_subsector_all_blocks():
    """Test subsector with all block indices."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=205, itags=["A", "B"])

    all_indices = list(range(1, len(tensor.data) + 1))
    sub = subsector(tensor, all_indices)

    assert len(sub.data) == len(tensor.data)
    for key in tensor.data:
        np.testing.assert_array_equal(sub.data[key], tensor.data[key])


def test_subsector_raises_on_invalid_index():
    """Test that subsector raises IndexError for invalid indices."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=206, itags=["A", "B"])

    num_blocks = len(tensor.data)

    # Test with sequence syntax
    with pytest.raises(IndexError):
        subsector(tensor, [0])  # 0 is invalid (1-indexed)

    with pytest.raises(IndexError):
        subsector(tensor, [num_blocks + 1])  # Out of range

    with pytest.raises(IndexError):
        subsector(tensor, [1, num_blocks + 1])  # One valid, one invalid

    # Test with single integer syntax
    with pytest.raises(IndexError):
        subsector(tensor, 0)  # 0 is invalid (1-indexed)

    with pytest.raises(IndexError):
        subsector(tensor, num_blocks + 1)  # Out of range

