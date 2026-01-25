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


"""Tests for tensor manipulation operations: conj, permute, transpose, retag."""

import numpy as np
import pytest

from nicole import Direction, Index, Sector, Tensor
from nicole import conj, permute, transpose, merge_axes, contract
from nicole import ProductGroup, U1Group, Z2Group


# Conjugation tests

def test_conj_functional_returns_new_instance():
    """Test that functional conj returns a new instance."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=123, dtype=np.complex128, itags=["A", "B"])

    tensor_conj = conj(tensor)
    
    assert tensor_conj is not tensor  # Verify it's a new instance


def test_conj_functional_conjugates_data():
    """Test that functional conj conjugates the data."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=123, dtype=np.complex128, itags=["A", "B"])

    tensor_conj = conj(tensor)
    
    for key in tensor.data:
        np.testing.assert_allclose(tensor_conj.data[key], np.conjugate(tensor.data[key]))


def test_conj_functional_flips_directions():
    """Test that functional conj flips index directions."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=123, dtype=np.complex128, itags=["A", "B"])

    tensor_conj = conj(tensor)
    
    for orig_idx, new_idx in zip(tensor.indices, tensor_conj.indices):
        assert new_idx.direction == orig_idx.direction.reverse()


def test_conj_inplace_modifies_original():
    """Test that in-place conj modifies the original tensor."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=1, dtype=np.complex128, itags=["A", "B"])
    
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    original_direction = tensor.indices[0].direction
    
    tensor.conj()
    
    # Verify data was conjugated
    for key in original_data:
        np.testing.assert_allclose(tensor.data[key], np.conjugate(original_data[key]))
    
    # Verify direction was flipped
    assert tensor.indices[0].direction == original_direction.reverse()


def test_conj_real_dtype_no_data_change():
    """Test that conj on real dtype doesn't change data."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=1, dtype=np.float64, itags=["A", "B"])
    
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    result = conj(tensor)
    
    # Data should be unchanged (just copied) for real dtype
    for key in original_data:
        np.testing.assert_allclose(result.data[key], original_data[key])


def test_conj_double_application():
    """Test that conjugating twice returns to original."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=1, dtype=np.complex128, itags=["A", "B"])
    
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    original_direction = tensor.indices[0].direction
    
    double_conj = conj(conj(tensor))
    
    for key in original_data:
        np.testing.assert_allclose(double_conj.data[key], original_data[key])
    
    assert double_conj.indices[0].direction == original_direction


# Permutation tests

def test_permute_functional_returns_new_instance():
    """Test that functional permute returns a new instance."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 1))),
    ]
    tensor = Tensor.random(indices, seed=10, itags=["a", "b", "c"])
    
    permuted = permute(tensor, [2, 0, 1])
    
    assert permuted is not tensor


def test_permute_reorders_indices_and_blocks():
    """Test that permute correctly reorders indices and blocks."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 1))),
        Index(Direction.IN, group, sectors=(Sector(0, 1),)),
    ]
    itags = ["a", "b", "c", "d"]

    tensor = Tensor.random(indices, seed=10, itags=itags)
    order = [2, 0, 3, 1]
    
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    permuted = permute(tensor, order)

    assert list(permuted.itags) == [itags[i] for i in order]
    for key, block in original_data.items():
        new_key = tuple(key[i] for i in order)
        np.testing.assert_allclose(
            permuted.data[new_key],
            np.transpose(block, axes=order),
        )


def test_permute_inplace():
    """Test in-place permute method."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 1),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
    ]
    tensor = Tensor.random(indices, seed=1, itags=["a", "b"])
    
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    tensor.permute([1, 0])
    
    assert list(tensor.itags) == ["b", "a"]
    for key, block in original_data.items():
        new_key = (key[1], key[0])
        np.testing.assert_allclose(tensor.data[new_key], np.transpose(block, axes=[1, 0]))


def test_permute_identity():
    """Test that identity permutation leaves tensor unchanged."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx, idx], seed=1, itags=["a", "b", "c"])
    
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    permuted = permute(tensor, [0, 1, 2])
    
    assert list(permuted.itags) == ["a", "b", "c"]
    for key in original_data:
        np.testing.assert_allclose(permuted.data[key], original_data[key])


def test_permute_invalid_order():
    """Test that invalid permutation raises error."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    with pytest.raises(ValueError, match="Invalid permutation"):
        permute(tensor, [0, 0])
    
    with pytest.raises(ValueError, match="Invalid permutation"):
        permute(tensor, [0, 2])


# Transpose tests

def test_transpose_functional_returns_new_instance():
    """Test that functional transpose returns a new instance."""
    group = U1Group()
    indices = [Index(Direction.OUT, group, sectors=(Sector(0, 1),)) for _ in range(3)]
    tensor = Tensor.random(indices, seed=1, itags=["a", "b", "c"])
    
    transposed = transpose(tensor)
    
    assert transposed is not tensor


def test_transpose_default_reverses_order():
    """Test that transpose with no args reverses order."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 1))),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1))),
    ]
    itags = ["i0", "i1", "i2"]
    tensor = Tensor.random(indices, seed=11, itags=itags)
    
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    transposed = transpose(tensor)

    assert list(transposed.itags) == list(reversed(itags))
    for key, block in original_data.items():
        new_key = tuple(reversed(key))
        np.testing.assert_allclose(
            transposed.data[new_key],
            np.transpose(block, axes=(2, 1, 0)),
        )


def test_transpose_with_explicit_order():
    """Test transpose with explicit order."""
    group = U1Group()
    indices = [Index(Direction.OUT, group, sectors=(Sector(0, 1),)) for _ in range(3)]
    tensor = Tensor.random(indices, seed=1, itags=["a", "b", "c"])
    
    transposed = transpose(tensor, 1, 0, 2)
    
    assert list(transposed.itags) == ["b", "a", "c"]


def test_transpose_inplace():
    """Test in-place transpose method."""
    group = U1Group()
    indices = [Index(Direction.OUT, group, sectors=(Sector(0, 1),)) for _ in range(2)]
    tensor = Tensor.random(indices, seed=1, itags=["a", "b"])
    
    original_itags = list(tensor.itags)
    
    tensor.transpose()
    
    assert list(tensor.itags) == list(reversed(original_itags))


def test_transpose_double_application():
    """Test that transposing twice returns to original."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    double_transpose = transpose(transpose(tensor))
    
    assert list(double_transpose.itags) == ["a", "b"]
    for key in original_data:
        np.testing.assert_allclose(double_transpose.data[key], original_data[key])


# Retag tests

def test_retag_mode1_mapping():
    """Test retag with mapping dictionary (mode 1)."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 1),)),
    ]
    itags = ["x", "y", "z"]
    
    tensor = Tensor.random(indices, seed=12, itags=itags)
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    result = tensor.retag({"x": "left", "y": "right"})
    
    assert result is None  # retag() is in-place
    assert list(tensor.itags) == ["left", "right", "z"]
    for key in original_data:
        np.testing.assert_allclose(tensor.data[key], original_data[key])


def test_retag_mode2_full_replacement():
    """Test retag with full replacement (mode 2)."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 1),)),
    ]
    itags = ["x", "y", "z"]
    
    tensor = Tensor.random(indices, seed=13, itags=itags)
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    result = tensor.retag(["a", "b", "c"])
    
    assert result is None
    assert list(tensor.itags) == ["a", "b", "c"]
    for key in original_data:
        np.testing.assert_allclose(tensor.data[key], original_data[key])


def test_retag_mode3_selective_update():
    """Test retag with selective update by index (mode 3)."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 1),)),
    ]
    itags = ["x", "y", "z"]
    
    tensor = Tensor.random(indices, seed=14, itags=itags)
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    result = tensor.retag([0, 2], ["first", "third"])
    
    assert result is None
    assert list(tensor.itags) == ["first", "y", "third"]
    for key in original_data:
        np.testing.assert_allclose(tensor.data[key], original_data[key])


def test_retag_single_int_and_str():
    """Test retag with single int and single str (mode 3 simplified)."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 1),)),
    ]
    itags = ["x", "y", "z"]
    
    tensor = Tensor.random(indices, seed=15, itags=itags)
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    # Test single int with single str
    result = tensor.retag(1, "middle")
    
    assert result is None
    assert list(tensor.itags) == ["x", "middle", "z"]
    for key in original_data:
        np.testing.assert_allclose(tensor.data[key], original_data[key])
    
    # Test that it works with another index
    tensor.retag(0, "left")
    assert list(tensor.itags) == ["left", "middle", "z"]


def test_retag_mixed_single_and_sequence():
    """Test retag with int and sequence of strings."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 1),)),
    ]
    
    tensor = Tensor.random(indices, seed=16, itags=["a", "b", "c"])
    
    # Single int with sequence of strings should raise ValueError
    with pytest.raises(ValueError):
        tensor.retag(0, ["x", "y"])
    
    # Sequence of ints with single string should raise ValueError  
    with pytest.raises(ValueError):
        tensor.retag([0, 1], "x")


def test_retag_mapping_unmapped_tags_preserved():
    """Test that unmapped tags are preserved in mode 1."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx, idx], seed=1, itags=["a", "b", "c"])
    
    tensor.retag({"a": "alpha"})
    
    assert list(tensor.itags) == ["alpha", "b", "c"]


def test_retag_mode2_wrong_count_raises():
    """Test that mode 2 with wrong count raises error."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    with pytest.raises(ValueError, match="must match number of indices"):
        tensor.retag(["x", "y", "z"])


def test_retag_mode3_wrong_count_raises():
    """Test that mode 3 with mismatched counts raises error."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    with pytest.raises(ValueError, match="must match number of new tags"):
        tensor.retag([0], ["x", "y"])


def test_retag_mode3_out_of_range_raises():
    """Test that mode 3 with out of range index raises error."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    with pytest.raises(IndexError, match="out of range"):
        tensor.retag([5], ["x"])


def test_retag_preserves_tensor_data():
    """Test that retag never modifies tensor data or structure."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    tensor = Tensor.random([idx, idx.flip()], seed=1, itags=["original", "second"])
    
    original_norm = tensor.norm()
    original_keys = set(tensor.data.keys())
    
    tensor.retag(["new_name", "new_second"])
    
    assert tensor.norm() == original_norm
    assert set(tensor.data.keys()) == original_keys


# Flip tests

def test_flip_single_index():
    """Test flipping a single index."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_direction_0 = tensor.indices[0].direction
    original_direction_1 = tensor.indices[1].direction
    original_charges_0 = tensor.indices[0].charges()
    original_charges_1 = tensor.indices[1].charges()
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    # Flip first index
    tensor.flip(0)
    
    # Verify direction changed for index 0
    assert tensor.indices[0].direction == original_direction_0.reverse()
    # Verify direction unchanged for index 1
    assert tensor.indices[1].direction == original_direction_1
    # Verify charges conjugated for index 0 (using dual)
    assert tensor.indices[0].charges() == tuple(group.dual(c) for c in original_charges_0)
    # Verify charges unchanged for index 1
    assert tensor.indices[1].charges() == original_charges_1
    # Verify data unchanged
    for key in original_data:
        np.testing.assert_allclose(tensor.data[key], original_data[key])


def test_flip_multiple_indices():
    """Test flipping multiple indices at once."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
    
    tensor = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    original_directions = [idx.direction for idx in tensor.indices]
    original_charges = [idx.charges() for idx in tensor.indices]
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    # Flip indices 0 and 2
    tensor.flip([0, 2])
    
    # Verify directions changed for indices 0 and 2
    assert tensor.indices[0].direction == original_directions[0].reverse()
    assert tensor.indices[1].direction == original_directions[1]  # unchanged
    assert tensor.indices[2].direction == original_directions[2].reverse()
    # Verify charges conjugated for indices 0 and 2, unchanged for index 1
    assert tensor.indices[0].charges() == tuple(group.dual(c) for c in original_charges[0])
    assert tensor.indices[1].charges() == original_charges[1]
    assert tensor.indices[2].charges() == tuple(group.dual(c) for c in original_charges[2])
    # Verify data unchanged
    for key in original_data:
        np.testing.assert_allclose(tensor.data[key], original_data[key])


def test_flip_uses_dual():
    """Test that tensor.flip() uses Index.dual() to maintain charge conservation."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(-1, 2)))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    
    # Get the original index
    original_charges = tensor.indices[0].charges()
    original_direction = tensor.indices[0].direction
    
    # Flip the tensor index
    tensor.flip(0)
    
    # Tensor.flip() should use Index.dual() internally
    # This means both direction is reversed AND charges are conjugated
    assert tensor.indices[0].direction == original_direction.reverse()
    assert tensor.indices[0].charges() == tuple(group.dual(c) for c in original_charges)
    assert tensor.indices[0].charges() == (0, -1, 1)  # For U1, dual(q) = -q


def test_flip_updates_block_keys():
    """Test that flip updates block keys to reflect conjugated charges."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_keys = set(tensor.data.keys())
    
    # Store data arrays by their original keys
    original_data_by_key = {k: v.copy() for k, v in tensor.data.items()}
    
    # Flip first index
    tensor.flip(0)
    
    # Block keys should be updated: charge at position 0 should be conjugated
    # For U1: dual(0) = 0, dual(1) = -1
    expected_keys = set()
    for old_key in original_keys:
        new_key = (group.dual(old_key[0]), old_key[1])
        expected_keys.add(new_key)
    
    assert set(tensor.data.keys()) == expected_keys
    
    # Verify that data arrays are still the same, just under new keys
    for old_key, old_arr in original_data_by_key.items():
        new_key = (group.dual(old_key[0]), old_key[1])
        assert new_key in tensor.data
        np.testing.assert_allclose(tensor.data[new_key], old_arr)


def test_flip_multiple_indices_updates_keys():
    """Test that flipping multiple indices updates all relevant positions in block keys."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
    
    tensor = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    original_keys = set(tensor.data.keys())
    
    # Flip indices 0 and 2
    tensor.flip([0, 2])
    
    # Block keys should be updated at positions 0 and 2
    expected_keys = set()
    for old_key in original_keys:
        new_key = (group.dual(old_key[0]), old_key[1], group.dual(old_key[2]))
        expected_keys.add(new_key)
    
    assert set(tensor.data.keys()) == expected_keys


def test_flip_all_indices():
    """Test flipping all indices."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_directions = [idx.direction for idx in tensor.indices]
    
    # Flip all indices
    tensor.flip([0, 1])
    
    # All directions should be reversed
    for i, orig_dir in enumerate(original_directions):
        assert tensor.indices[i].direction == orig_dir.reverse()


def test_flip_inplace_modification():
    """Test that flip modifies tensor in-place."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    original_id = id(tensor)
    
    result = tensor.flip(0)
    
    # Should return None (in-place operation)
    assert result is None
    # Tensor object should be the same
    assert id(tensor) == original_id


def test_flip_preserves_data():
    """Test that flip preserves all tensor data."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_norm = tensor.norm()
    original_keys = set(tensor.data.keys())
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    tensor.flip([0, 1])
    
    # Verify norm preserved
    assert np.isclose(tensor.norm(), original_norm)
    # Verify keys unchanged
    assert set(tensor.data.keys()) == original_keys
    # Verify data values unchanged
    for key in original_data:
        np.testing.assert_allclose(tensor.data[key], original_data[key])


def test_flip_preserves_itags():
    """Test that flip preserves index tags."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip(), idx], seed=42, itags=["x", "y", "z"])
    original_itags = list(tensor.itags)
    
    tensor.flip([0, 2])
    
    # Tags should be unchanged
    assert list(tensor.itags) == original_itags


def test_flip_out_of_range_raises():
    """Test that out of range index positions raise error."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    
    # Position too large
    with pytest.raises(IndexError, match="out of range"):
        tensor.flip(2)
    
    # Negative position
    with pytest.raises(IndexError, match="out of range"):
        tensor.flip(-1)
    
    # In list
    with pytest.raises(IndexError, match="out of range"):
        tensor.flip([0, 5])


def test_flip_z2_group():
    """Test flip with Z2 symmetry group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_charges = tensor.indices[0].charges()
    
    tensor.flip(0)
    
    # For Z2, dual(0) = 0 and dual(1) = 1, so charges appear unchanged
    # but they were conjugated (Z2 charges are self-dual)
    assert tensor.indices[0].charges() == original_charges
    # Direction should be reversed
    assert tensor.indices[0].direction == Direction.IN


def test_flip_product_group():
    """Test flip with product group."""
    group = ProductGroup([U1Group(), U1Group()])
    # Use matching charges for both indices to ensure charge conservation
    idx1 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2), Sector((1, -1), 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector((0, 0), 2), Sector((1, -1), 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_charges = tensor.indices[1].charges()
    original_direction = tensor.indices[1].direction
    
    tensor.flip(1)
    
    # Charges should be conjugated: dual((0,0)) = (0,0), dual((1,-1)) = (-1,1)
    expected_charges = tuple(group.dual(c) for c in original_charges)
    assert tensor.indices[1].charges() == expected_charges
    # Direction should be reversed
    assert tensor.indices[1].direction == original_direction.reverse()


def test_flip_double_application():
    """Test that flipping twice returns to original direction, charges, and data."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    original_direction = tensor.indices[0].direction
    original_charges = tensor.indices[0].charges()
    original_norm = tensor.norm()
    original_keys = set(tensor.data.keys())
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    # Flip twice
    tensor.flip(0)
    tensor.flip(0)
    
    # Should be back to original
    assert tensor.indices[0].direction == original_direction
    assert tensor.indices[0].charges() == original_charges
    # Verify norm preserved
    assert np.isclose(tensor.norm(), original_norm)
    # Verify keys unchanged
    assert set(tensor.data.keys()) == original_keys
    # Verify data values unchanged
    for key in original_data:
        np.testing.assert_allclose(tensor.data[key], original_data[key])


# insert_index tests

def test_insert_index_at_beginning():
    """Test inserting a trivial index at the beginning."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_norm = tensor.norm()
    original_block_00 = tensor.data[(0, 0)].copy()
    
    # Insert at position 0
    tensor.insert_index(0, Direction.OUT, itag="new")
    
    # Verify structure
    assert len(tensor.indices) == 3
    assert len(tensor.itags) == 3
    assert tensor.itags[0] == "new"
    assert tensor.itags[1] == "a"
    assert tensor.itags[2] == "b"
    
    # Verify new index is trivial
    assert len(tensor.indices[0].sectors) == 1
    assert tensor.indices[0].sectors[0].charge == 0
    assert tensor.indices[0].sectors[0].dim == 1
    assert tensor.indices[0].direction == Direction.OUT
    
    # Verify block keys updated
    assert (0, 0, 0) in tensor.data
    assert (0, 1, 1) in tensor.data
    
    # Verify block shapes updated (added dimension at axis 0)
    assert tensor.data[(0, 0, 0)].shape == (1, 2, 2)
    assert tensor.data[(0, 1, 1)].shape == (1, 3, 3)
    
    # Verify data preserved (just reshaped)
    assert np.allclose(tensor.data[(0, 0, 0)][0], original_block_00)
    assert tensor.norm() == original_norm


def test_insert_index_at_end():
    """Test inserting a trivial index at the end."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_norm = tensor.norm()
    
    # Insert at end (position 2)
    tensor.insert_index(2, Direction.IN, itag="new")
    
    # Verify structure
    assert len(tensor.indices) == 3
    assert tensor.itags[2] == "new"
    
    # Verify block keys updated
    assert (0, 0, 0) in tensor.data
    assert (1, 1, 0) in tensor.data
    
    # Verify block shapes updated (added dimension at axis 2)
    assert tensor.data[(0, 0, 0)].shape == (2, 2, 1)
    assert tensor.data[(1, 1, 0)].shape == (3, 3, 1)
    
    # Verify norm preserved
    assert np.isclose(tensor.norm(), original_norm)


def test_insert_index_in_middle():
    """Test inserting a trivial index in the middle."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    
    tensor = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    
    # Insert at position 1
    tensor.insert_index(1, Direction.IN, itag="mid")
    
    # Verify structure
    assert len(tensor.indices) == 4
    assert list(tensor.itags) == ["a", "mid", "b", "c"]
    
    # Verify block keys updated (neutral charge inserted at position 1)
    for key in tensor.data:
        assert len(key) == 4
        assert key[1] == 0  # Neutral charge at position 1
    
    # Verify dimensions
    for arr in tensor.data.values():
        assert arr.ndim == 4
        assert arr.shape[1] == 1  # Singleton dimension at axis 1


def test_insert_index_default_itag():
    """Test inserting index without specifying itag."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    tensor.insert_index(1, Direction.OUT)
    
    # Should use default "_init_" tag
    assert tensor.itags[1] == "_init_"


def test_insert_index_inherits_group():
    """Test that inserted index inherits group from existing indices."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    
    # Insert index - should inherit Z2 group
    tensor.insert_index(1, Direction.OUT, itag="z2_trivial")
    
    # Verify the new index has Z2 group (inherited)
    assert tensor.indices[1].group == group
    assert tensor.indices[1].sectors[0].charge == 0  # Z2 neutral is also 0


def test_insert_index_preserves_data_values():
    """Test that insertion preserves all data values."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    
    # Store all original values
    original_values = {}
    for key, arr in tensor.data.items():
        original_values[key] = arr.copy()
    
    # Insert index
    tensor.insert_index(1, Direction.OUT, itag="inserted")
    
    # Verify all values preserved (just reshaped)
    for old_key, old_arr in original_values.items():
        new_key = (old_key[0], 0, old_key[1])  # Insert neutral charge
        assert new_key in tensor.data
        assert np.allclose(tensor.data[new_key][:, 0, :], old_arr)


def test_insert_index_multiple_insertions():
    """Test multiple consecutive insertions."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_norm = tensor.norm()
    
    # Insert at beginning
    tensor.insert_index(0, Direction.OUT, itag="first")
    assert len(tensor.indices) == 3
    
    # Insert at end
    tensor.insert_index(3, Direction.IN, itag="last")
    assert len(tensor.indices) == 4
    
    # Insert in middle
    tensor.insert_index(2, Direction.OUT, itag="middle")
    assert len(tensor.indices) == 5
    
    # Verify structure
    assert list(tensor.itags) == ["first", "a", "middle", "b", "last"]
    
    # Verify all new indices are trivial
    assert tensor.indices[0].sectors[0].dim == 1
    assert tensor.indices[2].sectors[0].dim == 1
    assert tensor.indices[4].sectors[0].dim == 1
    
    # Verify norm preserved
    assert np.isclose(tensor.norm(), original_norm)


def test_insert_index_position_validation():
    """Test that invalid positions raise errors."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    
    # Position too large
    with pytest.raises(ValueError, match="out of range"):
        tensor.insert_index(3, Direction.OUT)
    
    # Negative position
    with pytest.raises(ValueError, match="out of range"):
        tensor.insert_index(-1, Direction.OUT)


def test_insert_index_scalar_raises_error():
    """Test that inserting into scalar tensor raises error."""
    tensor = Tensor.from_scalar(1.0)
    
    with pytest.raises(ValueError, match="Cannot insert index into scalar tensor"):
        tensor.insert_index(0, Direction.OUT)


def test_insert_index_z2_group():
    """Test inserting trivial index with Z2 group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    tensor.insert_index(1, Direction.OUT, itag="z2_trivial")
    
    # Z2 neutral is 0
    assert tensor.indices[1].sectors[0].charge == 0
    assert tensor.indices[1].sectors[0].dim == 1
    
    # Verify keys
    assert (0, 0, 0) in tensor.data
    assert (1, 0, 1) in tensor.data


def test_insert_index_product_group():
    """Test inserting trivial index with product group."""
    group = ProductGroup([U1Group(), U1Group()])
    idx1 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2), Sector((1, -1), 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector((0, 0), 2), Sector((1, -1), 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    tensor.insert_index(1, Direction.OUT, itag="prod_trivial")
    
    # Product group neutral is (0, 0)
    assert tensor.indices[1].sectors[0].charge == (0, 0)
    assert tensor.indices[1].sectors[0].dim == 1


def test_insert_index_complex_dtype():
    """Test inserting index preserves complex dtype."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx1, idx2], seed=42, dtype=np.complex128, itags=["a", "b"])
    
    tensor.insert_index(1, Direction.OUT)
    
    # Verify dtype preserved
    assert tensor.dtype == np.complex128
    for arr in tensor.data.values():
        assert arr.dtype == np.complex128


def test_insert_index_inplace_modification():
    """Test that insert_index modifies tensor in-place."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    original_id = id(tensor)
    
    result = tensor.insert_index(1, Direction.OUT)
    
    # Should return None (in-place operation)
    assert result is None
    
    # Tensor object should be the same
    assert id(tensor) == original_id
    
    # But structure should be modified
    assert len(tensor.indices) == 3


def test_insert_index_preserves_label():
    """Test that insert_index preserves tensor label."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    tensor.label = "MyTensor"
    
    tensor.insert_index(1, Direction.OUT)
    
    assert tensor.label == "MyTensor"


def test_insert_index_invalidates_sorted_keys():
    """Test that insert_index invalidates the sorted keys cache."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    
    # Access sorted_keys to populate cache
    _ = tensor.sorted_keys
    
    # Insert index
    tensor.insert_index(1, Direction.OUT)
    
    # Sorted keys should be recalculated with new structure
    keys = tensor.sorted_keys
    for key in keys:
        assert len(key) == 3  # Now 3D


# Merge axes tests

def test_merge_axes_basic():
    """Test basic merge_axes functionality with 3 axes."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(2, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 1), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 3), Sector(1, 1)))
    idx4 = Index(Direction.IN, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    
    T = Tensor.random([idx1, idx2, idx3, idx4], seed=42, itags=['a', 'b', 'c', 'd'])
    
    merged, iso_conj = merge_axes(T, ['a', 'b', 'c'], merged_tag='abc')
    
    # Check merged tensor structure
    assert len(merged.indices) == 2
    assert 'abc' in merged.itags
    assert 'd' in merged.itags
    
    # Check isometry conjugate structure
    assert len(iso_conj.indices) == 4  # 3 unfused + 1 fused


def test_merge_axes_with_int_positions():
    """Test merge_axes using integer axis positions."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 1), Sector(2, 1)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 3), Sector(1, 1)))
    idx4 = Index(Direction.IN, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    
    # Use 4 indices to avoid 1-index tensor after merging
    T = Tensor.random([idx1, idx2, idx3, idx4], seed=1, itags=['a', 'b', 'c', 'd'])
    
    merged, iso_conj = merge_axes(T, [0, 1, 2], merged_tag='merged')
    
    assert len(merged.indices) == 2
    assert 'merged' in merged.itags
    assert 'd' in merged.itags


def test_merge_axes_unfuse_with_conjugate():
    """Test that contracting with conjugate isometry unfuses the axis."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 1), Sector(2, 1)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(-2, 1), Sector(0, 2), Sector(1, 1)))
    
    T = Tensor.random([idx1, idx2, idx3], seed=123, itags=['a', 'b', 'c'])
    
    # Merge first two axes
    merged, iso_conj = merge_axes(T, [0, 1], merged_tag='ab')
    
    # Unfuse by contracting with conjugate
    unmerged = contract(merged, iso_conj)
    
    # Should have 3 indices again
    assert len(unmerged.indices) == 3
    # Tags might be in different order
    assert set(unmerged.itags) == {'a', 'b', 'c'}
    
    # Verify data is identical to original
    # Need to permute unmerged to match original order
    tag_to_pos_original = {tag: i for i, tag in enumerate(T.itags)}
    tag_to_pos_unmerged = {tag: i for i, tag in enumerate(unmerged.itags)}
    perm = [tag_to_pos_unmerged[tag] for tag in T.itags]
    unmerged.permute(perm)
    
    # Now compare block by block
    assert set(T.data.keys()) == set(unmerged.data.keys())
    for key in T.data.keys():
        np.testing.assert_allclose(T.data[key], unmerged.data[key], rtol=1e-10, atol=1e-12)


def test_merge_axes_direction_parameter():
    """Test merge_axes with custom direction parameter."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 1), Sector(1, 2)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    
    T = Tensor.random([idx1, idx2, idx3], seed=1, itags=['a', 'b', 'c'])
    
    # Merge with Direction.IN
    merged_in, _ = merge_axes(T, [0, 1], merged_tag='ab', direction=Direction.IN)
    
    # Find the merged index
    merged_idx = merged_in.indices[merged_in.itags.index('ab')]
    assert merged_idx.direction == Direction.IN
    
    # Merge with Direction.OUT (default)
    merged_out, _ = merge_axes(T, [0, 1], merged_tag='ab', direction=Direction.OUT)
    merged_idx_out = merged_out.indices[merged_out.itags.index('ab')]
    assert merged_idx_out.direction == Direction.OUT


def test_merge_axes_too_few_axes_raises():
    """Test that merge_axes raises error with fewer than 2 axes."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    T = Tensor.random([idx, idx.flip()], seed=1, itags=['a', 'b'])
    
    with pytest.raises(ValueError, match="at least 2 axes"):
        merge_axes(T, ['a'])
    
    with pytest.raises(ValueError, match="at least 2 axes"):
        merge_axes(T, [])


def test_merge_axes_invalid_tag_raises():
    """Test that merge_axes raises error for invalid tag."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    T = Tensor.random([idx, idx.flip(), idx], seed=1, itags=['a', 'b', 'c'])
    
    with pytest.raises(ValueError, match="not found"):
        merge_axes(T, ['a', 'invalid_tag'])


def test_merge_axes_invalid_position_raises():
    """Test that merge_axes raises error for invalid position."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    T = Tensor.random([idx, idx.flip()], seed=1, itags=['a', 'b'])
    
    with pytest.raises(ValueError, match="out of range"):
        merge_axes(T, [0, 5])
    
    with pytest.raises(ValueError, match="out of range"):
        merge_axes(T, [-1, 0])


def test_merge_axes_mixed_int_str():
    """Test merge_axes with mixed int and str specifications."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    T = Tensor.random([idx, idx.flip(), idx, idx.flip()], seed=1, itags=['a', 'b', 'c', 'd'])
    
    merged, _ = merge_axes(T, [0, 'b', 2], merged_tag='abc')
    
    assert len(merged.indices) == 2
    assert 'abc' in merged.itags
    assert 'd' in merged.itags


def test_merge_axes_duplicate_axes():
    """Test that merge_axes raises error for duplicate axis specifications."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    T = Tensor.random([idx, idx.flip(), idx], seed=1, itags=['a', 'b', 'c'])
    
    # Specifying same axis twice should raise an error
    with pytest.raises(ValueError, match="ambiguities"):
        merge_axes(T, ['a', 'b', 'a'], merged_tag='ab')


def test_merge_axes_default_merged_tag():
    """Test merge_axes with default merged tag."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    T = Tensor.random([idx, idx.flip(), idx], seed=1, itags=['a', 'b', 'c'])
    
    merged, _ = merge_axes(T, [0, 1])
    
    assert '_merged_' in merged.itags


def test_merge_axes_z2_symmetry():
    """Test merge_axes with Z2 symmetry."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2, idx3], seed=42, itags=['a', 'b', 'c'])
    
    merged, iso_conj = merge_axes(T, ['a', 'b'], merged_tag='ab')
    
    assert len(merged.indices) == 2
    assert 'ab' in merged.itags


def test_merge_axes_product_group():
    """Test merge_axes with ProductGroup and validate unfusing."""
    u1 = U1Group()
    z2 = Z2Group()
    group = ProductGroup([u1, z2])
    
    idx1 = Index(Direction.OUT, group, sectors=(
        Sector((-2, 0), 1), Sector((-1, 1), 2), Sector((0, 0), 2), Sector((1, 1), 1)))
    idx2 = Index(Direction.OUT, group, sectors=(
        Sector((-1, 0), 1), Sector((0, 0), 1), Sector((1, 1), 2), Sector((2, 0), 1)))
    idx3 = Index(Direction.IN, group, sectors=(
        Sector((-1, 0), 1), Sector((0, 0), 2), Sector((1, 1), 1)))
    
    T = Tensor.random([idx1, idx2, idx3], seed=42, itags=['a', 'b', 'c'])
    
    merged, iso_conj = merge_axes(T, [0, 1], merged_tag='ab')
    
    assert len(merged.indices) == 2
    assert 'ab' in merged.itags
    
    # Validate unfusing restores original
    unmerged = contract(merged, iso_conj)
    
    assert len(unmerged.indices) == 3
    assert set(unmerged.itags) == {'a', 'b', 'c'}
    
    # Permute to match original order
    tag_to_pos_original = {tag: i for i, tag in enumerate(T.itags)}
    tag_to_pos_unmerged = {tag: i for i, tag in enumerate(unmerged.itags)}
    perm = [tag_to_pos_unmerged[tag] for tag in T.itags]
    unmerged.permute(perm)
    
    # Verify data blocks match
    assert set(T.data.keys()) == set(unmerged.data.keys())
    for key in T.data.keys():
        assert np.allclose(T.data[key], unmerged.data[key])


def test_merge_axes_preserves_dtype():
    """Test that merge_axes preserves tensor dtype."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 2), Sector(1, 1)))
    
    # Test with complex dtype
    T = Tensor.random([idx, idx.flip(), idx], seed=1, dtype=np.complex128, itags=['a', 'b', 'c'])
    
    merged, iso_conj = merge_axes(T, [0, 1], merged_tag='ab')
    
    assert merged.dtype == np.complex128
    # Isometry uses the tensor's dtype
    assert iso_conj.dtype == np.complex128


# Trim zero sectors tests

def test_trim_zero_sectors_single_block():
    """Test removing a single near-zero block."""
    group = U1Group()
    idx_in = Index(
        direction=Direction.IN,
        group=group,
        sectors=(
            Sector(charge=-1, dim=2),
            Sector(charge=0, dim=2),
            Sector(charge=1, dim=2),
        )
    )
    idx_out = Index(
        direction=Direction.OUT,
        group=group,
        sectors=(
            Sector(charge=-1, dim=2),
            Sector(charge=0, dim=2),
            Sector(charge=1, dim=2),
        )
    )
    
    # Create data with one near-zero block
    data = {
        (-1, -1): np.array([[1.0, 0.5], [0.3, 0.8]]),
        (0, 0): np.array([[1e-20, 1e-20], [1e-20, 1e-20]]),  # Near-zero
        (1, 1): np.array([[0.7, 0.2], [0.4, 0.9]]),
    }
    
    tensor = Tensor(
        indices=(idx_in, idx_out),
        itags=("in", "out"),
        data=data,
        dtype=np.float64
    )
    
    # Apply trim
    tensor.trim_zero_sectors()
    
    # Verify near-zero block is removed
    assert len(tensor.data) == 2
    assert (0, 0) not in tensor.data
    assert (-1, -1) in tensor.data
    assert (1, 1) in tensor.data
    
    # Verify sectors are updated
    assert len(tensor.indices[0].sectors) == 2
    assert len(tensor.indices[1].sectors) == 2
    
    charges_in = [s.charge for s in tensor.indices[0].sectors]
    charges_out = [s.charge for s in tensor.indices[1].sectors]
    
    assert -1 in charges_in and 1 in charges_in
    assert 0 not in charges_in
    assert -1 in charges_out and 1 in charges_out
    assert 0 not in charges_out


def test_trim_zero_sectors_multiple_blocks():
    """Test removing multiple near-zero blocks."""
    group = U1Group()
    idx = Index(
        direction=Direction.IN,
        group=group,
        sectors=(
            Sector(charge=-1, dim=1),
            Sector(charge=0, dim=1),
            Sector(charge=1, dim=1),
            Sector(charge=2, dim=1),
        )
    )
    
    # Multiple near-zero blocks
    eps = np.finfo(np.float64).eps
    data = {
        (-1, -1): np.array([[1.0]]),
        (0, 0): np.array([[eps / 2]]),  # Below threshold
        (1, 1): np.array([[eps / 10]]),  # Below threshold
        (2, 2): np.array([[2.0]]),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=np.float64
    )
    
    tensor.trim_zero_sectors()
    
    # Only two blocks should remain
    assert len(tensor.data) == 2
    assert (-1, -1) in tensor.data
    assert (2, 2) in tensor.data
    assert (0, 0) not in tensor.data
    assert (1, 1) not in tensor.data
    
    # Sectors should be trimmed
    charges = [s.charge for s in tensor.indices[0].sectors]
    assert set(charges) == {-1, 2}


def test_trim_zero_sectors_no_removal():
    """Test that trim does nothing when all blocks are non-zero."""
    group = U1Group()
    idx = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=0, dim=2), Sector(charge=1, dim=2))
    )
    
    data = {
        (0, 0): np.array([[1.0, 0.5], [0.3, 0.8]]),
        (1, 1): np.array([[0.7, 0.2], [0.4, 0.9]]),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=np.float64
    )
    
    # Store original state
    original_keys = set(tensor.data.keys())
    original_sectors = len(tensor.indices[0].sectors)
    
    tensor.trim_zero_sectors()
    
    # Nothing should change
    assert set(tensor.data.keys()) == original_keys
    assert len(tensor.indices[0].sectors) == original_sectors


def test_trim_zero_sectors_inplace():
    """Test that trim modifies the tensor in-place."""
    group = U1Group()
    idx = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=0, dim=1), Sector(charge=1, dim=1))
    )
    
    data = {
        (0, 0): np.array([[1e-20]]),
        (1, 1): np.array([[1.0]]),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=np.float64
    )
    
    # Get object id before
    tensor_id = id(tensor)
    
    # Apply trim (returns None for in-place)
    result = tensor.trim_zero_sectors()
    
    assert result is None  # In-place methods return None
    assert id(tensor) == tensor_id  # Same object
    assert len(tensor.data) == 1  # But modified


def test_trim_zero_sectors_negative_values():
    """Test trim with negative values in blocks."""
    group = U1Group()
    idx = Index(
        direction=Direction.IN,
        group=group,
        sectors=(
            Sector(charge=-1, dim=2),
            Sector(charge=0, dim=2),
            Sector(charge=1, dim=2),
        )
    )
    
    eps = np.finfo(np.float64).eps
    data = {
        (-1, -1): np.array([[-1.0, -0.5], [-0.3, -0.8]]),  # All negative, non-zero
        (0, 0): np.array([[-eps/2, -eps/3], [-eps/4, -eps/5]]),  # All negative, near-zero
        (1, 1): np.array([[0.7, 0.2], [0.4, 0.9]]),  # All positive, non-zero
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=np.float64
    )
    
    tensor.trim_zero_sectors()
    
    # Near-zero block should be removed despite negative values
    assert len(tensor.data) == 2
    assert (-1, -1) in tensor.data
    assert (1, 1) in tensor.data
    assert (0, 0) not in tensor.data
    
    # Verify negative values are preserved
    np.testing.assert_array_equal(
        tensor.data[(-1, -1)],
        np.array([[-1.0, -0.5], [-0.3, -0.8]])
    )


def test_trim_zero_sectors_mixed_signs():
    """Test trim with mixed positive and negative values in blocks."""
    group = U1Group()
    idx = Index(
        direction=Direction.IN,
        group=group,
        sectors=(
            Sector(charge=-1, dim=2),
            Sector(charge=0, dim=2),
            Sector(charge=1, dim=2),
        )
    )
    
    eps = np.finfo(np.float64).eps
    data = {
        # Mixed signs with large magnitude - should be kept
        (-1, -1): np.array([[1.5, -2.3], [-0.8, 1.2]]),
        # Mixed signs with tiny magnitude - should be removed
        (0, 0): np.array([[eps/2, -eps/3], [-eps/4, eps/5]]),
        # Mixed signs with one large value - should be kept
        (1, 1): np.array([[eps/2, -eps/3], [2.0, -eps/5]]),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=np.float64
    )
    
    tensor.trim_zero_sectors()
    
    # Block with all tiny values should be removed
    # Blocks with at least one large value should be kept
    assert len(tensor.data) == 2
    assert (-1, -1) in tensor.data
    assert (1, 1) in tensor.data
    assert (0, 0) not in tensor.data
    
    # Verify mixed-sign data is preserved exactly
    np.testing.assert_array_equal(
        tensor.data[(-1, -1)],
        np.array([[1.5, -2.3], [-0.8, 1.2]])
    )
    assert np.max(np.abs(tensor.data[(1, 1)])) >= 2.0  # Has the large value


def test_trim_zero_sectors_complex_values():
    """Test trim with complex-valued data."""
    group = U1Group()
    idx = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=0, dim=2), Sector(charge=1, dim=2))
    )
    
    # Complex near-zero block (both real and imaginary parts near zero)
    eps = np.finfo(np.float64).eps
    data = {
        (0, 0): np.array([[eps/2 + 1j*eps/3, eps/4 + 1j*eps/5],
                        [eps/6 + 1j*eps/7, eps/8 + 1j*eps/9]], dtype=np.complex128),
        (1, 1): np.array([[1.0 + 1.0j, 2.0 + 2.0j],
                        [3.0 + 3.0j, 4.0 + 4.0j]], dtype=np.complex128),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=np.complex128
    )
    
    tensor.trim_zero_sectors()
    
    # Near-zero complex block should be removed
    assert len(tensor.data) == 1
    assert (1, 1) in tensor.data
    assert (0, 0) not in tensor.data


def test_trim_zero_sectors_z2_symmetry():
    """Test trim with Z2 symmetry group."""
    group = Z2Group()
    idx = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=0, dim=2), Sector(charge=1, dim=2))
    )
    
    data = {
        (0, 0): np.array([[1e-20, 1e-20], [1e-20, 1e-20]]),
        (1, 1): np.array([[0.5, 0.3], [0.2, 0.8]]),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=np.float64
    )
    
    tensor.trim_zero_sectors()
    
    assert len(tensor.data) == 1
    assert (1, 1) in tensor.data
    assert (0, 0) not in tensor.data
    
    # Z2 charge 1 should remain
    charges = [s.charge for s in tensor.indices[0].sectors]
    assert charges == [1]


def test_trim_zero_sectors_product_group():
    """Test trim with ProductGroup symmetry."""
    group = ProductGroup([U1Group(), U1Group()])
    idx = Index(
        direction=Direction.IN,
        group=group,
        sectors=(
            Sector(charge=(0, 0), dim=1),
            Sector(charge=(0, 1), dim=1),
            Sector(charge=(1, 0), dim=1),
        )
    )
    
    data = {
        ((0, 0), (0, 0)): np.array([[1.0]]),
        ((0, 1), (0, 1)): np.array([[1e-20]]),  # Near-zero
        ((1, 0), (1, 0)): np.array([[0.5]]),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=np.float64
    )
    
    tensor.trim_zero_sectors()
    
    assert len(tensor.data) == 2
    assert ((0, 1), (0, 1)) not in tensor.data
    
    charges = [s.charge for s in tensor.indices[0].sectors]
    assert (0, 0) in charges
    assert (1, 0) in charges
    assert (0, 1) not in charges

