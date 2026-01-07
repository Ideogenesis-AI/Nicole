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


"""Tests for tensor manipulation operations: copy, conj, permute, transpose, retag."""

import numpy as np
import pytest

from nicole import Direction, Tensor, U1Group, conj, permute, transpose
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


# Conjugation tests

def test_conj_functional_returns_new_instance():
    """Test that functional conj returns a new instance."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)
    tensor = Tensor.random([idx_a, idx_b], seed=123, dtype=np.complex128, itags=["A", "B"])

    tensor_conj = conj(tensor)
    
    assert tensor_conj is not tensor  # Verify it's a new instance


def test_conj_functional_conjugates_data():
    """Test that functional conj conjugates the data."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)
    tensor = Tensor.random([idx_a, idx_b], seed=123, dtype=np.complex128, itags=["A", "B"])

    tensor_conj = conj(tensor)
    
    for key in tensor.data:
        np.testing.assert_allclose(tensor_conj.data[key], np.conjugate(tensor.data[key]))


def test_conj_functional_flips_directions():
    """Test that functional conj flips index directions."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 2)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 1), (-1, 1)], group)
    tensor = Tensor.random([idx_a, idx_b], seed=123, dtype=np.complex128, itags=["A", "B"])

    tensor_conj = conj(tensor)
    
    for orig_idx, new_idx in zip(tensor.indices, tensor_conj.indices):
        assert new_idx.direction == orig_idx.direction.reverse()


def test_conj_inplace_modifies_original():
    """Test that in-place conj modifies the original tensor."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx], seed=1, dtype=np.complex128, itags=["A"])
    
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
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx], seed=1, dtype=np.float64, itags=["A"])
    
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    result = conj(tensor)
    
    # Data should be unchanged (just copied) for real dtype
    for key in original_data:
        np.testing.assert_allclose(result.data[key], original_data[key])


def test_conj_double_application():
    """Test that conjugating twice returns to original."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx], seed=1, dtype=np.complex128, itags=["A"])
    
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
        make_u1_index(Direction.OUT, [(0, 1), (1, 2)], group),
        make_u1_index(Direction.IN, [(0, 2)], group),
        make_u1_index(Direction.OUT, [(-1, 1), (0, 1)], group),
    ]
    tensor = Tensor.random(indices, seed=10, itags=["a", "b", "c"])
    
    permuted = permute(tensor, [2, 0, 1])
    
    assert permuted is not tensor


def test_permute_reorders_indices_and_blocks():
    """Test that permute correctly reorders indices and blocks."""
    group = U1Group()
    indices = [
        make_u1_index(Direction.OUT, [(0, 1), (1, 2)], group),
        make_u1_index(Direction.IN, [(0, 2)], group),
        make_u1_index(Direction.OUT, [(-1, 1), (0, 1)], group),
        make_u1_index(Direction.IN, [(0, 1)], group),
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
        make_u1_index(Direction.OUT, [(0, 1)], group),
        make_u1_index(Direction.IN, [(0, 2)], group),
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
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx, idx, idx], seed=1, itags=["a", "b", "c"])
    
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    permuted = permute(tensor, [0, 1, 2])
    
    assert list(permuted.itags) == ["a", "b", "c"]
    for key in original_data:
        np.testing.assert_allclose(permuted.data[key], original_data[key])


def test_permute_invalid_order():
    """Test that invalid permutation raises error."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    with pytest.raises(ValueError, match="Invalid permutation"):
        permute(tensor, [0, 0])
    
    with pytest.raises(ValueError, match="Invalid permutation"):
        permute(tensor, [0, 2])


# Transpose tests

def test_transpose_functional_returns_new_instance():
    """Test that functional transpose returns a new instance."""
    group = U1Group()
    indices = [make_u1_index(Direction.OUT, [(0, 1)], group) for _ in range(3)]
    tensor = Tensor.random(indices, seed=1, itags=["a", "b", "c"])
    
    transposed = transpose(tensor)
    
    assert transposed is not tensor


def test_transpose_default_reverses_order():
    """Test that transpose with no args reverses order."""
    group = U1Group()
    indices = [
        make_u1_index(Direction.OUT, [(0, 1), (2, 1)], group),
        make_u1_index(Direction.IN, [(0, 2)], group),
        make_u1_index(Direction.OUT, [(0, 1), (1, 1)], group),
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
    indices = [make_u1_index(Direction.OUT, [(0, 1)], group) for _ in range(3)]
    tensor = Tensor.random(indices, seed=1, itags=["a", "b", "c"])
    
    transposed = transpose(tensor, 1, 0, 2)
    
    assert list(transposed.itags) == ["b", "a", "c"]


def test_transpose_inplace():
    """Test in-place transpose method."""
    group = U1Group()
    indices = [make_u1_index(Direction.OUT, [(0, 1)], group) for _ in range(2)]
    tensor = Tensor.random(indices, seed=1, itags=["a", "b"])
    
    original_itags = list(tensor.itags)
    
    tensor.transpose()
    
    assert list(tensor.itags) == list(reversed(original_itags))


def test_transpose_double_application():
    """Test that transposing twice returns to original."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
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
        make_u1_index(Direction.OUT, [(0, 2)], group),
        make_u1_index(Direction.IN, [(0, 2)], group),
        make_u1_index(Direction.OUT, [(0, 1)], group),
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
        make_u1_index(Direction.OUT, [(0, 2)], group),
        make_u1_index(Direction.IN, [(0, 2)], group),
        make_u1_index(Direction.OUT, [(0, 1)], group),
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
        make_u1_index(Direction.OUT, [(0, 2)], group),
        make_u1_index(Direction.IN, [(0, 2)], group),
        make_u1_index(Direction.OUT, [(0, 1)], group),
    ]
    itags = ["x", "y", "z"]
    
    tensor = Tensor.random(indices, seed=14, itags=itags)
    original_data = {k: v.copy() for k, v in tensor.data.items()}
    
    result = tensor.retag([0, 2], ["first", "third"])
    
    assert result is None
    assert list(tensor.itags) == ["first", "y", "third"]
    for key in original_data:
        np.testing.assert_allclose(tensor.data[key], original_data[key])


def test_retag_mapping_unmapped_tags_preserved():
    """Test that unmapped tags are preserved in mode 1."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx, idx, idx], seed=1, itags=["a", "b", "c"])
    
    tensor.retag({"a": "alpha"})
    
    assert list(tensor.itags) == ["alpha", "b", "c"]


def test_retag_mode2_wrong_count_raises():
    """Test that mode 2 with wrong count raises error."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    with pytest.raises(ValueError, match="must match number of indices"):
        tensor.retag(["x", "y", "z"])


def test_retag_mode3_wrong_count_raises():
    """Test that mode 3 with mismatched counts raises error."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    with pytest.raises(ValueError, match="must match number of new tags"):
        tensor.retag([0], ["x", "y"])


def test_retag_mode3_out_of_range_raises():
    """Test that mode 3 with out of range index raises error."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    with pytest.raises(IndexError, match="out of range"):
        tensor.retag([5], ["x"])


def test_retag_preserves_tensor_data():
    """Test that retag never modifies tensor data or structure."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2), (1, 3)], group)
    tensor = Tensor.random([idx], seed=1, itags=["original"])
    
    original_norm = tensor.norm()
    original_keys = set(tensor.data.keys())
    
    tensor.retag(["new_name"])
    
    assert tensor.norm() == original_norm
    assert set(tensor.data.keys()) == original_keys

