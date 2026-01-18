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


"""Tests for tensor manipulation operations: conj, permute, transpose, retag."""

import numpy as np
import pytest

from nicole import Direction, Index, Sector, Tensor, conj, permute, transpose
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

