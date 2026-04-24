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


"""Tests for tensor helper operations: clone, keys & blocks, regularize, serialize."""

import math

import torch
import pytest

from nicole.symmetry import delegate as dg
from nicole import Tensor, U1Group, Z2Group, SU2Group, ProductGroup
from nicole import Direction, Index, Sector
from nicole import filter_blocks
from nicole.serialize import serialize, deserialize
from ..utils import populate_random_weights


# Clone tests

def test_clone_returns_new_instance():
    """Test that clone returns a new tensor instance."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=123, itags=["A", "B"])

    cloned = tensor.clone()

    assert cloned is not tensor


def test_clone_has_identical_data():
    """Test that clone has identical data values."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=456, dtype=torch.complex128, itags=["A", "B"])

    cloned = tensor.clone()

    assert set(cloned.data.keys()) == set(tensor.data.keys())
    for key in tensor.data:
        assert torch.equal(cloned.data[key], tensor.data[key])


def test_clone_creates_independent_data():
    """Test that modifying a clone doesn't affect the original."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=789, itags=["X", "Y"])

    original_data = {k: v.clone() for k, v in tensor.data.items()}
    cloned = tensor.clone()

    # Modify the clone's data
    for key in cloned.data:
        cloned.data[key] *= 100.0

    # Original should be unchanged
    for key in original_data:
        assert torch.equal(tensor.data[key], original_data[key])


def test_clone_preserves_metadata():
    """Test that clone preserves indices, itags, dtype, and label."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 2)))
    tensor = Tensor.random([idx_a, idx_b], seed=111, dtype=torch.complex128, itags=["left", "right"])
    tensor.label = "MyTensor"

    cloned = tensor.clone()

    assert cloned.indices == tensor.indices
    assert cloned.itags == tensor.itags
    assert cloned.dtype == tensor.dtype
    assert cloned.label == tensor.label


def test_clone_shares_immutable_indices():
    """Test that clone shares the same Index objects (since they're immutable)."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=222, itags=["A", "B"])

    cloned = tensor.clone()

    # Index objects should be the same (shared) since they're immutable
    for orig_idx, clone_idx in zip(tensor.indices, cloned.indices):
        assert orig_idx is clone_idx


def test_clone_data_arrays_are_independent():
    """Test that torch tensors in a clone are different objects."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=333, itags=["A", "B"])

    cloned = tensor.clone()

    # Each array should be a different object
    for key in tensor.data:
        assert cloned.data[key] is not tensor.data[key]


def test_clone_su2_clones_intw():
    """Test that clone() deep copies the intertwiner for SU2."""
    group = SU2Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    tensor = Tensor.zeros([idx1, idx2], dtype=torch.float64)
    cloned = tensor.clone()
    
    # Verify intw exists and is deep copied
    assert cloned.intw is not None
    assert cloned.intw is not tensor.intw  # Different dict
    
    # Verify each Bridge is cloned
    for key in tensor.intw.keys():
        assert key in cloned.intw
        # Same cgspec (immutable)
        assert cloned.intw[key].cgspec == tensor.intw[key].cgspec
        # Cloned weights (different tensor objects)
        assert cloned.intw[key].weights is not tensor.intw[key].weights
        assert torch.allclose(cloned.intw[key].weights, tensor.intw[key].weights)


def test_clone_abelian_no_intw():
    """Test that clone() preserves intw=None for Abelian tensors."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.zeros([idx1, idx2])
    cloned = tensor.clone()
    
    assert tensor.intw is None
    assert cloned.intw is None


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
        assert torch.equal(tensor.block(i), tensor.data[key])


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
    tensor.permute([1, 0], in_place=True)
    
    # Cache should be invalidated
    assert tensor._sorted_keys is None
    
    # Access new keys (this repopulates the cache)
    new_keys = tensor.sorted_keys
    
    # New keys should be valid for the permuted data
    assert set(new_keys) == set(tensor.data.keys())
    # Cache should now be repopulated
    assert tensor._sorted_keys is not None


def test_display_numbering_matches_block_index():
    """Test that block numbers in the display string match block() / key() indexing."""
    import re
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1), Sector(-1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1), Sector(1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=108, itags=["A", "B"])

    display_str = str(tensor)

    # Each block line starts with a right-aligned number followed by a period,
    # e.g. "     1.  1x1     |  1x1     [ 0 ;  0 ]  ...".
    # Extract the leading block numbers from the display.
    displayed_numbers = [
        int(m.group(1))
        for m in re.finditer(r"^\s+(\d+)\.", display_str, re.MULTILINE)
    ]

    # The display must list every block (no truncation for 3 blocks)
    assert displayed_numbers == list(range(1, len(tensor.data) + 1))

    # Each displayed number i must correspond to the correct key in sorted order
    for i, key in zip(displayed_numbers, tensor.sorted_keys):
        assert tensor.key(i) == key


# filter_blocks tests

def test_filter_blocks_returns_new_instance():
    """Test that filter_blocks returns a new tensor instance."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1), Sector(1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=200, itags=["A", "B"])

    sub = filter_blocks(tensor, [1, 2])

    assert sub is not tensor


def test_filter_blocks_contains_only_specified_blocks():
    """Test that filter_blocks returns only the specified blocks."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1), Sector(1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=201, itags=["A", "B"])

    indices_to_get = [1, 3]
    sub = filter_blocks(tensor, indices_to_get)

    # Should have exactly the specified number of blocks
    assert len(sub.data) == len(indices_to_get)
    
    # Should contain the correct keys
    expected_keys = {tensor.key(i) for i in indices_to_get}
    assert set(sub.data.keys()) == expected_keys


def test_filter_blocks_data_is_cloned():
    """Test that filter_blocks creates independent cloned data."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=202, itags=["A", "B"])

    sub = filter_blocks(tensor, 1)
    
    # Modify the sub tensor's data
    for key in sub.data:
        original_value = tensor.data[key].clone()
        sub.data[key] *= 100.0
        # Original should be unchanged
        assert torch.equal(tensor.data[key], original_value)


def test_filter_blocks_preserves_metadata():
    """Test that filter_blocks preserves itags, dtype, and label."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=203, dtype=torch.complex128, itags=["left", "right"])
    tensor.label = "TestTensor"

    sub = filter_blocks(tensor, 1)

    assert sub.itags == tensor.itags
    assert sub.dtype == tensor.dtype
    assert sub.label == tensor.label


def test_filter_blocks_prunes_unused_sectors():
    """Test that filter_blocks removes sectors not present in selected blocks."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1), Sector(1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=203, itags=["A", "B"])

    # Get only first block
    sub = filter_blocks(tensor, 1)

    # filter_blocks should only have sectors that appear in block 1
    key_1 = tensor.key(1)
    
    # Check that filter_blocks indices only contain used charges
    for axis, charge in enumerate(key_1):
        used_charges = [s.charge for s in sub.indices[axis].sectors]
        assert charge in used_charges
        # Should have fewer or equal sectors than original
        assert len(sub.indices[axis].sectors) <= len(tensor.indices[axis].sectors)


def test_filter_blocks_single_block():
    """Test filter_blocks with a single block index."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=204, itags=["A", "B"])

    sub = filter_blocks(tensor, 1)

    assert len(sub.data) == 1
    key = tensor.key(1)
    assert torch.equal(sub.data[key], tensor.data[key])


def test_filter_blocks_integer_vs_list_syntax():
    """Test that single integer and list syntax produce identical results."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1), Sector(1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=207, itags=["A", "B"])

    # Get block using single integer syntax
    sub_int = filter_blocks(tensor, 2)
    
    # Get same block using list syntax
    sub_list = filter_blocks(tensor, [2])

    # Both should have the same number of blocks
    assert len(sub_int.data) == len(sub_list.data) == 1
    
    # Both should have the same keys
    assert set(sub_int.data.keys()) == set(sub_list.data.keys())
    
    # Both should have the same data
    for key in sub_int.data:
        assert torch.equal(sub_int.data[key], sub_list.data[key])
    
    # Both should preserve the same metadata
    assert sub_int.indices == sub_list.indices
    assert sub_int.itags == sub_list.itags
    assert sub_int.dtype == sub_list.dtype


def test_filter_blocks_all_blocks():
    """Test filter_blocks with all block indices."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=205, itags=["A", "B"])

    all_indices = list(range(1, len(tensor.data) + 1))
    sub = filter_blocks(tensor, all_indices)

    assert len(sub.data) == len(tensor.data)
    for key in tensor.data:
        assert torch.equal(sub.data[key], tensor.data[key])


def test_filter_blocks_raises_on_invalid_index():
    """Test that filter_blocks raises IndexError for invalid indices."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=206, itags=["A", "B"])

    num_blocks = len(tensor.data)

    # Test with sequence syntax
    with pytest.raises(IndexError):
        filter_blocks(tensor, [0])  # 0 is invalid (1-indexed)

    with pytest.raises(IndexError):
        filter_blocks(tensor, [num_blocks + 1])  # Out of range

    with pytest.raises(IndexError):
        filter_blocks(tensor, [1, num_blocks + 1])  # One valid, one invalid

    # Test with single integer syntax
    with pytest.raises(IndexError):
        filter_blocks(tensor, 0)  # 0 is invalid (1-indexed)

    with pytest.raises(IndexError):
        filter_blocks(tensor, num_blocks + 1)  # Out of range


def test_filter_blocks_su2_single_block():
    """Test filter_blocks with SU(2) single block extraction."""
    from nicole.identity import identity
    
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 3), Sector(2, 2)
    ))
    
    tensor = identity(idx)
    assert tensor.intw is not None
    
    # Extract block 2
    sub = filter_blocks(tensor, 2)
    
    # Should have only 1 block
    assert len(sub.data) == 1
    assert len(sub.intw) == 1
    
    # Block and intw should match original
    key = tensor.key(2)
    assert key in sub.data
    assert key in sub.intw
    assert torch.equal(sub.data[key], tensor.data[key])
    assert torch.equal(sub.intw[key].weights, tensor.intw[key].weights)


def test_filter_blocks_su2_multiple_blocks():
    """Test filter_blocks with SU(2) multiple block extraction."""
    group = SU2Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2))),
        Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3))),
        Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 2))),
    ]
    tensor = Tensor.random(indices, seed=42, itags=["a", "b", "c"])
    
    assert tensor.intw is not None
    num_blocks = len(tensor.data)
    
    # Extract several blocks
    block_indices = [1, 3, min(5, num_blocks)]
    sub = filter_blocks(tensor, block_indices)
    
    # Should have specified number of blocks
    assert len(sub.data) == len(block_indices)
    assert len(sub.intw) == len(block_indices)
    
    # Each block and intw should match original
    for i in block_indices:
        key = tensor.key(i)
        assert key in sub.data
        assert key in sub.intw
        assert torch.equal(sub.data[key], tensor.data[key])
        assert torch.equal(sub.intw[key].weights, tensor.intw[key].weights)


def test_filter_blocks_su2_preserves_cgspec():
    """Test that filter_blocks preserves CGSpec structure in intw."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    assert tensor.intw is not None
    
    # Extract block 1
    sub = filter_blocks(tensor, 1)
    
    key = tensor.key(1)
    
    # CGSpec should be identical (same edges)
    orig_bridge = tensor.intw[key]
    sub_bridge = sub.intw[key]
    
    assert orig_bridge.cgspec.num_external() == sub_bridge.cgspec.num_external()
    assert orig_bridge.cgspec.om_dimension() == sub_bridge.cgspec.om_dimension()
    
    # Edges should match
    orig_edges = orig_bridge.cgspec.edges
    sub_edges = sub_bridge.cgspec.edges
    for orig_e, sub_e in zip(orig_edges, sub_edges):
        assert orig_e.j.twice() == sub_e.j.twice()
        assert orig_e.dir == sub_e.dir


def test_filter_blocks_su2_clones_weights():
    """Test that filter_blocks clones weights for independence."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=99, itags=["a", "b"])
    assert tensor.intw is not None
    
    # Extract block 1
    sub = filter_blocks(tensor, 1)
    
    key = tensor.key(1)
    
    # Weights should be cloned (different objects)
    orig_bridge = tensor.intw[key]
    sub_bridge = sub.intw[key]
    
    assert sub_bridge.weights is not orig_bridge.weights
    
    # But values should match
    assert torch.equal(sub_bridge.weights, orig_bridge.weights)
    
    # Verify independence: modify sub weights
    original_value = orig_bridge.weights.clone()
    sub_bridge.weights[:] = 0.0
    
    # Original should be unchanged
    assert torch.equal(orig_bridge.weights, original_value)


# regularize tests

def test_regularize_no_op_for_abelian():
    """Test that regularize leaves Abelian tensors (intw=None) untouched."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=300, itags=["A", "B"])

    original_data = {k: v.clone() for k, v in tensor.data.items()}
    tensor.regularize()

    assert tensor.intw is None
    for key, original in original_data.items():
        assert torch.equal(tensor.data[key], original)


def test_regularize_2nd_order_sets_weight_to_sqrt_irrep_dim():
    """Test that Bridge weights equal sqrt(irrep_dim(q)) after regularize."""
    group = SU2Group()
    # Tensor.random initialises Bridge weights to 1.0 via from_block
    idx = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    tensor = Tensor.random([idx, idx.flip()], seed=301, itags=["a", "b"])

    tensor.regularize()

    for key, bridge in tensor.intw.items():
        q = key[0]
        expected = math.sqrt(group.irrep_dim(q))
        assert torch.isclose(
            bridge.weights[0, 0],
            torch.tensor(expected, dtype=tensor.dtype),
        ), f"key={key}: expected weight {expected}, got {bridge.weights[0, 0].item()}"


def test_regularize_2nd_order_preserves_physical_content():
    """Test that regularize preserves R * W for every block."""
    group = SU2Group()
    idx = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    tensor = Tensor.random([idx, idx.flip()], seed=302, itags=["a", "b"])

    # Record physical content (R * W) before regularization
    physical_before = {
        key: tensor.data[key] * tensor.intw[key].weights[0, 0]
        for key in tensor.data
    }

    tensor.regularize()

    for key in tensor.data:
        physical_after = tensor.data[key] * tensor.intw[key].weights[0, 0]
        assert torch.allclose(physical_after, physical_before[key]), (
            f"Physical content changed for block {key}"
        )


def test_regularize_2nd_order_identity_already_canonical():
    """Test that regularize leaves an identity tensor's data unchanged."""
    from nicole.identity import identity
    group = SU2Group()
    idx = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2)))
    tensor = identity(idx)

    # Record R and W before regularization
    data_before = {k: v.clone() for k, v in tensor.data.items()}
    weights_before = {k: b.weights.clone() for k, b in tensor.intw.items()}

    tensor.regularize()

    for key in tensor.data:
        assert torch.allclose(tensor.data[key], data_before[key]), (
            f"R changed for canonical block {key}"
        )
        assert torch.allclose(tensor.intw[key].weights, weights_before[key]), (
            f"W changed for canonical block {key}"
        )


def test_regularize_2nd_order_idempotent():
    """Test that calling regularize twice gives the same result as calling it once."""
    group = SU2Group()
    idx = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    tensor = Tensor.random([idx, idx.flip()], seed=303, itags=["a", "b"])

    tensor.regularize()

    data_after_first = {k: v.clone() for k, v in tensor.data.items()}
    weights_after_first = {k: b.weights.clone() for k, b in tensor.intw.items()}

    tensor.regularize()

    for key in tensor.data:
        assert torch.allclose(tensor.data[key], data_after_first[key]), (
            f"Second regularize changed R for block {key}"
        )
        assert torch.allclose(tensor.intw[key].weights, weights_after_first[key]), (
            f"Second regularize changed W for block {key}"
        )


def test_regularize_higher_order_large_weight_unchanged():
    """Test that higher-order blocks with weight > eps are left unchanged."""
    group = SU2Group()
    idx1 = Index(Direction.IN,  group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 3)))
    tensor = Tensor.random([idx1, idx2, idx3], seed=304, itags=["a", "b", "c"])
    assert tensor.intw is not None

    # from_block initialises weights[0,0]=1.0, which is well above float32 eps
    data_before   = {k: v.clone() for k, v in tensor.data.items()}
    weights_before = {k: b.weights.clone() for k, b in tensor.intw.items()}

    tensor.regularize()

    for key in tensor.data:
        assert torch.allclose(tensor.data[key], data_before[key]), (
            f"R unexpectedly changed for higher-order block {key}"
        )
        assert torch.allclose(tensor.intw[key].weights, weights_before[key]), (
            f"W unexpectedly changed for higher-order block {key}"
        )


def test_regularize_higher_order_small_weight_normalized():
    """Test that a small (< float32 eps) single-scalar weight is normalized to 1."""
    group = SU2Group()
    idx1 = Index(Direction.IN,  group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 3)))
    tensor = Tensor.random([idx1, idx2, idx3], seed=305, itags=["a", "b", "c"])
    assert tensor.intw is not None

    # Pick a block whose Bridge weight is a single scalar and force it small
    target_key = None
    for key, bridge in tensor.intw.items():
        if bridge.weights.numel() == 1:
            target_key = key
            break
    if target_key is None:
        pytest.skip("No single-scalar Bridge found in this tensor")

    small_w = torch.tensor(1e-10, dtype=tensor.dtype)
    tensor.intw[target_key].weights[0, 0] = small_w
    r_before = tensor.data[target_key].clone()

    tensor.regularize()

    # Weight must now be 1
    assert torch.isclose(
        tensor.intw[target_key].weights[0, 0],
        torch.ones(1, dtype=tensor.dtype),
    )
    # R must carry the absorbed factor
    assert torch.allclose(tensor.data[target_key], r_before * small_w)


# Compression-aspect tests for regularize

def test_regularize_compresses_dependent_components():
    """regularize must remove linearly dependent components in higher-order tensors."""
    group = SU2Group()
    # Four spin-1/2 indices give om_dim = 2 for the (1,1,1,1) block.
    idx1 = Index(Direction.IN,  group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.IN,  group, sectors=(Sector(1, 2),))
    idx4 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))

    tensor = Tensor.random([idx1, idx2, idx3, idx4], seed=77, itags=["a", "b", "c", "d"])
    assert tensor.intw is not None

    # Inject a redundant extra component (multiple of the existing row) into each block.
    for key, bridge in list(tensor.intw.items()):
        W = bridge.weights                          # (k, om_dim)
        dep = W[0:1, :] * 2.5                      # rank-1 addition — linearly dependent
        new_weights = torch.cat([W, dep], dim=0)   # (k+1, om_dim)
        tensor.intw[key] = dg.Bridge(cgspec=bridge.cgspec, weights=new_weights)
        tensor.data[key] = torch.cat(
            [tensor.data[key], tensor.data[key][..., 0:1] * 2.5], dim=-1
        )

    original_norm = tensor.norm()
    # Every block now has an extra dependent component.
    max_before = max(b.num_components for b in tensor.intw.values())

    tensor.regularize()

    # At least one block should have been compressed.
    max_after = max(b.num_components for b in tensor.intw.values())
    assert max_after < max_before, "Expected at least one block to be compressed"
    # Physical content must be preserved.
    assert math.isclose(tensor.norm(), original_norm, rel_tol=1e-8)


def test_regularize_does_not_compress_independent_components():
    """regularize must retain all components when they are linearly independent."""
    group = SU2Group()
    idx1 = Index(Direction.IN,  group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.IN,  group, sectors=(Sector(1, 2),))
    idx4 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))

    tensor = Tensor.random([idx1, idx2, idx3, idx4], seed=88, itags=["a", "b", "c", "d"])
    assert tensor.intw is not None

    # Replace weights with two orthogonal rows — guaranteed independent.
    key = next(iter(tensor.intw))
    bridge = tensor.intw[key]
    ind_weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    tensor.intw[key] = dg.Bridge(cgspec=bridge.cgspec, weights=ind_weights)
    tensor.data[key] = torch.randn(*tensor.data[key].shape[:-1], 2, dtype=torch.float64)

    n_before = tensor.intw[key].num_components

    tensor.regularize()

    assert tensor.intw[key].num_components == n_before, (
        "regularize should not compress linearly independent components"
    )


def test_regularize_2nd_order_compress_is_noop():
    """regularize on a 2nd-order SU(2) tensor must not change num_components (om=1 by Schur)."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(1, 2), Sector(2, 3)))

    tensor = Tensor.random([idx1, idx2], seed=11, itags=["a", "b"])
    assert tensor.intw is not None

    components_before = {k: b.num_components for k, b in tensor.intw.items()}
    tensor.regularize()
    components_after  = {k: b.num_components for k, b in tensor.intw.items()}

    assert components_before == components_after


def test_regularize_cutoff_forwarded():
    """A non-default cutoff passed to regularize must control what gets truncated.

    After regularize normalizes all weight rows to unit norm the singular values
    of the normalized weight matrix are O(1).  A cutoff larger than any of those
    singular values forces block_compress to retain only 1 component (the minimum
    guaranteed by the `max(1, ...)` guard); a cutoff of 1e-20 retains all
    components because every singular value exceeds it.
    """

    group = SU2Group()
    # Four spin-1/2 indices → om_dim = 2 for the single block.
    idx1 = Index(Direction.IN,  group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.IN,  group, sectors=(Sector(1, 2),))
    idx4 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))

    tensor = Tensor.random([idx1, idx2, idx3, idx4], seed=55, itags=["a", "b", "c", "d"])
    assert tensor.intw is not None

    key = next(iter(tensor.intw))
    bridge = tensor.intw[key]
    W = bridge.weights                      # (k, om_dim)

    # Give the block two genuinely independent rows so that both singular
    # values of the normalized weight matrix are O(1) — well above 1e-20 but
    # well below 2.0 (which exceeds sqrt(2), the maximum possible SV for a
    # 2×2 matrix of unit-norm rows).
    ind_weights = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    tensor.intw[key] = dg.Bridge(cgspec=bridge.cgspec, weights=ind_weights)
    tensor.data[key] = torch.randn(
        *tensor.data[key].shape[:-1], 2, dtype=torch.float64
    )
    n_before = 2  # two genuinely independent components

    # cutoff=2.0 — above all SVs of a unit-norm 2×2 matrix → compress forces 1 component.
    t_high = tensor.clone()
    t_high.regularize(cutoff=2.0)
    assert t_high.intw[key].num_components == 1

    # cutoff=1e-20 — below all SVs → both components retained.
    t_low = tensor.clone()
    t_low.regularize(cutoff=1e-20)
    assert t_low.intw[key].num_components == n_before


# Serialize / deserialize tests

def _assert_tensors_equal(a: Tensor, b: Tensor) -> None:
    """Assert that two tensors have identical metadata, block data, and intertwiners."""
    assert a.label == b.label
    assert a.dtype == b.dtype
    assert a.itags == b.itags
    assert len(a.indices) == len(b.indices)
    for ia, ib in zip(a.indices, b.indices):
        assert ia.direction == ib.direction
        assert ia.group == ib.group
        assert ia.sectors == ib.sectors
    assert set(a.data.keys()) == set(b.data.keys())
    for key in a.data:
        assert torch.equal(a.data[key], b.data[key])
    assert (a.intw is None) == (b.intw is None)
    if a.intw is not None:
        assert set(a.intw.keys()) == set(b.intw.keys())
        for key in a.intw:
            assert torch.allclose(a.intw[key].weights, b.intw[key].weights)
            assert a.intw[key].cgspec.get_spins() == b.intw[key].cgspec.get_spins()
            assert a.intw[key].cgspec.get_directions() == b.intw[key].cgspec.get_directions()


def test_serialize_roundtrip_u1():
    """Test serialize/deserialize round-trip for U1 tensors of order 2, 3, and 6."""
    group = U1Group()
    s = lambda q, d: Sector(q, d)
    idx = lambda dir, *sectors: Index(dir, group, sectors)
    O, I = Direction.OUT, Direction.IN

    # 2nd order — 5 sectors each
    t2 = Tensor.random([
        idx(O, s(-2,1), s(-1,2), s(0,3), s(1,2), s(2,1)),
        idx(I, s(-2,1), s(-1,2), s(0,3), s(1,2), s(2,1)),
    ], seed=10, itags=["a", "b"])
    t2.label = "U1-2nd"
    _assert_tensors_equal(deserialize(serialize(t2)), t2)

    # 3rd order — 2 OUT + 1 IN, 4-5 sectors each
    t3 = Tensor.random([
        idx(O, s(-1,2), s(0,3), s(1,2), s(2,1)),
        idx(O, s(-2,1), s(-1,2), s(0,3), s(1,2)),
        idx(I, s(-3,1), s(-2,2), s(-1,3), s(0,3), s(1,2), s(2,1), s(3,1)),
    ], seed=11, itags=["a", "b", "c"])
    _assert_tensors_equal(deserialize(serialize(t3)), t3)

    # 6th order — 3 OUT + 3 IN, 4 sectors each
    idx_o = idx(O, s(-1,2), s(0,2), s(1,2), s(2,2))
    idx_i = idx(I, s(-1,2), s(0,2), s(1,2), s(2,2))
    t6 = Tensor.random([idx_o, idx_o, idx_o, idx_i, idx_i, idx_i],
                       seed=12, itags=["a", "b", "c", "d", "e", "f"])
    _assert_tensors_equal(deserialize(serialize(t6)), t6)


def test_serialize_roundtrip_z2():
    """Test serialize/deserialize round-trip for Z2 tensors of order 2, 3, and 6."""
    group = Z2Group()
    s = lambda q, d: Sector(q, d)
    idx = lambda dir, *sectors: Index(dir, group, sectors)
    O, I = Direction.OUT, Direction.IN

    # 2nd order — both sectors, larger dims
    t2 = Tensor.random([
        idx(O, s(0,5), s(1,5)),
        idx(I, s(0,5), s(1,5)),
    ], seed=20, itags=["p", "q"])
    _assert_tensors_equal(deserialize(serialize(t2)), t2)

    # 3rd order — 2 OUT + 1 IN
    t3 = Tensor.random([
        idx(O, s(0,4), s(1,4)),
        idx(O, s(0,3), s(1,3)),
        idx(I, s(0,5), s(1,5)),
    ], seed=21, itags=["p", "q", "r"])
    _assert_tensors_equal(deserialize(serialize(t3)), t3)

    # 6th order — 3 OUT + 3 IN
    idx_o = idx(O, s(0,3), s(1,3))
    idx_i = idx(I, s(0,3), s(1,3))
    t6 = Tensor.random([idx_o, idx_o, idx_o, idx_i, idx_i, idx_i],
                       seed=22, itags=["a", "b", "c", "d", "e", "f"])
    _assert_tensors_equal(deserialize(serialize(t6)), t6)


def test_serialize_roundtrip_product_abelian():
    """Test serialize/deserialize round-trip for U1×Z2 tensors of order 2, 3, and 6."""
    group = ProductGroup([U1Group(), Z2Group()])
    s = lambda q, d: Sector(q, d)
    idx = lambda dir, *sectors: Index(dir, group, sectors)
    O, I = Direction.OUT, Direction.IN

    # 2nd order — 6 sectors
    t2 = Tensor.random([
        idx(O, s((0,0),2), s((0,1),2), s((1,0),2), s((1,1),2), s((-1,0),2), s((-1,1),2)),
        idx(I, s((0,0),2), s((0,1),2), s((1,0),2), s((1,1),2), s((-1,0),2), s((-1,1),2)),
    ], seed=30, itags=["x", "y"])
    _assert_tensors_equal(deserialize(serialize(t2)), t2)

    # 3rd order
    t3 = Tensor.random([
        idx(O, s((0,0),2), s((1,0),2), s((0,1),2), s((1,1),2)),
        idx(O, s((0,0),2), s((-1,0),2), s((0,1),2), s((-1,1),2)),
        idx(I, s((0,0),2), s((0,1),2), s((1,0),2), s((1,1),2), s((-1,0),2), s((-1,1),2)),
    ], seed=31, itags=["x", "y", "z"])
    _assert_tensors_equal(deserialize(serialize(t3)), t3)

    # 6th order
    idx_o = idx(O, s((0,0),2), s((1,0),2), s((0,1),2), s((1,1),2))
    idx_i = idx(I, s((0,0),2), s((1,0),2), s((0,1),2), s((1,1),2))
    t6 = Tensor.random([idx_o, idx_o, idx_o, idx_i, idx_i, idx_i],
                       seed=32, itags=["a", "b", "c", "d", "e", "f"])
    _assert_tensors_equal(deserialize(serialize(t6)), t6)


def test_serialize_roundtrip_su2():
    """Test serialize/deserialize round-trip for SU2 tensors of order 2, 3, and 6."""
    group = SU2Group()
    s = lambda q, d: Sector(q, d)
    idx = lambda dir, *sectors: Index(dir, group, sectors)
    O, I = Direction.OUT, Direction.IN

    # 2nd order — 4 sectors, populated weights
    t2 = Tensor.random([
        idx(I, s(0,2), s(1,3), s(2,2), s(3,1)),
        idx(O, s(0,2), s(1,3), s(2,2), s(3,1)),
    ], seed=40, itags=["a", "b"])
    populate_random_weights(t2, seed=41)
    _assert_tensors_equal(deserialize(serialize(t2)), t2)

    # 3rd order — 3-4 sectors each, populated weights
    t3 = Tensor.random([
        idx(I, s(1,2), s(2,2), s(3,1)),
        idx(I, s(0,2), s(1,3), s(2,2)),
        idx(O, s(0,2), s(1,3), s(2,2), s(3,1)),
    ], seed=42, itags=["a", "b", "c"])
    populate_random_weights(t3, seed=43)
    _assert_tensors_equal(deserialize(serialize(t3)), t3)

    # 6th order — 3 IN + 3 OUT, populated weights
    idx_i = idx(I, s(1,2), s(2,2))
    idx_o = idx(O, s(1,2), s(2,2))
    t6 = Tensor.random([idx_i, idx_i, idx_i, idx_o, idx_o, idx_o],
                       seed=44, itags=["a", "b", "c", "d", "e", "f"])
    populate_random_weights(t6, seed=45)
    _assert_tensors_equal(deserialize(serialize(t6)), t6)


def test_serialize_roundtrip_product_su2():
    """Test serialize/deserialize round-trip for U1×SU2 tensors of order 2, 3, and 6."""
    group = ProductGroup([U1Group(), SU2Group()])
    s = lambda q, d: Sector(q, d)
    idx = lambda dir, *sectors: Index(dir, group, sectors)
    O, I = Direction.OUT, Direction.IN

    # 2nd order — 5 sectors, populated weights
    t2 = Tensor.random([
        idx(O, s((0,0),2), s((0,1),2), s((0,2),2), s((1,1),2), s((-1,1),2)),
        idx(I, s((0,0),2), s((0,1),2), s((0,2),2), s((1,1),2), s((-1,1),2)),
    ], seed=50, itags=["u", "v"])
    populate_random_weights(t2, seed=51)
    _assert_tensors_equal(deserialize(serialize(t2)), t2)

    # 3rd order — 3-4 sectors, populated weights
    t3 = Tensor.random([
        idx(O, s((0,0),2), s((1,0),2), s((0,1),2), s((1,1),2)),
        idx(O, s((0,0),2), s((-1,0),2), s((0,1),2), s((-1,1),2)),
        idx(I, s((0,0),2), s((0,1),2), s((1,0),2), s((1,1),2), s((-1,0),2), s((-1,1),2)),
    ], seed=52, itags=["u", "v", "w"])
    populate_random_weights(t3, seed=53)
    _assert_tensors_equal(deserialize(serialize(t3)), t3)

    # 6th order — 3 OUT + 3 IN, populated weights
    idx_o = idx(O, s((0,1),2), s((1,1),2), s((-1,1),2))
    idx_i = idx(I, s((0,1),2), s((1,1),2), s((-1,1),2))
    t6 = Tensor.random([idx_o, idx_o, idx_o, idx_i, idx_i, idx_i],
                       seed=54, itags=["a", "b", "c", "d", "e", "f"])
    populate_random_weights(t6, seed=55)
    _assert_tensors_equal(deserialize(serialize(t6)), t6)


def test_serialize_roundtrip_complex_dtype():
    """Test serialize/deserialize round-trip for complex128 dtype."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, (Sector(-1, 2), Sector(0, 3), Sector(1, 2), Sector(2, 1)))
    idx_b = Index(Direction.IN,  group, (Sector(-1, 2), Sector(0, 3), Sector(1, 2), Sector(2, 1)))
    t = Tensor.random([idx_a, idx_b], seed=60, dtype=torch.complex128, itags=["a", "b"])

    rt = deserialize(serialize(t))

    assert rt.dtype == torch.complex128
    _assert_tensors_equal(rt, t)


def test_serialize_roundtrip_scalar():
    """Test serialize/deserialize round-trip for a scalar tensor."""
    t = Tensor.from_scalar(3.14, label="pi")

    rt = deserialize(serialize(t))

    assert rt.label == "pi"
    assert rt.is_scalar()
    assert torch.equal(rt.data[()], t.data[()])


def test_serialize_payload_primitive_types():
    """Test that the serialized dict contains only primitives and torch.Tensor."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, (Sector(-1, 2), Sector(0, 3), Sector(1, 2)))
    idx_b = Index(Direction.IN,  group, (Sector(-1, 2), Sector(0, 3), Sector(1, 2)))
    t = Tensor.random([idx_a, idx_b], seed=70, itags=["a", "b"])

    payload = serialize(t)

    def _check(obj):
        if isinstance(obj, torch.Tensor):
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert isinstance(k, str), f"dict key {k!r} is not str"
                _check(v)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _check(item)
        else:
            assert isinstance(obj, (int, float, str, bool, type(None))), (
                f"unexpected type {type(obj).__name__}: {obj!r}"
            )

    _check(payload)


def test_serialize_torch_save_load(tmp_path):
    """Test that serialize output survives torch.save / torch.load with weights_only=True."""
    group = ProductGroup([U1Group(), SU2Group()])
    idx_o = Index(Direction.OUT, group, (Sector((0,1), 2), Sector((1,1), 2), Sector((-1,1), 2)))
    idx_i = Index(Direction.IN,  group, (Sector((0,1), 2), Sector((1,1), 2), Sector((-1,1), 2)))
    t = Tensor.random([idx_o, idx_o, idx_o, idx_i, idx_i, idx_i],
                      seed=80, itags=["a", "b", "c", "d", "e", "f"])
    populate_random_weights(t, seed=81)

    path = tmp_path / "tensor.tnsr"
    torch.save(serialize(t), str(path))
    payload = torch.load(str(path), weights_only=True)
    rt = deserialize(payload)

    _assert_tensors_equal(rt, t)


def test_deserialize_device_arg():
    """Test that deserialize places tensors on the requested device."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, (Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx_b = Index(Direction.IN,  group, (Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    t = Tensor.random([idx_a, idx_b], seed=90, itags=["a", "b"])

    rt = deserialize(serialize(t), device="cpu")

    for block in rt.data.values():
        assert block.device.type == "cpu"


def test_deserialize_unknown_version_raises():
    """Test that deserialize raises on an unsupported version number."""
    payload = {"version": 99, "label": "x", "dtype": "float64",
               "itags": (), "indices": [], "data": [], "intw": None}
    with pytest.raises(ValueError, match="Unsupported serialization version"):
        deserialize(payload)

