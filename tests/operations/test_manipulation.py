# Copyright (C) 2025-2026 Changkai Zhang.
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


"""Tests for tensor manipulation operations: conj, permute, transpose, retag."""

import math
import torch
import pytest
import yuzuha

from nicole import Direction, Index, Sector, Tensor
from nicole import conj, permute, transpose, merge_axes, contract
from nicole import ProductGroup, U1Group, Z2Group, SU2Group
from ..utils import assert_blocks_equal, assert_charge_neutral


# ============================================================================
# Conjugation tests
# ============================================================================
# Three modes of conjugation:
# 1. tensor.conj(in_place=False) - Method, default, shares data efficiently
# 2. tensor.conj(in_place=True) - Method, in-place modification, returns self
# 3. conj(tensor) - Functional, clones all data for full isolation
# ============================================================================

# --- Method: tensor.conj(in_place=False) - Default, efficient sharing ---

def test_conj_method_basic():
    """Test method conj(in_place=False) basic behavior."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=123, dtype=torch.complex128, itags=["A", "B"])

    tensor_conj = tensor.conj(in_place=False)
    
    # Returns new instance
    assert tensor_conj is not tensor
    
    # Conjugates data
    for key in tensor.data:
        assert torch.allclose(tensor_conj.data[key], torch.conj(tensor.data[key]))
    
    # Flips directions
    for orig_idx, new_idx in zip(tensor.indices, tensor_conj.indices):
        assert new_idx.direction == orig_idx.direction.reverse()


def test_conj_method_shares_data_real():
    """Test that method conj(in_place=False) shares data for real dtype."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    tensor = Tensor.random([idx, idx.flip()], seed=1, dtype=torch.float64, itags=["A", "B"])
    
    tensor_conj = tensor.conj(in_place=False)
    
    # Data blocks are shared (same object)
    for key in tensor.data:
        assert tensor_conj.data[key] is tensor.data[key]


def test_conj_method_shares_views_complex():
    """Test that method conj(in_place=False) creates conjugate views for complex dtype."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=1, dtype=torch.complex128, itags=["A", "B"])
    
    tensor_conj = tensor.conj(in_place=False)
    
    # torch.conj returns a view, not a clone
    # Modify original and verify conj view reflects it
    key = list(tensor.data.keys())[0]
    original_value = tensor.data[key].clone()
    tensor.data[key][:] = 0
    
    # Conjugate view should also be zero now
    assert torch.allclose(tensor_conj.data[key], torch.zeros_like(original_value))


def test_conj_method_double_application():
    """Test method double conjugation restores original (involution)."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    tensor = Tensor.random([idx, idx.flip()], seed=1, dtype=torch.complex128, itags=["A", "B"])
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    original_direction = tensor.indices[0].direction
    
    double_conj = tensor.conj().conj()
    
    # Data restored
    for key in original_data:
        assert torch.allclose(double_conj.data[key], original_data[key])
    
    # Directions restored
    assert double_conj.indices[0].direction == original_direction


def test_conj_method_su2_basic():
    """Test method conj(in_place=False) basic behavior with SU2 group."""
    from nicole.identity import identity
    
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 3), Sector(2, 4), Sector(3, 2)
    ))
    
    tensor = identity(idx)
    assert tensor.intw is not None
    
    tensor_conj = tensor.conj(in_place=False)
    
    # Returns new instance
    assert tensor_conj is not tensor
    
    # Has intw
    assert tensor_conj.intw is not None
    assert len(tensor_conj.intw) == len(tensor.intw)
    
    # Flips directions
    for orig_idx, new_idx in zip(tensor.indices, tensor_conj.indices):
        assert new_idx.direction == orig_idx.direction.reverse()


def test_conj_method_su2_shares_data_and_weights():
    """Test that method conj(in_place=False) shares data and scales intw weights for SU2."""
    from nicole.identity import identity
    
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(2, 3)))
    
    tensor = identity(idx)
    tensor_conj = tensor.conj(in_place=False)
    
    # Data blocks are shared
    for key in tensor.data:
        assert tensor_conj.data[key] is tensor.data[key]
    
    # intw weights are scaled by the FS phase (±1)
    for key in tensor.intw:
        orig_bridge = tensor.intw[key]
        new_bridge = tensor_conj.intw[key]
        phase, _ = yuzuha.compute_conjugate(orig_bridge.cgspec)
        assert torch.allclose(new_bridge.weights, orig_bridge.weights * phase)


def test_conj_method_su2_three_indices():
    """Test method conj(in_place=False) with SU2 three-index tensor (isometry)."""
    from nicole.identity import isometry
    
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(3, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 3), Sector(2, 2)))
    
    tensor = isometry(idx1, idx2)
    assert tensor.intw is not None
    
    tensor_conj = tensor.conj(in_place=False)
    
    # All indices flipped
    for orig_idx, new_idx in zip(tensor.indices, tensor_conj.indices):
        assert new_idx.direction == orig_idx.direction.reverse()
    
    # intw updated for all blocks
    assert len(tensor_conj.intw) == len(tensor.intw)
    for key in tensor.intw:
        assert key in tensor_conj.intw
        
        orig_bridge = tensor.intw[key]
        new_bridge = tensor_conj.intw[key]
        
        # Weights scaled by the FS phase (±1)
        phase, _ = yuzuha.compute_conjugate(orig_bridge.cgspec)
        assert torch.allclose(new_bridge.weights, orig_bridge.weights * phase)
        
        # Bridge directions flipped
        orig_edges = orig_bridge.cgspec.edges
        new_edges = new_bridge.cgspec.edges
        
        for orig_edge, new_edge in zip(orig_edges, new_edges):
            assert orig_edge.dir == new_edge.dir.flip()
            assert orig_edge.j.twice() == new_edge.j.twice()


# --- Method: tensor.conj(in_place=True) - In-place modification ---

def test_conj_inplace_modifies_and_returns_self():
    """Test method conj(in_place=True) modifies in place and returns self."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    tensor = Tensor.random([idx, idx.flip()], seed=1, dtype=torch.complex128, itags=["A", "B"])
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    original_direction = tensor.indices[0].direction
    
    result = tensor.conj(in_place=True)
    
    # Returns self
    assert result is tensor
    
    # Data conjugated
    for key in original_data:
        assert torch.allclose(tensor.data[key], torch.conj(original_data[key]))
    
    # Direction flipped
    assert tensor.indices[0].direction == original_direction.reverse()


def test_conj_inplace_allows_chaining():
    """Test that method conj(in_place=True) enables chaining."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    tensor = Tensor.random([idx, idx.flip()], seed=1, dtype=torch.complex128, itags=["A", "B"])
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    # Chain double conjugation
    result = tensor.conj(in_place=True).conj(in_place=True)
    
    # Returns self
    assert result is tensor
    
    # Double conj restores original
    for key in original_data:
        assert torch.allclose(tensor.data[key], original_data[key])
    assert tensor.indices[0].direction == Direction.OUT


def test_conj_inplace_su2_modifies_and_returns_self():
    """Test method conj(in_place=True) with SU2 modifies in place and returns self."""
    from nicole.identity import identity
    
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 3), Sector(2, 4), Sector(3, 2)
    ))
    
    tensor = identity(idx)
    assert tensor.intw is not None
    
    original_weights = {k: v.weights.clone() for k, v in tensor.intw.items()}
    original_directions = [idx.direction for idx in tensor.indices]
    
    result = tensor.conj(in_place=True)
    
    # Returns self
    assert result is tensor
    
    # intw was updated
    assert tensor.intw is not None
    assert len(tensor.intw) == len(original_weights)
    
    # Weights unchanged
    for key in original_weights:
        assert torch.allclose(tensor.intw[key].weights, original_weights[key])
    
    # Directions flipped
    for orig_dir, new_idx in zip(original_directions, tensor.indices):
        assert new_idx.direction == orig_dir.reverse()
    
    # Bridge directions flipped
    for key, bridge in tensor.intw.items():
        edges = bridge.cgspec.edges
        for i, edge in enumerate(edges):
            if original_directions[i].value == 1:  # IN
                assert edge.dir.is_outgoing()
            else:  # OUT
                assert edge.dir.is_incoming()


# --- Functional: conj(tensor) - Full cloning for isolation ---

def test_conj_functional_clones_data():
    """Test functional conj() clones all data for independence."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    tensor = Tensor.random([idx, idx.flip()], seed=1, dtype=torch.float64, itags=["A", "B"])
    
    tensor_conj = conj(tensor)
    
    # Returns new instance
    assert tensor_conj is not tensor
    
    # Data blocks are cloned (different objects)
    for key in tensor.data:
        assert tensor_conj.data[key] is not tensor.data[key]
    
    # Directions flipped
    for orig_idx, new_idx in zip(tensor.indices, tensor_conj.indices):
        assert new_idx.direction == orig_idx.direction.reverse()


def test_conj_functional_real_dtype():
    """Test functional conj() with real dtype clones data."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=1, dtype=torch.float64, itags=["A", "B"])
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    tensor_conj = conj(tensor)
    
    # Data values unchanged for real dtype
    for key in original_data:
        assert torch.allclose(tensor_conj.data[key], original_data[key])
    
    # But data is cloned
    for key in tensor.data:
        assert tensor_conj.data[key] is not tensor.data[key]


def test_conj_functional_double_application():
    """Test functional conj() double application restores original (involution)."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    tensor = Tensor.random([idx, idx.flip()], seed=1, dtype=torch.complex128, itags=["A", "B"])
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    original_direction = tensor.indices[0].direction
    
    double_conj = conj(conj(tensor))
    
    # Data restored
    for key in original_data:
        assert torch.allclose(double_conj.data[key], original_data[key])
    
    # Directions restored
    assert double_conj.indices[0].direction == original_direction


def test_conj_functional_su2_clones_and_updates_intw():
    """Test functional conj() with SU2 clones data and updates intw."""
    from nicole.identity import identity
    
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 3), Sector(2, 4), Sector(3, 2)
    ))
    
    tensor = identity(idx)
    assert tensor.intw is not None
    
    tensor_conj = conj(tensor)
    
    # Returns new instance
    assert tensor_conj is not tensor
    
    # Data is cloned (functional version clones for isolation)
    for key in tensor.data:
        assert tensor_conj.data[key] is not tensor.data[key]
    
    # intw was updated
    assert tensor_conj.intw is not None
    assert len(tensor_conj.intw) == len(tensor.intw)
    
    # Directions were flipped in Bridge objects
    for key in tensor.intw:
        orig_bridge = tensor.intw[key]
        new_bridge = tensor_conj.intw[key]
        
        # Weights are scaled by the FS phase (±1)
        phase, _ = yuzuha.compute_conjugate(orig_bridge.cgspec)
        assert torch.allclose(new_bridge.weights, orig_bridge.weights * phase)
        
        # CGSpec has flipped directions
        orig_edges = orig_bridge.cgspec.edges
        new_edges = new_bridge.cgspec.edges
        
        for orig_edge, new_edge in zip(orig_edges, new_edges):
            assert orig_edge.dir == new_edge.dir.flip()
            assert orig_edge.j.twice() == new_edge.j.twice()


def test_conj_functional_su2_double_application():
    """Test functional conj() double application with SU2 preserves intw."""
    from nicole.identity import identity
    
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 3), Sector(2, 4), Sector(3, 2)
    ))
    
    tensor = identity(idx)
    original_weights = {k: v.weights.clone() for k, v in tensor.intw.items()}
    original_directions = [idx.direction for idx in tensor.indices]
    
    double_conj = conj(conj(tensor))
    
    # Weights unchanged after double conjugation
    for key in original_weights:
        assert torch.allclose(double_conj.intw[key].weights, original_weights[key])
    
    # Directions restored
    for orig_idx, final_idx in zip(tensor.indices, double_conj.indices):
        assert orig_idx.direction == final_idx.direction


def test_conj_functional_su2_three_indices():
    """Test functional conj() with SU2 three-index tensor (isometry)."""
    from nicole.identity import isometry
    
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(3, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 3), Sector(2, 2)))
    
    tensor = isometry(idx1, idx2)
    assert tensor.intw is not None
    
    tensor_conj = conj(tensor)
    
    # All indices flipped
    for orig_idx, new_idx in zip(tensor.indices, tensor_conj.indices):
        assert new_idx.direction == orig_idx.direction.reverse()
    
    # intw updated for all blocks
    assert len(tensor_conj.intw) == len(tensor.intw)
    for key in tensor.intw:
        assert key in tensor_conj.intw
        
        orig_bridge = tensor.intw[key]
        new_bridge = tensor_conj.intw[key]
        
        # Weights scaled by the FS phase (±1)
        phase, _ = yuzuha.compute_conjugate(orig_bridge.cgspec)
        assert torch.allclose(new_bridge.weights, orig_bridge.weights * phase)
        
        # Bridge directions flipped
        orig_edges = orig_bridge.cgspec.edges
        new_edges = new_bridge.cgspec.edges
        
        for orig_edge, new_edge in zip(orig_edges, new_edges):
            assert orig_edge.dir == new_edge.dir.flip()
            assert orig_edge.j.twice() == new_edge.j.twice()

# ============================================================================
# Permutation tests
# ============================================================================
# Three modes of permutation:
# 1. tensor.permute(order, in_place=True) - Method, default, modifies in-place
# 2. tensor.permute(order, in_place=False) - Method, shares data efficiently
# 3. permute(tensor, order) - Functional, clones all data for full isolation
# ============================================================================

# --- Method: tensor.permute(order, in_place=True) - Default, in-place ---

def test_permute_method_inplace_basic():
    """Test method permute(in_place=True) basic behavior."""
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
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    result = tensor.permute(order)
    
    # Returns self
    assert result is tensor
    
    # itags reordered
    assert list(tensor.itags) == [itags[i] for i in order]
    
    # Data blocks reordered
    for key, block in original_data.items():
        new_key = tuple(key[i] for i in order)
        assert torch.allclose(tensor.data[new_key], torch.permute(block, order))


def test_permute_method_inplace_allows_chaining():
    """Test that method permute(in_place=True) enables chaining."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 1))),
    ]
    tensor = Tensor.random(indices, seed=10, itags=["a", "b", "c"])
    
    # Chain: [a,b,c] -> [c,a,b] -> [b,c,a]
    result = tensor.permute([2, 0, 1]).permute([2, 0, 1])
    
    # Returns self
    assert result is tensor
    
    # Final order
    assert list(tensor.itags) == ["b", "c", "a"]


def test_permute_method_inplace_identity():
    """Test that identity permutation with in_place=True returns self."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx, idx], seed=1, itags=["a", "b", "c"])
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    result = tensor.permute([0, 1, 2])
    
    # Returns self
    assert result is tensor
    
    # Data unchanged
    assert list(tensor.itags) == ["a", "b", "c"]
    for key in original_data:
        assert torch.allclose(tensor.data[key], original_data[key])


def test_permute_method_inplace_invalid_order():
    """Test that invalid permutation raises error."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    with pytest.raises(ValueError, match="Invalid permutation"):
        tensor.permute([0, 0])
    
    with pytest.raises(ValueError, match="Invalid permutation"):
        tensor.permute([0, 2])


# --- Method: tensor.permute(order, in_place=False) - Efficient sharing ---

def test_permute_method_shares_data():
    """Test that method permute(in_place=False) shares data efficiently."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
    ]
    tensor = Tensor.random(indices, seed=1, itags=["a", "b"])
    
    tensor_perm = tensor.permute([1, 0], in_place=False)
    
    # Returns new instance
    assert tensor_perm is not tensor
    
    # torch.permute creates views, so data is shared
    # Modify original and verify permuted view reflects it
    key = list(tensor.data.keys())[0]
    original_value = tensor.data[key].clone()
    tensor.data[key][:] = 0
    
    # Permuted view should also be zero now
    new_key = (key[1], key[0])
    assert torch.allclose(tensor_perm.data[new_key], torch.zeros_like(original_value).T)


def test_permute_method_not_inplace_basic():
    """Test method permute(in_place=False) basic behavior."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 1))),
    ]
    tensor = Tensor.random(indices, seed=10, itags=["a", "b", "c"])
    order = [2, 0, 1]
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    original_itags = list(tensor.itags)
    
    tensor_perm = tensor.permute(order, in_place=False)
    
    # Returns new instance
    assert tensor_perm is not tensor
    
    # Original unchanged
    assert list(tensor.itags) == original_itags
    
    # New tensor has reordered itags
    assert list(tensor_perm.itags) == [original_itags[i] for i in order]


# --- Functional: permute(tensor, order) - Full cloning for isolation ---

def test_permute_functional_clones_data():
    """Test functional permute() clones all data for independence."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 1))),
    ]
    tensor = Tensor.random(indices, seed=10, itags=["a", "b", "c"])
    
    permuted = permute(tensor, [2, 0, 1])
    
    # Returns new instance
    assert permuted is not tensor
    
    # Data blocks are cloned (different objects)
    for key in tensor.data:
        new_key = tuple(key[i] for i in [2, 0, 1])
        assert permuted.data[new_key] is not tensor.data[key]


def test_permute_functional_reorders_correctly():
    """Test that functional permute correctly reorders indices and blocks."""
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
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    permuted = permute(tensor, order)
    
    # itags reordered
    assert list(permuted.itags) == [itags[i] for i in order]
    
    # Data blocks reordered
    for key, block in original_data.items():
        new_key = tuple(key[i] for i in order)
        assert torch.allclose(permuted.data[new_key], torch.permute(block, order))


def test_permute_functional_identity():
    """Test that functional identity permutation still clones data."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx, idx], seed=1, itags=["a", "b", "c"])
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    permuted = permute(tensor, [0, 1, 2])
    
    # Returns new instance
    assert permuted is not tensor
    
    # itags unchanged
    assert list(permuted.itags) == ["a", "b", "c"]
    
    # Data values unchanged
    for key in original_data:
        assert torch.allclose(permuted.data[key], original_data[key])
    
    # But data is cloned
    for key in tensor.data:
        assert permuted.data[key] is not tensor.data[key]


def test_permute_functional_invalid_order():
    """Test that functional permute with invalid order raises error."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    with pytest.raises(ValueError, match="Invalid permutation"):
        permute(tensor, [0, 0])
    
    with pytest.raises(ValueError, match="Invalid permutation"):
        permute(tensor, [0, 2])


# --- SU(2) permutation tests ---

def test_permute_method_su2_basic():
    """Test method permute with SU(2) tensor updates intw correctly."""
    from nicole.identity import identity
    
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 3), Sector(2, 2)
    ))
    
    tensor = identity(idx)
    assert tensor.intw is not None
    original_intw_keys = set(tensor.intw.keys())
    
    # Swap two identical indices
    result = tensor.permute([1, 0], in_place=False)
    
    # Returns new instance
    assert result is not tensor
    
    # intw keys should be swapped
    assert len(result.intw) == len(original_intw_keys)
    for key in original_intw_keys:
        swapped_key = (key[1], key[0])
        assert swapped_key in result.intw


def test_permute_method_su2_three_indices():
    """Test method permute with SU(2) three-index tensor (isometry)."""
    from nicole.identity import isometry
    
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 3), Sector(3, 2)))
    
    tensor = isometry(idx1, idx2)
    assert tensor.intw is not None
    
    # Permute: [0, 1, 2] -> [2, 1, 0]
    order = [2, 1, 0]
    tensor_perm = tensor.permute(order, in_place=False)
    
    # All indices permuted
    for i, orig_idx in enumerate(tensor.indices):
        assert tensor_perm.indices[order.index(i)].sectors == orig_idx.sectors
    
    # intw updated for all blocks
    assert len(tensor_perm.intw) == len(tensor.intw)
    for key in tensor.intw:
        new_key = tuple(key[i] for i in order)
        assert new_key in tensor_perm.intw


def test_permute_method_su2_preserves_norm():
    """Test that permute with SU(2) preserves tensor norm (unitarity of R-symbol)."""
    from nicole.identity import isometry
    
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 2)))
    
    tensor = isometry(idx1, idx2)
    original_norm = tensor.norm()
    
    # Permute
    tensor_perm = tensor.permute([1, 2, 0], in_place=False)
    perm_norm = tensor_perm.norm()
    
    # Norm should be preserved (R is unitary)
    assert math.isclose(original_norm, perm_norm, rel_tol=1e-12, abs_tol=1e-15)


def test_permute_functional_su2_clones_and_updates_intw():
    """Test functional permute with SU(2) clones data and updates intw."""
    from nicole.identity import identity
    
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 3), Sector(2, 2)
    ))
    
    tensor = identity(idx)
    assert tensor.intw is not None
    
    tensor_perm = permute(tensor, [1, 0])
    
    # Returns new instance
    assert tensor_perm is not tensor
    
    # Data is cloned (functional version clones for isolation)
    for key in tensor.data:
        swapped_key = (key[1], key[0])
        assert tensor_perm.data[swapped_key] is not tensor.data[key]
    
    # intw was updated
    assert tensor_perm.intw is not None
    assert len(tensor_perm.intw) == len(tensor.intw)


def test_permute_inplace_su2_modifies_and_returns_self():
    """Test method permute(in_place=True) with SU(2) modifies in place."""
    from nicole.identity import identity
    
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 3)
    ))
    
    tensor = identity(idx)
    assert tensor.intw is not None
    
    original_intw_keys = list(tensor.intw.keys())
    
    result = tensor.permute([1, 0])
    
    # Returns self
    assert result is tensor
    
    # intw was updated
    assert tensor.intw is not None
    for key in original_intw_keys:
        swapped_key = (key[1], key[0])
        assert swapped_key in tensor.intw


def test_permute_su2_four_indices():
    """Test permute with SU(2) four-index tensor."""
    group = SU2Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3))),
        Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 2))),
        Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2))),
    ]
    tensor = Tensor.random(indices, seed=42, dtype=torch.float64, itags=["a", "b", "c", "d"])
    
    assert tensor.intw is not None
    original_norm = tensor.norm()
    
    # Complex permutation: [0,1,2,3] -> [2,0,3,1]
    order = [2, 0, 3, 1]
    tensor_perm = tensor.permute(order, in_place=False)
    
    # Norm preserved
    perm_norm = tensor_perm.norm()
    assert math.isclose(original_norm, perm_norm, rel_tol=1e-12, abs_tol=1e-15)
    
    # itags reordered
    assert list(tensor_perm.itags) == ["c", "a", "d", "b"]
    
    # intw updated for all blocks
    assert len(tensor_perm.intw) == len(tensor.intw)


def test_permute_su2_inverse_restores_original():
    """Test that permuting and then applying inverse permutation restores original."""
    from nicole.identity import isometry
    
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3), Sector(3, 2)))
    
    tensor = isometry(idx1, idx2)
    
    # Store original data and weights
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    original_weights = {k: v.weights.clone() for k, v in tensor.intw.items()}
    original_itags = list(tensor.itags)
    
    # Permute: [0, 1, 2] -> [2, 0, 1]
    order = [2, 0, 1]
    tensor_perm = tensor.permute(order, in_place=False)
    
    # Inverse permutation to restore: [2, 0, 1] -> [0, 1, 2]
    # If order[i] = j, then inverse_order[j] = i
    inverse_order = [0] * len(order)
    for i, j in enumerate(order):
        inverse_order[j] = i
    
    tensor_restored = tensor_perm.permute(inverse_order, in_place=False)
    
    # itags restored
    assert list(tensor_restored.itags) == original_itags
    
    # Data restored
    for key in original_data:
        assert key in tensor_restored.data
        assert torch.allclose(tensor_restored.data[key], original_data[key], rtol=1e-12, atol=1e-15)
    
    # Weights restored
    for key in original_weights:
        assert key in tensor_restored.intw
        assert torch.allclose(tensor_restored.intw[key].weights, original_weights[key], rtol=1e-12, atol=1e-15)


def test_permute_su2_double_application_identity():
    """Test that applying the same permutation twice may not restore original (not involution)."""
    from nicole.identity import identity
    
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 3), Sector(2, 2)
    ))
    
    tensor = identity(idx)
    original_norm = tensor.norm()
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    original_weights = {k: v.weights.clone() for k, v in tensor.intw.items()}
    
    # Permute twice with involution permutation: [0,1] -> [1,0] -> [0,1]
    # This IS an involution (swap is its own inverse)
    tensor_twice = tensor.permute([1, 0], in_place=False).permute([1, 0], in_place=False)
    
    # Should restore original
    assert list(tensor_twice.itags) == list(tensor.itags)
    assert math.isclose(tensor_twice.norm(), original_norm, rel_tol=1e-12, abs_tol=1e-15)
    
    # Data restored
    for key in original_data:
        assert key in tensor_twice.data
        assert torch.allclose(tensor_twice.data[key], original_data[key], rtol=1e-12, atol=1e-15)
    
    # Weights restored
    for key in original_weights:
        assert key in tensor_twice.intw
        assert torch.allclose(tensor_twice.intw[key].weights, original_weights[key], rtol=1e-12, atol=1e-15)
    
    # For consistency, test a cycle: [0,1,2] -> [1,2,0] -> [2,0,1] -> [0,1,2]
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    tensor3 = Tensor.random([idx3, idx3, idx3], seed=123, dtype=torch.float64, itags=["a", "b", "c"])
    
    original_norm3 = tensor3.norm()
    original_data3 = {k: v.clone() for k, v in tensor3.data.items()}
    original_weights3 = {k: v.weights.clone() for k, v in tensor3.intw.items()}
    
    # Apply cycle 3 times: should restore
    cycle = [1, 2, 0]
    tensor3_cycled = tensor3.permute(cycle, in_place=False).permute(cycle, in_place=False).permute(cycle, in_place=False)
    
    assert list(tensor3_cycled.itags) == ["a", "b", "c"]
    assert math.isclose(tensor3_cycled.norm(), original_norm3, rel_tol=1e-12, abs_tol=1e-15)
    
    # Data restored
    for key in original_data3:
        assert key in tensor3_cycled.data
        assert torch.allclose(tensor3_cycled.data[key], original_data3[key], rtol=1e-12, atol=1e-15)
    
    # Weights restored
    for key in original_weights3:
        assert key in tensor3_cycled.intw
        assert torch.allclose(tensor3_cycled.intw[key].weights, original_weights3[key], rtol=1e-12, atol=1e-15)


def test_permute_su2_five_indices():
    """Test permute with SU(2) five-index tensor."""
    group = SU2Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 2))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 2))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
    ]
    tensor = Tensor.random(indices, seed=99, dtype=torch.float64, itags=["a", "b", "c", "d", "e"])
    
    assert tensor.intw is not None
    original_norm = tensor.norm()
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    original_weights = {k: v.weights.clone() for k, v in tensor.intw.items()}
    
    # Complex permutation: [0,1,2,3,4] -> [4,2,0,1,3]
    order = [4, 2, 0, 1, 3]
    tensor_perm = tensor.permute(order, in_place=False)
    
    # Norm preserved
    perm_norm = tensor_perm.norm()
    assert math.isclose(original_norm, perm_norm, rel_tol=1e-12, abs_tol=1e-15)
    
    # itags reordered
    assert list(tensor_perm.itags) == ["e", "c", "a", "b", "d"]
    
    # Verify we can permute back
    inverse_order = [0] * len(order)
    for i, j in enumerate(order):
        inverse_order[j] = i
    
    tensor_restored = tensor_perm.permute(inverse_order, in_place=False)
    assert list(tensor_restored.itags) == ["a", "b", "c", "d", "e"]
    assert math.isclose(tensor_restored.norm(), original_norm, rel_tol=1e-12, abs_tol=1e-15)
    
    # Data restored
    for key in original_data:
        assert key in tensor_restored.data
        assert torch.allclose(tensor_restored.data[key], original_data[key], rtol=1e-12, atol=1e-15)
    
    # Weights restored
    for key in original_weights:
        assert key in tensor_restored.intw
        assert torch.allclose(tensor_restored.intw[key].weights, original_weights[key], rtol=1e-12, atol=1e-15)


def test_permute_su2_multiple_sectors_per_index():
    """Test permute with SU(2) tensor having multiple sectors per index."""
    group = SU2Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(2, 3), Sector(3, 2))),
        Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3), Sector(3, 2), Sector(4, 2))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2), Sector(3, 3))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 3), Sector(3, 2))),
    ]
    tensor = Tensor.random(indices, seed=777, dtype=torch.float64, itags=["i", "j", "k", "l"])
    
    assert tensor.intw is not None
    original_norm = tensor.norm()
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    original_weights = {k: v.weights.clone() for k, v in tensor.intw.items()}
    num_blocks = len(tensor.data)
    
    # Permute
    order = [3, 1, 0, 2]
    tensor_perm = tensor.permute(order, in_place=False)
    
    # Norm preserved
    assert math.isclose(tensor_perm.norm(), original_norm, rel_tol=1e-12, abs_tol=1e-15)
    
    # Same number of blocks
    assert len(tensor_perm.data) == num_blocks
    assert len(tensor_perm.intw) == len(tensor.intw)
    
    # Inverse permutation restores
    inverse_order = [0] * len(order)
    for i, j in enumerate(order):
        inverse_order[j] = i
    
    tensor_restored = tensor_perm.permute(inverse_order, in_place=False)
    assert math.isclose(tensor_restored.norm(), original_norm, rel_tol=1e-12, abs_tol=1e-15)
    
    # Data restored
    for key in original_data:
        assert key in tensor_restored.data
        assert torch.allclose(tensor_restored.data[key], original_data[key], rtol=1e-12, atol=1e-15)
    
    # Weights restored
    for key in original_weights:
        assert key in tensor_restored.intw
        assert torch.allclose(tensor_restored.intw[key].weights, original_weights[key], rtol=1e-12, atol=1e-15)


# ============================================================================
# Transpose tests
# ============================================================================
# Three modes of transpose:
# 1. tensor.transpose(*order, in_place=True) - Method, default, modifies in-place
# 2. tensor.transpose(*order, in_place=False) - Method, shares data efficiently
# 3. transpose(tensor, *order) - Functional, clones all data for full isolation
# ============================================================================

# --- Method: tensor.transpose(*order, in_place=True) - Default, in-place ---

def test_transpose_method_inplace_default_reverses():
    """Test method transpose(in_place=True) with default reverses order."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 1))),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1))),
    ]
    itags = ["i0", "i1", "i2"]
    tensor = Tensor.random(indices, seed=11, itags=itags)
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    result = tensor.transpose()
    
    # Returns self
    assert result is tensor
    
    # itags reversed
    assert list(tensor.itags) == list(reversed(itags))
    
    # Data blocks transposed
    for key, block in original_data.items():
        new_key = tuple(reversed(key))
        assert torch.allclose(tensor.data[new_key], torch.permute(block, (2, 1, 0)))


def test_transpose_method_inplace_explicit_order():
    """Test method transpose(in_place=True) with explicit order."""
    group = U1Group()
    indices = [Index(Direction.OUT, group, sectors=(Sector(0, 1),)) for _ in range(3)]
    tensor = Tensor.random(indices, seed=1, itags=["a", "b", "c"])
    
    result = tensor.transpose(1, 0, 2)
    
    # Returns self
    assert result is tensor
    
    # itags reordered
    assert list(tensor.itags) == ["b", "a", "c"]


def test_transpose_method_inplace_allows_chaining():
    """Test that method transpose(in_place=True) enables chaining."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    # Chain double transpose (should restore)
    result = tensor.transpose().transpose()
    
    # Returns self
    assert result is tensor
    
    # Double transpose restores original
    assert list(tensor.itags) == ["a", "b"]
    for key in original_data:
        assert torch.allclose(tensor.data[key], original_data[key])


# --- Method: tensor.transpose(*order, in_place=False) - Efficient sharing ---

def test_transpose_method_not_inplace_basic():
    """Test method transpose(in_place=False) basic behavior."""
    group = U1Group()
    indices = [Index(Direction.OUT, group, sectors=(Sector(0, 1),)) for _ in range(2)]
    tensor = Tensor.random(indices, seed=1, itags=["a", "b"])
    
    original_itags = list(tensor.itags)
    
    tensor_T = tensor.transpose(in_place=False)
    
    # Returns new instance
    assert tensor_T is not tensor
    
    # Original unchanged
    assert list(tensor.itags) == original_itags
    
    # New tensor has reversed itags
    assert list(tensor_T.itags) == list(reversed(original_itags))


# --- Functional: transpose(tensor, *order) - Full cloning for isolation ---

def test_transpose_functional_default_reverses():
    """Test functional transpose() with default reverses order."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 1))),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1))),
    ]
    itags = ["i0", "i1", "i2"]
    tensor = Tensor.random(indices, seed=11, itags=itags)
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    transposed = transpose(tensor)
    
    # Returns new instance
    assert transposed is not tensor
    
    # itags reversed
    assert list(transposed.itags) == list(reversed(itags))
    
    # Data blocks transposed
    for key, block in original_data.items():
        new_key = tuple(reversed(key))
        assert torch.allclose(transposed.data[new_key], torch.permute(block, (2, 1, 0)))


def test_transpose_functional_explicit_order():
    """Test functional transpose() with explicit order."""
    group = U1Group()
    indices = [Index(Direction.OUT, group, sectors=(Sector(0, 1),)) for _ in range(3)]
    tensor = Tensor.random(indices, seed=1, itags=["a", "b", "c"])
    
    transposed = transpose(tensor, 1, 0, 2)
    
    # Returns new instance
    assert transposed is not tensor
    
    # itags reordered
    assert list(transposed.itags) == ["b", "a", "c"]


def test_transpose_functional_clones_data():
    """Test that functional transpose() clones all data for independence."""
    group = U1Group()
    indices = [Index(Direction.OUT, group, sectors=(Sector(0, 1),)) for _ in range(2)]
    tensor = Tensor.random(indices, seed=1, itags=["a", "b"])
    
    transposed = transpose(tensor)
    
    # Data blocks are cloned
    for key in tensor.data:
        new_key = (key[1], key[0])
        assert transposed.data[new_key] is not tensor.data[key]


def test_transpose_functional_double_application():
    """Test that functional transpose twice restores original (involution)."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx], seed=1, itags=["a", "b"])
    
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    double_transpose = transpose(transpose(tensor))
    
    # itags restored
    assert list(double_transpose.itags) == ["a", "b"]
    
    # Data restored
    for key in original_data:
        assert torch.allclose(double_transpose.data[key], original_data[key])

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
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    result = tensor.retag({"x": "left", "y": "right"})
    
    assert result is None  # retag() is in-place
    assert list(tensor.itags) == ["left", "right", "z"]
    for key in original_data:
        assert torch.allclose(tensor.data[key], original_data[key])


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
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    result = tensor.retag(["a", "b", "c"])
    
    assert result is None
    assert list(tensor.itags) == ["a", "b", "c"]
    for key in original_data:
        assert torch.allclose(tensor.data[key], original_data[key])


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
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    result = tensor.retag([0, 2], ["first", "third"])
    
    assert result is None
    assert list(tensor.itags) == ["first", "y", "third"]
    for key in original_data:
        assert torch.allclose(tensor.data[key], original_data[key])


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
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    # Test single int with single str
    result = tensor.retag(1, "middle")
    
    assert result is None
    assert list(tensor.itags) == ["x", "middle", "z"]
    for key in original_data:
        assert torch.allclose(tensor.data[key], original_data[key])
    
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


# Invert tests

def test_invert_single_index():
    """Test inverting a single index."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_direction_0 = tensor.indices[0].direction
    original_direction_1 = tensor.indices[1].direction
    original_charges_0 = tensor.indices[0].charges()
    original_charges_1 = tensor.indices[1].charges()
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    # Invert first index
    tensor.invert(0)
    
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
        assert torch.allclose(tensor.data[key], original_data[key])


def test_invert_multiple_indices():
    """Test inverting multiple indices at once."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
    
    tensor = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    original_directions = [idx.direction for idx in tensor.indices]
    original_charges = [idx.charges() for idx in tensor.indices]
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    # Invert indices 0 and 2
    tensor.invert([0, 2])
    
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
        assert torch.allclose(tensor.data[key], original_data[key])


def test_invert_uses_dual():
    """Test that tensor.invert() uses Index.dual() to maintain charge conservation."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(-1, 2)))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    
    # Get the original index
    original_charges = tensor.indices[0].charges()
    original_direction = tensor.indices[0].direction
    
    # Invert the tensor index
    tensor.invert(0)
    
    # Tensor.invert() should use Index.dual() internally
    # This means both direction is reversed AND charges are conjugated
    assert tensor.indices[0].direction == original_direction.reverse()
    assert tensor.indices[0].charges() == tuple(group.dual(c) for c in original_charges)
    assert tensor.indices[0].charges() == (0, -1, 1)  # For U1, dual(q) = -q


def test_invert_updates_block_keys():
    """Test that invert updates block keys to reflect conjugated charges."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_keys = set(tensor.data.keys())
    
    # Store data arrays by their original keys
    original_data_by_key = {k: v.clone() for k, v in tensor.data.items()}
    
    # Invert first index
    tensor.invert(0)
    
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
        assert torch.allclose(tensor.data[new_key], old_arr)


def test_invert_multiple_indices_updates_keys():
    """Test that inverting multiple indices updates all relevant positions in block keys."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
    
    tensor = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    original_keys = set(tensor.data.keys())
    
    # Invert indices 0 and 2
    tensor.invert([0, 2])
    
    # Block keys should be updated at positions 0 and 2
    expected_keys = set()
    for old_key in original_keys:
        new_key = (group.dual(old_key[0]), old_key[1], group.dual(old_key[2]))
        expected_keys.add(new_key)
    
    assert set(tensor.data.keys()) == expected_keys


def test_invert_all_indices():
    """Test inverting all indices."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_directions = [idx.direction for idx in tensor.indices]
    
    # Invert all indices
    tensor.invert([0, 1])
    
    # All directions should be reversed
    for i, orig_dir in enumerate(original_directions):
        assert tensor.indices[i].direction == orig_dir.reverse()


def test_invert_inplace_modification():
    """Test that invert modifies tensor in-place."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    original_id = id(tensor)
    
    result = tensor.invert(0)
    
    # Should return None (in-place operation)
    assert result is None
    # Tensor object should be the same
    assert id(tensor) == original_id


def test_invert_preserves_data():
    """Test that invert preserves all tensor data."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_norm = tensor.norm()
    original_keys = set(tensor.data.keys())
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    tensor.invert([0, 1])
    
    # Verify norm preserved
    assert math.isclose(tensor.norm(), original_norm)
    # Verify keys unchanged
    assert set(tensor.data.keys()) == original_keys
    # Verify data values unchanged
    for key in original_data:
        assert torch.allclose(tensor.data[key], original_data[key])


def test_invert_preserves_itags():
    """Test that invert preserves index tags."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip(), idx], seed=42, itags=["x", "y", "z"])
    original_itags = list(tensor.itags)
    
    tensor.invert([0, 2])
    
    # Tags should be unchanged
    assert list(tensor.itags) == original_itags


def test_invert_out_of_range_raises():
    """Test that out of range index positions raise error."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    
    # Position too large
    with pytest.raises(IndexError, match="out of range"):
        tensor.invert(2)
    
    # Negative position
    with pytest.raises(IndexError, match="out of range"):
        tensor.invert(-1)
    
    # In list
    with pytest.raises(IndexError, match="out of range"):
        tensor.invert([0, 5])


def test_invert_z2_group():
    """Test invert with Z2 symmetry group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_charges = tensor.indices[0].charges()
    
    tensor.invert(0)
    
    # For Z2, dual(0) = 0 and dual(1) = 1, so charges appear unchanged
    # but they were conjugated (Z2 charges are self-dual)
    assert tensor.indices[0].charges() == original_charges
    # Direction should be reversed
    assert tensor.indices[0].direction == Direction.IN


def test_invert_product_group():
    """Test invert with product group."""
    group = ProductGroup([U1Group(), U1Group()])
    # Use matching charges for both indices to ensure charge conservation
    idx1 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2), Sector((1, -1), 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector((0, 0), 2), Sector((1, -1), 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_charges = tensor.indices[1].charges()
    original_direction = tensor.indices[1].direction
    
    tensor.invert(1)
    
    # Charges should be conjugated: dual((0,0)) = (0,0), dual((1,-1)) = (-1,1)
    expected_charges = tuple(group.dual(c) for c in original_charges)
    assert tensor.indices[1].charges() == expected_charges
    # Direction should be reversed
    assert tensor.indices[1].direction == original_direction.reverse()


def test_invert_double_application():
    """Test that inverting twice returns to original direction, charges, and data."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    original_direction = tensor.indices[0].direction
    original_charges = tensor.indices[0].charges()
    original_norm = tensor.norm()
    original_keys = set(tensor.data.keys())
    original_data = {k: v.clone() for k, v in tensor.data.items()}
    
    # Invert twice
    tensor.invert(0)
    tensor.invert(0)
    
    # Should be back to original
    assert tensor.indices[0].direction == original_direction
    assert tensor.indices[0].charges() == original_charges
    # Verify norm preserved
    assert math.isclose(tensor.norm(), original_norm)
    # Verify keys unchanged
    assert set(tensor.data.keys()) == original_keys
    # Verify data values unchanged
    for key in original_data:
        assert torch.allclose(tensor.data[key], original_data[key])


# insert_index tests

def test_insert_index_at_beginning():
    """Test inserting a trivial index at the beginning."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    original_norm = tensor.norm()
    original_block_00 = tensor.data[(0, 0)].clone()
    
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
    assert torch.allclose(tensor.data[(0, 0, 0)][0], original_block_00)
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
    assert math.isclose(tensor.norm(), original_norm)


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
        original_values[key] = arr.clone()
    
    # Insert index
    tensor.insert_index(1, Direction.OUT, itag="inserted")
    
    # Verify all values preserved (just reshaped)
    for old_key, old_arr in original_values.items():
        new_key = (old_key[0], 0, old_key[1])  # Insert neutral charge
        assert new_key in tensor.data
        assert torch.allclose(tensor.data[new_key][:, 0, :], old_arr)


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
    assert math.isclose(tensor.norm(), original_norm)


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
    
    tensor = Tensor.random([idx1, idx2], seed=42, dtype=torch.complex128, itags=["a", "b"])
    
    tensor.insert_index(1, Direction.OUT)
    
    # Verify dtype preserved
    assert tensor.dtype == torch.complex128
    for arr in tensor.data.values():
        assert arr.dtype == torch.complex128


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
        assert torch.allclose(T.data[key], unmerged.data[key], rtol=1e-10, atol=1e-12)


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
        assert torch.allclose(T.data[key], unmerged.data[key], rtol=1e-10, atol=1e-12)


def test_merge_axes_preserves_dtype():
    """Test that merge_axes preserves tensor dtype."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 2), Sector(1, 1)))
    
    # Test with complex dtype
    T = Tensor.random([idx, idx.flip(), idx], seed=1, dtype=torch.complex128, itags=['a', 'b', 'c'])
    
    merged, iso_conj = merge_axes(T, [0, 1], merged_tag='ab')
    
    assert merged.dtype == torch.complex128
    # Isometry uses the tensor's dtype
    assert iso_conj.dtype == torch.complex128


# Trim zero sectors tests

def test_trim_zero_blocks_single_block():
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
        (-1, -1): torch.tensor([[1.0, 0.5], [0.3, 0.8]]),
        (0, 0): torch.tensor([[1e-20, 1e-20], [1e-20, 1e-20]]),  # Near-zero
        (1, 1): torch.tensor([[0.7, 0.2], [0.4, 0.9]]),
    }
    
    tensor = Tensor(
        indices=(idx_in, idx_out),
        itags=("in", "out"),
        data=data,
        dtype=torch.float64
    )
    
    # Apply trim
    tensor.trim_zero_blocks()
    
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


def test_trim_zero_blocks_multiple_blocks():
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
    eps = torch.finfo(torch.float64).eps
    data = {
        (-1, -1): torch.tensor([[1.0]]),
        (0, 0): torch.tensor([[eps / 2]]),  # Below threshold
        (1, 1): torch.tensor([[eps / 10]]),  # Below threshold
        (2, 2): torch.tensor([[2.0]]),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=torch.float64
    )
    
    tensor.trim_zero_blocks()
    
    # Only two blocks should remain
    assert len(tensor.data) == 2
    assert (-1, -1) in tensor.data
    assert (2, 2) in tensor.data
    assert (0, 0) not in tensor.data
    assert (1, 1) not in tensor.data
    
    # Sectors should be trimmed
    charges = [s.charge for s in tensor.indices[0].sectors]
    assert set(charges) == {-1, 2}


def test_trim_zero_blocks_no_removal():
    """Test that trim does nothing when all blocks are non-zero."""
    group = U1Group()
    idx = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=0, dim=2), Sector(charge=1, dim=2))
    )
    
    data = {
        (0, 0): torch.tensor([[1.0, 0.5], [0.3, 0.8]]),
        (1, 1): torch.tensor([[0.7, 0.2], [0.4, 0.9]]),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=torch.float64
    )
    
    # Store original state
    original_keys = set(tensor.data.keys())
    original_sectors = len(tensor.indices[0].sectors)
    
    tensor.trim_zero_blocks()
    
    # Nothing should change
    assert set(tensor.data.keys()) == original_keys
    assert len(tensor.indices[0].sectors) == original_sectors


def test_trim_zero_blocks_inplace():
    """Test that trim modifies the tensor in-place."""
    group = U1Group()
    idx = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=0, dim=1), Sector(charge=1, dim=1))
    )
    
    data = {
        (0, 0): torch.tensor([[1e-20]]),
        (1, 1): torch.tensor([[1.0]]),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=torch.float64
    )
    
    # Get object id before
    tensor_id = id(tensor)
    
    # Apply trim (returns None for in-place)
    result = tensor.trim_zero_blocks()
    
    assert result is None  # In-place methods return None
    assert id(tensor) == tensor_id  # Same object
    assert len(tensor.data) == 1  # But modified


def test_trim_zero_blocks_negative_values():
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
    
    eps = torch.finfo(torch.float64).eps
    data = {
        (-1, -1): torch.tensor([[-1.0, -0.5], [-0.3, -0.8]]),  # All negative, non-zero
        (0, 0): torch.tensor([[-eps/2, -eps/3], [-eps/4, -eps/5]]),  # All negative, near-zero
        (1, 1): torch.tensor([[0.7, 0.2], [0.4, 0.9]]),  # All positive, non-zero
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=torch.float64
    )
    
    tensor.trim_zero_blocks()
    
    # Near-zero block should be removed despite negative values
    assert len(tensor.data) == 2
    assert (-1, -1) in tensor.data
    assert (1, 1) in tensor.data
    assert (0, 0) not in tensor.data
    
    # Verify negative values are preserved
    assert torch.equal(
        tensor.data[(-1, -1)],
        torch.tensor([[-1.0, -0.5], [-0.3, -0.8]], dtype=tensor.data[(-1, -1)].dtype)
    )


def test_trim_zero_blocks_mixed_signs():
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
    
    eps = torch.finfo(torch.float64).eps
    data = {
        # Mixed signs with large magnitude - should be kept
        (-1, -1): torch.tensor([[1.5, -2.3], [-0.8, 1.2]]),
        # Mixed signs with tiny magnitude - should be removed
        (0, 0): torch.tensor([[eps/2, -eps/3], [-eps/4, eps/5]]),
        # Mixed signs with one large value - should be kept
        (1, 1): torch.tensor([[eps/2, -eps/3], [2.0, -eps/5]]),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=torch.float64
    )
    
    tensor.trim_zero_blocks()
    
    # Block with all tiny values should be removed
    # Blocks with at least one large value should be kept
    assert len(tensor.data) == 2
    assert (-1, -1) in tensor.data
    assert (1, 1) in tensor.data
    assert (0, 0) not in tensor.data
    
    # Verify mixed-sign data is preserved exactly
    assert torch.equal(
        tensor.data[(-1, -1)],
        torch.tensor([[1.5, -2.3], [-0.8, 1.2]], dtype=tensor.data[(-1, -1)].dtype)
    )
    assert torch.max(torch.abs(tensor.data[(1, 1)])).item() >= 2.0  # Has the large value


def test_trim_zero_blocks_complex_values():
    """Test trim with complex-valued data."""
    group = U1Group()
    idx = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=0, dim=2), Sector(charge=1, dim=2))
    )
    
    # Complex near-zero block (both real and imaginary parts near zero)
    eps = torch.finfo(torch.float64).eps
    data = {
        (0, 0): torch.tensor([[eps/2 + 1j*eps/3, eps/4 + 1j*eps/5],
                        [eps/6 + 1j*eps/7, eps/8 + 1j*eps/9]], dtype=torch.complex128),
        (1, 1): torch.tensor([[1.0 + 1.0j, 2.0 + 2.0j],
                        [3.0 + 3.0j, 4.0 + 4.0j]], dtype=torch.complex128),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=torch.complex128
    )
    
    tensor.trim_zero_blocks()
    
    # Near-zero complex block should be removed
    assert len(tensor.data) == 1
    assert (1, 1) in tensor.data
    assert (0, 0) not in tensor.data


def test_trim_zero_blocks_z2_symmetry():
    """Test trim with Z2 symmetry group."""
    group = Z2Group()
    idx = Index(
        direction=Direction.IN,
        group=group,
        sectors=(Sector(charge=0, dim=2), Sector(charge=1, dim=2))
    )
    
    data = {
        (0, 0): torch.tensor([[1e-20, 1e-20], [1e-20, 1e-20]]),
        (1, 1): torch.tensor([[0.5, 0.3], [0.2, 0.8]]),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=torch.float64
    )
    
    tensor.trim_zero_blocks()
    
    assert len(tensor.data) == 1
    assert (1, 1) in tensor.data
    assert (0, 0) not in tensor.data
    
    # Z2 charge 1 should remain
    charges = [s.charge for s in tensor.indices[0].sectors]
    assert charges == [1]


def test_trim_zero_blocks_product_group():
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
        ((0, 0), (0, 0)): torch.tensor([[1.0]]),
        ((0, 1), (0, 1)): torch.tensor([[1e-20]]),  # Near-zero
        ((1, 0), (1, 0)): torch.tensor([[0.5]]),
    }
    
    tensor = Tensor(
        indices=(idx, idx.flip()),
        itags=("a", "b"),
        data=data,
        dtype=torch.float64
    )
    
    tensor.trim_zero_blocks()
    
    assert len(tensor.data) == 2
    assert ((0, 1), (0, 1)) not in tensor.data
    
    charges = [s.charge for s in tensor.indices[0].sectors]
    assert (0, 0) in charges
    assert (1, 0) in charges
    assert (0, 1) not in charges


def test_trim_zero_blocks_su2_zero_data():
    """Test trim with SU(2) when reduced tensor data is zero."""
    group = SU2Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    
    # Set one block's data to near-zero
    key = (1, 1)
    if key in tensor.data:
        tensor.data[key] = torch.zeros_like(tensor.data[key]) * 1e-20
        
        original_blocks = len(tensor.data)
        tensor.trim_zero_blocks()
        
        # Block with zero data should be removed
        assert key not in tensor.data
        assert len(tensor.data) < original_blocks
        
        # Intertwiner for removed block should also be removed
        assert key not in tensor.intw


def test_trim_zero_blocks_su2_zero_weights():
    """Test trim with SU(2) when weights are zero (T = R @ 0 = 0)."""
    group = SU2Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    
    # Set one block's weights to near-zero (physical tensor becomes zero)
    key = (1, 1)
    if key in tensor.intw:
        tensor.intw[key].weights[:] = 0.0
        
        original_blocks = len(tensor.data)
        tensor.trim_zero_blocks()
        
        # Block with zero weights should be removed (even if data is non-zero)
        assert key not in tensor.data
        assert len(tensor.data) < original_blocks
        assert key not in tensor.intw


def test_trim_zero_blocks_su2_nonzero_preserved():
    """Test trim with SU(2) preserves non-zero blocks."""
    group = SU2Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    
    original_keys = set(tensor.data.keys())
    original_blocks = len(tensor.data)
    
    tensor.trim_zero_blocks()
    
    # All blocks should remain (none are zero)
    assert len(tensor.data) == original_blocks
    assert set(tensor.data.keys()) == original_keys
    
    # All intertwiners should remain
    assert set(tensor.intw.keys()) == original_keys


def test_trim_zero_blocks_su2_three_indices_mixed():
    """Test trim with SU(2) using 3 indices, some blocks zero."""
    group = SU2Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 3)))
    
    tensor = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    
    # Zero out some blocks via weights
    keys_to_zero = []
    for i, key in enumerate(tensor.data.keys()):
        if i % 2 == 0:  # Zero out every other block
            tensor.intw[key].weights[:] = 0.0
            keys_to_zero.append(key)
    
    tensor.trim_zero_blocks()
    
    # Zeroed blocks should be removed
    for key in keys_to_zero:
        assert key not in tensor.data
        assert key not in tensor.intw

