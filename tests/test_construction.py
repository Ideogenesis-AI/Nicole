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


"""Tests for Tensor construction methods."""

import numpy as np
import pytest

from nicole import Direction, Index, Sector, Tensor, U1Group, Z2Group
from nicole.symmetry.product import ProductGroup
from .utils import make_u1_index, assert_charge_neutral


def test_tensor_zeros_basic():
    """Test Tensor.zeros with basic indices."""
    group = U1Group()
    left = make_u1_index(Direction.OUT, [(0, 2), (1, 1)], group)
    right = make_u1_index(Direction.IN, [(0, 3), (1, 1)], group)

    tensor = Tensor.zeros([left, right], dtype=np.float64, itags=["L", "R"])
    
    # Allowed charge combinations: (0,0) and (1,1)
    assert set(tensor.data.keys()) == {(0, 0), (1, 1)}
    for block in tensor.data.values():
        assert block.shape in {(2, 3), (1, 1)}
        assert np.allclose(block, 0.0)
    assert_charge_neutral(tensor)


def test_tensor_zeros_two_indices():
    """Test Tensor.zeros with minimum two indices."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2), (1, 3)], group)
    
    tensor = Tensor.zeros([idx, idx.flip()], itags=["A", "B"])
    
    assert set(tensor.data.keys()) == {(0, 0), (1, 1)}
    assert tensor.data[(0, 0)].shape == (2, 2)
    assert tensor.data[(1, 1)].shape == (3, 3)


def test_tensor_zeros_no_itags():
    """Test Tensor.zeros without providing itags."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    
    tensor = Tensor.zeros([idx, idx.flip()])
    
    assert len(tensor.itags) == 2
    assert tensor.itags[0] == "_init_"
    assert tensor.itags[1] == "_init_"


def test_tensor_zeros_complex_dtype():
    """Test Tensor.zeros with complex dtype."""
    group = U1Group()
    left = make_u1_index(Direction.OUT, [(0, 2)], group)
    right = make_u1_index(Direction.IN, [(0, 3)], group)
    
    tensor = Tensor.zeros([left, right], dtype=np.complex128, itags=["L", "R"])
    
    assert tensor.dtype == np.complex128
    assert np.allclose(tensor.data[(0, 0)], 0.0 + 0.0j)


def test_tensor_zeros_z2():
    """Test Tensor.zeros with Z2 group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 3)))
    
    tensor = Tensor.zeros([idx1, idx2], itags=["A", "B"])
    
    # Neutral blocks: (0,0) and (1,1)
    assert set(tensor.data.keys()) == {(0, 0), (1, 1)}


def test_tensor_random_basic():
    """Test Tensor.random with basic indices."""
    group = U1Group()
    idx_a = make_u1_index(Direction.OUT, [(0, 2), (1, 1)], group)
    idx_b = make_u1_index(Direction.IN, [(0, 2), (1, 1)], group)
    idx_c = make_u1_index(Direction.OUT, [(0, 1), (-1, 2)], group)

    tensor = Tensor.random([idx_a, idx_b, idx_c], seed=2024, itags=["A", "B", "C"])
    
    assert tensor.data, "random tensor should have at least one block"
    assert_charge_neutral(tensor)


def test_tensor_random_seed_reproducible():
    """Test that Tensor.random with same seed gives same result."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 3)], group)
    
    t1 = Tensor.random([idx, idx.flip()], seed=42, itags=["A", "B"])
    t2 = Tensor.random([idx, idx.flip()], seed=42, itags=["A", "B"])
    
    np.testing.assert_allclose(t1.data[(0, 0)], t2.data[(0, 0)])


def test_tensor_random_different_seeds():
    """Test that Tensor.random with different seeds gives different results."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 3)], group)
    
    t1 = Tensor.random([idx, idx.flip()], seed=42, itags=["A", "B"])
    t2 = Tensor.random([idx, idx.flip()], seed=99, itags=["A", "B"])
    
    assert not np.allclose(t1.data[(0, 0)], t2.data[(0, 0)])


def test_tensor_random_complex():
    """Test Tensor.random with complex dtype."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 3)], group)
    
    tensor = Tensor.random([idx, idx.flip()], dtype=np.complex128, seed=123, itags=["A", "B"])
    
    assert tensor.dtype == np.complex128
    assert np.iscomplexobj(tensor.data[(0, 0)])


def test_tensor_random_no_itags():
    """Test Tensor.random without itags."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    
    tensor = Tensor.random([idx, idx.flip()], seed=1)
    
    assert tensor.itags[0] == "_init_"
    assert tensor.itags[1] == "_init_"


def test_tensor_norm_matches_manual():
    """Test that Tensor.norm() matches manual computation."""
    idx = make_u1_index(Direction.OUT, [(0, 3), (1, 2)])
    tensor = Tensor.random([idx, idx.flip()], seed=11, itags=["A", "B"])
    manual = np.sqrt(sum(np.sum(np.abs(block) ** 2) for block in tensor.data.values()))
    assert np.isclose(tensor.norm(), manual)


def test_tensor_norm_zero():
    """Test Tensor.norm() for zero tensor."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 3)], group)
    
    tensor = Tensor.zeros([idx, idx.flip()], itags=["A", "B"])
    
    assert tensor.norm() == 0.0


def test_tensor_norm_empty():
    """Test Tensor.norm() for tensor with no blocks."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=())
    
    tensor = Tensor(indices=(idx, idx.flip()), itags=("A", "B"), data={}, dtype=np.float64)
    
    assert tensor.norm() == 0.0


def test_tensor_validation_mismatched_itags():
    """Test that Tensor rejects mismatched itag count."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    
    with pytest.raises(ValueError, match="must match number of indices"):
        Tensor(indices=(idx, idx.flip()), itags=("A", "B", "C"), data={}, dtype=np.float64)


def test_tensor_validation_invalid_block_shape():
    """Test that Tensor rejects invalid block shapes."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    
    blocks = {(0, 0): np.zeros((3, 2))}  # Wrong shape, should be (2, 2)
    
    with pytest.raises(ValueError, match="expected"):
        Tensor(indices=(idx, idx.flip()), itags=("A", "B"), data=blocks, dtype=np.float64)


def test_tensor_validation_charge_violation():
    """Test that Tensor rejects non-conserving blocks."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 1)))
    
    # Block (1, 0) violates charge conservation
    blocks = {(1, 0): np.zeros((1, 3))}
    
    with pytest.raises(ValueError, match="violates charge conservation"):
        Tensor(indices=(idx1, idx2), itags=("A", "B"), data=blocks, dtype=np.float64)


def test_tensor_validation_requires_two_indices():
    """Test that Tensor rejects tensors with fewer than 2 indices."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    
    # Test with Tensor constructor
    with pytest.raises(ValueError, match="must have at least 2 indices"):
        Tensor(indices=(idx,), itags=("A",), data={}, dtype=np.float64)
    
    # Test with Tensor.zeros
    with pytest.raises(ValueError, match="must have at least 2 indices"):
        Tensor.zeros([idx], itags=["A"])
    
    # Test with Tensor.random
    with pytest.raises(ValueError, match="must have at least 2 indices"):
        Tensor.random([idx], seed=1, itags=["A"])


def test_tensor_str_repr():
    """Test Tensor.__str__ and __repr__ methods."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    
    tensor = Tensor.zeros([idx, idx.flip()], itags=["A", "B"])
    
    string_repr = str(tensor)
    assert "Tensor" in string_repr
    assert "A" in string_repr
    
    # __repr__ should be same as __str__
    assert repr(tensor) == str(tensor)


def test_tensor_construction_float32():
    """Test Tensor construction with float32 dtype."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    
    tensor = Tensor.zeros([idx, idx.flip()], dtype=np.float32, itags=["A", "B"])
    
    assert tensor.dtype == np.float32
    assert tensor.data[(0, 0)].dtype == np.float32


def test_tensor_construction_complex64():
    """Test Tensor construction with complex64 dtype."""
    group = U1Group()
    idx = make_u1_index(Direction.OUT, [(0, 2)], group)
    
    tensor = Tensor.random([idx, idx.flip()], dtype=np.complex64, seed=1, itags=["A", "B"])
    
    assert tensor.dtype == np.complex64
    assert tensor.data[(0, 0)].dtype == np.complex64


def test_tensor_zeros_three_indices():
    """Test Tensor.zeros with three indices."""
    group = U1Group()
    idx1 = make_u1_index(Direction.OUT, [(0, 2), (1, 1)], group)
    idx2 = make_u1_index(Direction.IN, [(0, 1), (-1, 2)], group)
    idx3 = make_u1_index(Direction.OUT, [(0, 3), (-1, 1)], group)
    
    tensor = Tensor.zeros([idx1, idx2, idx3], itags=["A", "B", "C"])
    
    assert_charge_neutral(tensor)
    for block in tensor.data.values():
        assert np.allclose(block, 0.0)


# ProductGroup integration tests

def test_tensor_zeros_product_group_u1_u1():
    """Test Tensor.zeros with U1×U1 ProductGroup."""
    group = ProductGroup([U1Group(), U1Group()])
    
    # Create indices with tuple charges
    left = Index(
        Direction.OUT,
        group,
        sectors=(
            Sector((0, 0), 2),
            Sector((1, 0), 1),
            Sector((0, 1), 1),
        )
    )
    right = Index(
        Direction.IN,
        group,
        sectors=(
            Sector((0, 0), 3),
            Sector((1, 0), 1),
            Sector((0, 1), 2),
        )
    )
    
    tensor = Tensor.zeros([left, right], itags=["L", "R"])
    
    # Charge conservation: OUT charges equal IN charges
    # Valid blocks: ((0,0), (0,0)), ((1,0), (1,0)), ((0,1), (0,1))
    assert set(tensor.data.keys()) == {((0, 0), (0, 0)), ((1, 0), (1, 0)), ((0, 1), (0, 1))}
    
    assert tensor.data[((0, 0), (0, 0))].shape == (2, 3)
    assert tensor.data[((1, 0), (1, 0))].shape == (1, 1)
    assert tensor.data[((0, 1), (0, 1))].shape == (1, 2)
    
    for block in tensor.data.values():
        assert np.allclose(block, 0.0)


def test_tensor_zeros_product_group_u1_z2():
    """Test Tensor.zeros with U1×Z2 ProductGroup."""
    group = ProductGroup([U1Group(), Z2Group()])
    
    left = Index(
        Direction.OUT,
        group,
        sectors=(
            Sector((0, 0), 2),
            Sector((1, 1), 1),
        )
    )
    right = Index(
        Direction.IN,
        group,
        sectors=(
            Sector((0, 0), 1),
            Sector((1, 1), 2),
        )
    )
    
    tensor = Tensor.zeros([left, right], itags=["L", "R"])
    
    assert set(tensor.data.keys()) == {((0, 0), (0, 0)), ((1, 1), (1, 1))}
    assert tensor.data[((0, 0), (0, 0))].shape == (2, 1)
    assert tensor.data[((1, 1), (1, 1))].shape == (1, 2)


def test_tensor_random_product_group():
    """Test Tensor.random with ProductGroup."""
    group = ProductGroup([U1Group(), U1Group()])
    
    left = Index(
        Direction.OUT,
        group,
        sectors=(Sector((0, 0), 2), Sector((1, -1), 1))
    )
    right = Index(
        Direction.IN,
        group,
        sectors=(Sector((0, 0), 3), Sector((1, -1), 2))
    )
    
    tensor = Tensor.random([left, right], seed=42, itags=["L", "R"])
    
    assert set(tensor.data.keys()) == {((0, 0), (0, 0)), ((1, -1), (1, -1))}
    assert tensor.data[((0, 0), (0, 0))].shape == (2, 3)
    assert tensor.data[((1, -1), (1, -1))].shape == (1, 2)
    
    # Check that blocks are not all zeros
    assert not np.allclose(tensor.data[((0, 0), (0, 0))], 0.0)
    assert tensor.norm() > 0.0

