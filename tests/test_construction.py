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


"""Tests for Tensor construction methods."""

import numpy as np
import pytest

from nicole import Direction, Index, Sector, Tensor, U1Group, Z2Group
from nicole.blocks import BlockSchema
from nicole.symmetry.product import ProductGroup
from .utils import assert_charge_neutral


def test_tensor_zeros_basic():
    """Test Tensor.zeros with basic indices."""
    group = U1Group()
    left = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    right = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 1)))

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
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.zeros([idx, idx.flip()], itags=["A", "B"])
    
    assert set(tensor.data.keys()) == {(0, 0), (1, 1)}
    assert tensor.data[(0, 0)].shape == (2, 2)
    assert tensor.data[(1, 1)].shape == (3, 3)


def test_tensor_zeros_no_itags():
    """Test Tensor.zeros without providing itags."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.zeros([idx, idx.flip()])
    
    assert len(tensor.itags) == 2
    assert tensor.itags[0] == "_init_"
    assert tensor.itags[1] == "_init_"


def test_tensor_zeros_complex_dtype():
    """Test Tensor.zeros with complex dtype."""
    group = U1Group()
    left = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    right = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    
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
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_c = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(-1, 2)))

    tensor = Tensor.random([idx_a, idx_b, idx_c], seed=2024, itags=["A", "B", "C"])
    
    assert tensor.data, "random tensor should have at least one block"
    assert_charge_neutral(tensor)


def test_tensor_random_seed_reproducible():
    """Test that Tensor.random with same seed gives same result."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    
    t1 = Tensor.random([idx, idx.flip()], seed=42, itags=["A", "B"])
    t2 = Tensor.random([idx, idx.flip()], seed=42, itags=["A", "B"])
    
    np.testing.assert_allclose(t1.data[(0, 0)], t2.data[(0, 0)])


def test_tensor_random_different_seeds():
    """Test that Tensor.random with different seeds gives different results."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    
    t1 = Tensor.random([idx, idx.flip()], seed=42, itags=["A", "B"])
    t2 = Tensor.random([idx, idx.flip()], seed=99, itags=["A", "B"])
    
    assert not np.allclose(t1.data[(0, 0)], t2.data[(0, 0)])


def test_tensor_random_complex():
    """Test Tensor.random with complex dtype."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    
    tensor = Tensor.random([idx, idx.flip()], dtype=np.complex128, seed=123, itags=["A", "B"])
    
    assert tensor.dtype == np.complex128
    assert np.iscomplexobj(tensor.data[(0, 0)])


def test_tensor_random_no_itags():
    """Test Tensor.random without itags."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], seed=1)
    
    assert tensor.itags[0] == "_init_"
    assert tensor.itags[1] == "_init_"


def test_tensor_norm_matches_manual():
    """Test that Tensor.norm() matches manual computation."""
    idx = Index(Direction.OUT, U1Group(), sectors=(Sector(0, 3), Sector(1, 2)))
    tensor = Tensor.random([idx, idx.flip()], seed=11, itags=["A", "B"])
    manual = np.sqrt(sum(np.sum(np.abs(block) ** 2) for block in tensor.data.values()))
    assert np.isclose(tensor.norm(), manual)


def test_tensor_norm_zero():
    """Test Tensor.norm() for zero tensor."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    
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
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    with pytest.raises(ValueError, match="must match number of indices"):
        Tensor(indices=(idx, idx.flip()), itags=("A", "B", "C"), data={}, dtype=np.float64)


def test_tensor_validation_invalid_block_shape():
    """Test that Tensor rejects invalid block shapes."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
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


def test_tensor_validation_rejects_single_index():
    """Test that Tensor rejects tensors with exactly 1 index."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    # Exactly 1 index should raise error
    with pytest.raises(ValueError, match="exactly 1 index"):
        Tensor(indices=(idx,), itags=("A",), data={}, dtype=np.float64)
    
    with pytest.raises(ValueError, match="exactly 1 index"):
        Tensor.zeros([idx], itags=["A"])
    
    with pytest.raises(ValueError, match="exactly 1 index"):
        Tensor.random([idx], seed=1, itags=["A"])


# Scalar (0D tensor) tests

def test_tensor_scalar_creation():
    """Test creating scalars (0D tensors) with from_scalar."""
    # Test with integer
    s_int = Tensor.from_scalar(42)
    assert s_int.is_scalar()
    assert s_int.item() == 42
    assert len(s_int.indices) == 0
    assert len(s_int.itags) == 0
    
    # Test with float
    s_float = Tensor.from_scalar(3.14, dtype=np.float64)
    assert s_float.is_scalar()
    assert np.isclose(s_float.item(), 3.14)
    
    # Test with complex
    s_complex = Tensor.from_scalar(1 + 2j, dtype=np.complex128)
    assert s_complex.is_scalar()
    assert s_complex.item() == 1 + 2j
    
    # Test with custom label
    s_labeled = Tensor.from_scalar(5.0, label="MyScalar")
    assert s_labeled.label == "MyScalar"


def test_tensor_scalar_operations():
    """Test arithmetic operations with scalars."""
    s1 = Tensor.from_scalar(2.0)
    s2 = Tensor.from_scalar(3.0)
    
    # Scalar addition
    s3 = s1 + s2
    assert s3.is_scalar()
    assert np.isclose(s3.item(), 5.0)
    
    # Scalar multiplication
    s4 = s1 * 2.5
    assert s4.is_scalar()
    assert np.isclose(s4.item(), 5.0)
    
    # Left scalar multiplication
    s5 = 1.5 * s1
    assert s5.is_scalar()
    assert np.isclose(s5.item(), 3.0)
    
    # Scalar subtraction
    s6 = s2 - s1
    assert s6.is_scalar()
    assert np.isclose(s6.item(), 1.0)


def test_tensor_scalar_norm():
    """Test norm of scalar tensors."""
    s = Tensor.from_scalar(3.0)
    assert np.isclose(s.norm(), 3.0)
    
    s_negative = Tensor.from_scalar(-4.0)
    assert np.isclose(s_negative.norm(), 4.0)
    
    s_complex = Tensor.from_scalar(3 + 4j, dtype=np.complex128)
    assert np.isclose(s_complex.norm(), 5.0)  # |3+4j| = 5


def test_tensor_scalar_copy():
    """Test copying scalar tensors."""
    s = Tensor.from_scalar(42.0)
    s_copy = s.copy()
    
    assert s_copy.is_scalar()
    assert s_copy.item() == s.item()
    assert s_copy.data[()] is not s.data[()]  # Different array objects


def test_tensor_scalar_display():
    """Test string representation of scalar tensors."""
    s = Tensor.from_scalar(3.14)
    str_repr = str(s)
    
    assert "0-D" in str_repr
    assert "3.14" in str_repr
    assert "0x { 1 x 0 }" in str_repr
    assert repr(s) == str(s)


def test_tensor_item_raises_on_non_scalar():
    """Test that item() raises error on non-scalar tensors."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    tensor = Tensor.random([idx, idx.flip()], seed=1, itags=["a", "b"])
    
    with pytest.raises(ValueError, match="can only be called on scalars"):
        tensor.item()


def test_tensor_scalar_validation():
    """Test scalar-specific validation."""
    # Scalar must have empty key
    with pytest.raises(ValueError, match="empty tuple"):
        Tensor(indices=(), itags=(), data={(0,): np.array(1.0)}, dtype=np.float64)
    
    # Scalar can only have one block
    with pytest.raises(ValueError, match="only have one block"):
        Tensor(indices=(), itags=(), data={(): np.array(1.0), (1,): np.array(2.0)}, dtype=np.float64)


def test_tensor_str_repr():
    """Test Tensor.__str__ and __repr__ methods."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.zeros([idx, idx.flip()], itags=["A", "B"])
    
    string_repr = str(tensor)
    assert "Tensor" in string_repr
    assert "A" in string_repr
    
    # __repr__ should be same as __str__
    assert repr(tensor) == str(tensor)


def test_tensor_construction_float32():
    """Test Tensor construction with float32 dtype."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.zeros([idx, idx.flip()], dtype=np.float32, itags=["A", "B"])
    
    assert tensor.dtype == np.float32
    assert tensor.data[(0, 0)].dtype == np.float32


def test_tensor_construction_complex64():
    """Test Tensor construction with complex64 dtype."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], dtype=np.complex64, seed=1, itags=["A", "B"])
    
    assert tensor.dtype == np.complex64
    assert tensor.data[(0, 0)].dtype == np.complex64


def test_tensor_zeros_three_indices():
    """Test Tensor.zeros with three indices."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(-1, 1)))
    
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


# ============================================================================
# Sector pruning tests
# ============================================================================

def test_zeros_prunes_unused_sectors():
    """Test that Tensor.zeros removes sectors that don't appear in any charge-conserving block."""
    group = U1Group()
    
    # Create indices where charge 2 can't be conserved with charge 0
    # OUT(2) + IN(0) = 2 - 0 = 2 (not conserved)
    # OUT(0) + IN(0) = 0 - 0 = 0 (conserved)
    # OUT(1) + IN(1) = 1 - 1 = 0 (conserved)
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    tensor = Tensor.zeros([idx_out, idx_in], itags=["a", "b"])
    
    # Verify that charge 2 was pruned from the first index
    assert len(tensor.indices[0].sectors) == 2  # Only 0 and 1 should remain
    charges_out = tensor.indices[0].charges()
    assert 0 in charges_out
    assert 1 in charges_out
    assert 2 not in charges_out  # This sector was pruned
    
    # Verify all charges in block keys match sectors in indices
    for block_key in tensor.data.keys():
        for axis, charge in enumerate(block_key):
            assert charge in tensor.indices[axis].charges(), \
                f"Charge {charge} in block key not found in index {axis} sectors"


def test_random_prunes_unused_sectors():
    """Test that Tensor.random removes sectors that don't appear in any charge-conserving block."""
    group = U1Group()
    
    # Create a scenario where multiple sectors won't be used
    # OUT(0) + OUT(1) + IN(0) = 0 + 1 - 0 = 1 (not conserved)
    # OUT(0) + OUT(0) + IN(0) = 0 + 0 - 0 = 0 (conserved)
    # OUT(1) + OUT(-1) + IN(0) = 1 + (-1) - 0 = 0 (conserved)
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 2), Sector(-2, 2)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    tensor = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    
    # Verify that sectors were pruned
    # Not all original sectors should remain
    assert len(tensor.indices[0].sectors) <= 3
    assert len(tensor.indices[1].sectors) <= 3
    assert len(tensor.indices[2].sectors) <= 2
    
    # Verify all charges in block keys match sectors in indices
    for block_key in tensor.data.keys():
        for axis, charge in enumerate(block_key):
            assert charge in tensor.indices[axis].charges(), \
                f"Charge {charge} at axis {axis} in block key not found in index sectors"


def test_zeros_random_consistency():
    """Test that zeros and random produce tensors with matching index structures."""
    group = U1Group()
    
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(2, 2), Sector(3, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2), Sector(-1, 2)))
    
    zeros_tensor = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    random_tensor = Tensor.random([idx1, idx2], seed=123, itags=["a", "b"])
    
    # Both should have pruned to the same index structure
    assert len(zeros_tensor.indices) == len(random_tensor.indices)
    for i in range(len(zeros_tensor.indices)):
        assert len(zeros_tensor.indices[i].sectors) == len(random_tensor.indices[i].sectors)
        assert set(zeros_tensor.indices[i].charges()) == set(random_tensor.indices[i].charges())
    
    # Both should have the same block keys
    assert set(zeros_tensor.data.keys()) == set(random_tensor.data.keys())


def test_prune_with_no_conserving_blocks():
    """Test pruning when no blocks satisfy charge conservation."""
    group = U1Group()
    
    # Create indices where no combination conserves charge
    # OUT(1) + OUT(1) + IN(0) = 1 + 1 - 0 = 2 (not conserved)
    # All combinations will fail
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.zeros([idx1, idx2, idx3], itags=["a", "b", "c"])
    
    # Should have no blocks
    assert len(tensor.data) == 0
    
    # All indices should be pruned to empty
    for idx in tensor.indices:
        assert len(idx.sectors) == 0


def test_prune_preserves_all_used_sectors():
    """Test that pruning doesn't remove sectors that ARE used."""
    group = U1Group()
    
    # Create indices where all sectors participate
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    tensor = Tensor.random([idx_out, idx_in], seed=99, itags=["a", "b"])
    
    # All sectors should be preserved (all can be conserved)
    assert len(tensor.indices[0].sectors) == 2
    assert len(tensor.indices[1].sectors) == 2
    assert set(tensor.indices[0].charges()) == {0, 1}
    assert set(tensor.indices[1].charges()) == {0, 1}


def test_prune_with_complex_charge_structure():
    """Test pruning with complex multi-index tensors."""
    group = U1Group()
    
    # 4-index tensor with various charge combinations
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2), Sector(3, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 2), Sector(-2, 2)))
    idx4 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    
    tensor = Tensor.random([idx1, idx2, idx3, idx4], seed=777, itags=["a", "b", "c", "d"])
    
    # Verify consistency: all block charges match index sectors
    for block_key in tensor.data.keys():
        assert len(block_key) == 4
        for axis, charge in enumerate(block_key):
            index_charges = tensor.indices[axis].charges()
            assert charge in index_charges, \
                f"Block charge {charge} at axis {axis} not in index: {index_charges}"
    
    # Verify charge conservation for all blocks
    for block_key in tensor.data.keys():
        assert BlockSchema.charges_conserved(tensor.indices, block_key)


def test_prune_maintains_block_validity():
    """Test that after pruning, all blocks remain valid."""
    group = U1Group()
    
    # Create indices with many sectors, some of which won't be used
    idx1 = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 2), Sector(2, 2), Sector(3, 2), Sector(4, 2)
    ))
    idx2 = Index(Direction.IN, group, sectors=(
        Sector(0, 2), Sector(1, 2), Sector(2, 2)
    ))
    idx3 = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(-1, 2), Sector(-2, 2)
    ))
    
    tensor = Tensor.random([idx1, idx2, idx3], seed=888, itags=["a", "b", "c"])
    
    # Verify all blocks have correct shapes
    for block_key, block in tensor.data.items():
        expected_shape = BlockSchema.shape_for_key(tensor.indices, block_key)
        assert block.shape == expected_shape, \
            f"Block {block_key} has shape {block.shape}, expected {expected_shape}"
    
    # Verify charge conservation
    for block_key in tensor.data.keys():
        assert BlockSchema.charges_conserved(tensor.indices, block_key)


def test_prune_with_single_sector_indices():
    """Test pruning when indices have only one sector."""
    group = U1Group()
    
    # Single sector per index
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 3),))
    
    tensor = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    
    # Should have one block
    assert len(tensor.data) == 1
    assert (1, 1) in tensor.data
    
    # Indices should still have their sectors
    assert len(tensor.indices[0].sectors) == 1
    assert len(tensor.indices[1].sectors) == 1


def test_prune_with_negative_charges():
    """Test pruning works correctly with negative charges."""
    group = U1Group()
    
    # Mix of positive and negative charges
    idx1 = Index(Direction.OUT, group, sectors=(
        Sector(-2, 2), Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 2)
    ))
    idx2 = Index(Direction.IN, group, sectors=(
        Sector(-2, 2), Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 2)
    ))
    
    tensor = Tensor.random([idx1, idx2], seed=999, itags=["a", "b"])
    
    # All sectors should be preserved (all can pair to conserve charge)
    assert len(tensor.indices[0].sectors) == 5
    assert len(tensor.indices[1].sectors) == 5
    
    # Verify blocks exist for all charge combinations
    assert len(tensor.data) == 5  # (-2,-2), (-1,-1), (0,0), (1,1), (2,2)


def test_rand_fill_basic():
    """Test rand_fill fills zero tensor with random values."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    
    # Verify initially zeros
    for block in tensor.data.values():
        assert np.allclose(block, 0.0)
    
    # Fill with random values
    tensor.rand_fill(seed=42)
    
    # Verify no longer all zeros
    for block in tensor.data.values():
        assert not np.allclose(block, 0.0)
    
    # Verify structure preserved
    assert len(tensor.data) == 2
    assert set(tensor.data.keys()) == {(0, 0), (1, 1)}
    assert tensor.data[(0, 0)].shape == (3, 2)
    assert tensor.data[(1, 1)].shape == (2, 3)


def test_rand_fill_reproducible():
    """Test rand_fill with seed produces reproducible results."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor1 = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    tensor2 = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    
    # Fill both with same seed
    tensor1.rand_fill(seed=123)
    tensor2.rand_fill(seed=123)
    
    # Should be identical
    for key in tensor1.data:
        assert np.allclose(tensor1.data[key], tensor2.data[key])


def test_rand_fill_different_seeds():
    """Test rand_fill with different seeds produces different results."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor1 = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    tensor2 = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    
    # Fill with different seeds
    tensor1.rand_fill(seed=1)
    tensor2.rand_fill(seed=2)
    
    # Should be different
    for key in tensor1.data:
        assert not np.allclose(tensor1.data[key], tensor2.data[key])


def test_rand_fill_complex_dtype():
    """Test rand_fill with complex dtype fills with complex values."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    
    tensor = Tensor.zeros([idx1, idx2], dtype=np.complex128, itags=["a", "b"])
    tensor.rand_fill(seed=42)
    
    block = tensor.data[(0, 0)]
    
    # Verify it's complex
    assert np.iscomplexobj(block)
    
    # Verify both real and imaginary parts are non-zero
    assert not np.allclose(block.real, 0.0)
    assert not np.allclose(block.imag, 0.0)


def test_rand_fill_inplace():
    """Test rand_fill modifies tensor in-place."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    tensor = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    original_id = id(tensor)
    original_data_id = id(tensor.data)
    
    # Fill in-place
    result = tensor.rand_fill(seed=42)
    
    # Should return None (in-place operation)
    assert result is None
    
    # Tensor object should be the same
    assert id(tensor) == original_id
    assert id(tensor.data) == original_data_id
    
    # But data should be modified
    for block in tensor.data.values():
        assert not np.allclose(block, 0.0)


def test_rand_fill_preserves_metadata():
    """Test rand_fill preserves tensor metadata."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    tensor = Tensor.zeros([idx1, idx2], dtype=np.float64, itags=["left", "right"])
    tensor.label = "TestTensor"
    
    original_indices = tensor.indices
    original_itags = tensor.itags
    original_dtype = tensor.dtype
    original_label = tensor.label
    
    tensor.rand_fill(seed=42)
    
    # All metadata should be preserved
    assert tensor.indices == original_indices
    assert tensor.itags == original_itags
    assert tensor.dtype == original_dtype
    assert tensor.label == original_label


def test_rand_fill_multiple_times():
    """Test rand_fill can be called multiple times."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    
    # First fill
    tensor.rand_fill(seed=1)
    first_values = tensor.data[(0, 0)].copy()
    
    # Second fill with different seed
    tensor.rand_fill(seed=2)
    second_values = tensor.data[(0, 0)]
    
    # Values should be different
    assert not np.allclose(first_values, second_values)


def test_rand_fill_z2_group():
    """Test rand_fill works with Z2 group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    tensor.rand_fill(seed=42)
    
    # Verify structure preserved
    assert set(tensor.data.keys()) == {(0, 0), (1, 1)}
    
    # Verify filled with non-zero values
    for block in tensor.data.values():
        assert not np.allclose(block, 0.0)


def test_rand_fill_product_group():
    """Test rand_fill works with product group."""
    group = ProductGroup([U1Group(), Z2Group()])
    
    idx1 = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    idx2 = Index(Direction.IN, group, sectors=(
        Sector((0, 0), 2), Sector((1, 1), 3)
    ))
    
    tensor = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    tensor.rand_fill(seed=42)
    
    # Verify structure preserved
    assert len(tensor.data) == 2
    
    # Verify filled with non-zero values
    for block in tensor.data.values():
        assert not np.allclose(block, 0.0)


def test_rand_fill_different_dtypes():
    """Test rand_fill works with different numpy dtypes."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    
    for dtype in dtypes:
        tensor = Tensor.zeros([idx1, idx2], dtype=dtype, itags=["a", "b"])
        tensor.rand_fill(seed=42)
        
        block = tensor.data[(0, 0)]
        assert block.dtype == dtype
        assert not np.allclose(block, 0.0)


def test_rand_fill_scalar_tensor():
    """Test rand_fill works with scalar tensor."""
    tensor = Tensor.from_scalar(0.0, dtype=np.float64)
    
    # Fill with random value
    tensor.rand_fill(seed=42)
    
    # Verify it's no longer zero
    assert tensor.item() != 0.0


def test_rand_fill_empty_tensor():
    """Test rand_fill on tensor with no blocks does nothing."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    
    # These indices can't form charge-neutral blocks
    tensor = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    
    # Should have no blocks
    assert len(tensor.data) == 0
    
    # rand_fill should do nothing (no error)
    tensor.rand_fill(seed=42)
    
    # Still no blocks
    assert len(tensor.data) == 0


def test_mixed_groups_raises_error():
    """Test that tensors with indices from different groups raise an error."""
    u1_group = U1Group()
    z2_group = Z2Group()
    
    idx1 = Index(Direction.OUT, u1_group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, z2_group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    with pytest.raises(ValueError, match="All indices must share the same symmetry group"):
        Tensor.zeros([idx1, idx2], itags=["a", "b"])


def test_mixed_groups_random_raises_error():
    """Test that random tensor with mixed groups raises an error."""
    u1_group = U1Group()
    z2_group = Z2Group()
    
    idx1 = Index(Direction.OUT, u1_group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, z2_group, sectors=(Sector(0, 2),))
    
    with pytest.raises(ValueError, match="All indices must share the same symmetry group"):
        Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])


def test_mixed_groups_direct_construction_raises_error():
    """Test that direct tensor construction with mixed groups raises an error."""
    u1_group = U1Group()
    z2_group = Z2Group()
    
    idx1 = Index(Direction.OUT, u1_group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, z2_group, sectors=(Sector(0, 2),))
    
    import numpy as np
    data = {(0, 0): np.zeros((2, 2))}
    
    with pytest.raises(ValueError, match="All indices must share the same symmetry group"):
        Tensor(indices=(idx1, idx2), itags=("a", "b"), data=data)


def test_same_group_type_different_instances_works():
    """Test that indices with the same group type but different instances work."""
    u1_group1 = U1Group()
    u1_group2 = U1Group()
    
    idx1 = Index(Direction.OUT, u1_group1, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, u1_group2, sectors=(Sector(0, 2), Sector(1, 3)))
    
    # Should work - same group type
    tensor = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    assert len(tensor.data) == 2


def test_product_group_consistency():
    """Test that all indices must use the same product group."""
    group1 = ProductGroup([U1Group(), U1Group()])
    group2 = ProductGroup([U1Group(), Z2Group()])
    
    idx1 = Index(Direction.OUT, group1, sectors=(Sector((0, 0), 2),))
    idx2 = Index(Direction.IN, group2, sectors=(Sector((0, 0), 2),))
    
    with pytest.raises(ValueError, match="All indices must share the same symmetry group"):
        Tensor.zeros([idx1, idx2], itags=["a", "b"])


def test_group_property():
    """Test that group property returns the correct symmetry group."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    
    assert tensor.group == group
    assert tensor.group.name == "U1"


def test_group_property_z2():
    """Test group property with Z2 group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    
    assert tensor.group == group
    assert tensor.group.name == "Z2"


def test_group_property_product_group():
    """Test group property with product group."""
    group = ProductGroup([U1Group(), Z2Group()])
    idx1 = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector((0, 0), 2),))
    
    tensor = Tensor.zeros([idx1, idx2], itags=["a", "b"])
    
    assert tensor.group == group
    assert tensor.group.name == "U1×Z2"


def test_group_property_scalar_raises():
    """Test that accessing group property on scalar tensor raises error."""
    tensor = Tensor.from_scalar(5.0)
    
    with pytest.raises(ValueError, match="Scalar tensor has no symmetry group"):
        _ = tensor.group

