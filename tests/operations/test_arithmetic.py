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


"""Tests for Tensor arithmetic operations."""

import math
import torch
import pytest

from nicole import Direction, Index, Sector, Tensor, U1Group, Z2Group, SU2Group
from nicole.symmetry.product import ProductGroup
import nicole.symmetry.delegate as dg
from ..utils import assert_blocks_equal


# Addition tests

def test_addition_simple():
    """Test simple tensor addition."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1))),
        Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 1))),
    ]
    itags = ["L", "R"]
    
    A = Tensor.random(indices, seed=1, itags=itags)
    B = Tensor.random(indices, seed=2, itags=itags)
    
    C = A + B
    
    for key in C.data:
        expected = A.data[key] + B.data[key]
        assert torch.allclose(C.data[key], expected)


def test_addition_multi_tensor():
    """Test addition of multiple tensors."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1))),
        Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(2, 1))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
    ]
    itags = ["L", "M", "R"]

    A = Tensor.random(indices, seed=1, itags=itags)
    B = Tensor.random(indices, seed=2, itags=itags)
    C = Tensor.random(indices, seed=3, itags=itags)

    sum_tensor = A + B + C
    manual_data = {}
    for key in sum_tensor.data:
        manual_data[key] = A.data[key] + B.data[key] + C.data[key]
        assert torch.allclose(sum_tensor.data[key], manual_data[key])


def test_addition_zero_tensor():
    """Test adding zero tensor."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    Z = Tensor.zeros([idx, idx.flip()], itags=["A", "B"])
    
    result = A + Z
    assert_blocks_equal(result, A)


def test_addition_commutative():
    """Test that addition is commutative."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    B = Tensor.random([idx, idx.flip()], seed=2, itags=["A", "B"])
    
    assert_blocks_equal(A + B, B + A)


def test_addition_associative():
    """Test that addition is associative."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    B = Tensor.random([idx, idx.flip()], seed=2, itags=["A", "B"])
    C = Tensor.random([idx, idx.flip()], seed=3, itags=["A", "B"])
    
    assert_blocks_equal((A + B) + C, A + (B + C))


def test_addition_requires_matching_structure():
    """Test that addition requires matching structure."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1),))
    tensor = Tensor.random([idx_a, idx_b], seed=7, itags=["A", "B"])

    # Different sector structure on B
    idx_b_mismatch = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    tensor_mismatch = Tensor.random([idx_a, idx_b_mismatch], seed=9, itags=["A", "B"])

    with pytest.raises(ValueError, match="Sector with charge .* has dimension"):
        _ = tensor + tensor_mismatch


def test_addition_requires_matching_directions():
    """Test that addition requires matching directions."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_flipped = idx.flip()
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx2], seed=1, itags=["A", "B"])
    B = Tensor.random([idx_flipped, idx2], seed=2, itags=["A", "B"])
    
    with pytest.raises(ValueError, match="directions must match"):
        _ = A + B


def test_addition_requires_matching_order():
    """Test that addition requires matching tensor order."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    B = Tensor.random([idx, idx.flip(), idx], seed=2, itags=["A", "B", "C"])
    
    with pytest.raises(ValueError, match="different order"):
        _ = A + B


# Subtraction tests

def test_subtraction_simple():
    """Test simple tensor subtraction."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    B = Tensor.random([idx, idx.flip()], seed=2, itags=["A", "B"])
    
    C = A - B
    
    for key in C.data:
        expected = A.data[key] - B.data[key]
        assert torch.allclose(C.data[key], expected)


def test_subtraction_self_gives_zero():
    """Test that A - A gives zero."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    
    result = A - A
    
    for block in result.data.values():
        assert torch.allclose(block, torch.zeros_like(block))


def test_subtraction_inverse_of_addition():
    """Test that subtraction is inverse of addition."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1))),
        Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(2, 1))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
    ]
    itags = ["L", "M", "R"]

    A = Tensor.random(indices, seed=1, itags=itags)
    B = Tensor.random(indices, seed=2, itags=itags)
    C = Tensor.random(indices, seed=3, itags=itags)

    sum_tensor = A + B + C
    restored = sum_tensor - B - C
    assert_blocks_equal(restored, A)


# Scalar multiplication tests

def test_scalar_multiplication_int():
    """Test scalar multiplication with integer."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    
    result = A * 3
    
    for key in A.data:
        assert torch.allclose(result.data[key], A.data[key] * 3)


def test_scalar_multiplication_float():
    """Test scalar multiplication with float."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    
    result = A * 2.5
    
    for key in A.data:
        assert torch.allclose(result.data[key], A.data[key] * 2.5)


def test_scalar_multiplication_complex():
    """Test scalar multiplication with complex number."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=123, dtype=torch.complex128, itags=["A", "B"])

    scaled = tensor * (2 - 3j)
    for key in tensor.data:
        assert torch.allclose(scaled.data[key], tensor.data[key] * (2 - 3j))


def test_scalar_multiplication_left():
    """Test left scalar multiplication (rmul)."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    
    result = 3.5 * A
    
    for key in A.data:
        assert torch.allclose(result.data[key], A.data[key] * 3.5)


def test_scalar_multiplication_commutative():
    """Test that scalar multiplication is commutative."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    
    assert_blocks_equal(A * 2.5, 2.5 * A)


def test_scalar_multiplication_zero():
    """Test scalar multiplication by zero."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    
    result = A * 0
    
    for block in result.data.values():
        assert torch.allclose(block, torch.zeros_like(block))


# Norm tests

def test_norm_positive():
    """Test that norm is always non-negative."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    
    assert A.norm() >= 0


def test_norm_zero_iff_zero_tensor():
    """Test that norm is zero iff tensor is zero."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    Z = Tensor.zeros([idx, idx.flip()], itags=["A", "B"])
    
    assert Z.norm() == 0.0


def test_norm_linear_scaling():
    """Test that norm scales linearly with scalar multiplication."""
    idx = Index(Direction.OUT, U1Group(), sectors=(Sector(0, 4),))
    tensor = Tensor.random([idx, idx.flip()], seed=0, itags=["X", "Y"])
    scaled = tensor * 5.0
    assert math.isclose(scaled.norm(), tensor.norm() * 5.0)


def test_norm_manual_computation():
    """Test norm against manual computation."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    tensor = Tensor.random([idx, idx.flip()], seed=11, itags=["A", "B"])
    
    manual = torch.sqrt(sum(torch.sum(torch.abs(block) ** 2) for block in tensor.data.values()))
    
    assert math.isclose(tensor.norm(), manual.item())


def test_norm_complex():
    """Test norm with complex tensors."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    
    tensor = Tensor.random([idx, idx.flip()], dtype=torch.complex128, seed=1, itags=["A", "B"])
    
    manual = float(torch.sqrt(sum(torch.sum(torch.abs(block) ** 2) for block in tensor.data.values())))
    
    assert math.isclose(tensor.norm(), manual)


# Mixed operations tests

def test_combined_arithmetic_operations():
    """Test combining multiple arithmetic operations."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    B = Tensor.random([idx, idx.flip()], seed=2, itags=["A", "B"])
    C = Tensor.random([idx, idx.flip()], seed=3, itags=["A", "B"])
    
    # Test: 2*A + 3*B - C
    result = 2 * A + 3 * B - C
    
    for key in result.data:
        expected = 2 * A.data[key] + 3 * B.data[key] - C.data[key]
        assert torch.allclose(result.data[key], expected)


def test_dtype_promotion():
    """Test that dtype is promoted correctly."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], dtype=torch.float32, seed=1, itags=["A", "B"])
    B = Tensor.random([idx, idx.flip()], dtype=torch.float64, seed=2, itags=["A", "B"])
    
    result = A + B
    
    assert result.dtype == torch.float64


def test_complex_dtype_promotion():
    """Test dtype promotion with complex types."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], dtype=torch.float64, seed=1, itags=["A", "B"])
    
    result = A * (1 + 2j)
    
    assert result.dtype.is_complex


def test_z2_arithmetic():
    """Test arithmetic operations with Z2 symmetry."""
    group = Z2Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    B = Tensor.random([idx, idx.flip()], seed=2, itags=["A", "B"])
    
    C = A + B
    D = A - B
    E = 2.5 * A
    
    # Just verify operations complete without errors
    assert C.norm() > 0
    assert D.norm() >= 0
    assert E.norm() > 0


# ProductGroup integration tests for arithmetic

def test_product_group_addition():
    """Test addition with ProductGroup."""
    group = ProductGroup([U1Group(), U1Group()])
    idx = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2),
        Sector((1, -1), 1),
    ))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["x", "y"])
    B = Tensor.random([idx, idx.flip()], seed=2, itags=["x", "y"])
    
    C = A + B
    
    assert set(C.data.keys()) == set(A.data.keys())
    # Verify block-wise addition
    for key in C.data:
        expected = A.data[key] + B.data[key]
        assert torch.allclose(C.data[key], expected)


def test_product_group_subtraction():
    """Test subtraction with ProductGroup."""
    group = ProductGroup([U1Group(), Z2Group()])
    idx = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2),
        Sector((1, 1), 1),
    ))
    
    A = Tensor.random([idx, idx.flip()], seed=10, itags=["y", "z"])
    B = Tensor.random([idx, idx.flip()], seed=11, itags=["y", "z"])
    
    C = A - B
    
    for key in C.data:
        expected = A.data[key] - B.data[key]
        assert torch.allclose(C.data[key], expected)


def test_product_group_scalar_multiplication():
    """Test scalar multiplication with ProductGroup."""
    group = ProductGroup([U1Group(), U1Group()])
    idx = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 3),
        Sector((1, 2), 2),
    ))
    
    A = Tensor.random([idx, idx.flip()], seed=42, itags=["z", "w"])
    scalar = 3.5
    
    B = scalar * A
    C = A * scalar
    
    # Both should give same result
    for key in A.data:
        expected = scalar * A.data[key]
        assert torch.allclose(B.data[key], expected)
        assert torch.allclose(C.data[key], expected)


def test_addition_non_overlapping_sectors():
    """Test addition of tensors with completely non-overlapping sectors."""
    group = U1Group()
    
    # Tensor A has sectors [0, 1]
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    A = Tensor(
        indices=[idx_a, idx_a.flip()],
        itags=["i", "j"],
        data={
            (0, 0): torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            (1, 1): torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        }
    )
    
    # Tensor B has sectors [0, -1] (only 0 overlaps)
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    B = Tensor(
        indices=[idx_b, idx_b.flip()],
        itags=["i", "j"],
        data={
            (0, 0): torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
            (-1, -1): torch.tensor([[50.0, 60.0], [70.0, 80.0]])
        }
    )
    
    # Add them
    C = A + B
    
    # Result should have all blocks
    assert set(C.data.keys()) == {(0, 0), (1, 1), (-1, -1)}
    
    # Check overlapping block (0, 0) was added
    expected_00 = torch.tensor([[11.0, 22.0], [33.0, 44.0]])
    assert torch.allclose(C.data[(0, 0)], expected_00)
    
    # Check non-overlapping blocks preserved
    assert torch.allclose(C.data[(1, 1)], A.data[(1, 1)])
    assert torch.allclose(C.data[(-1, -1)], B.data[(-1, -1)])
    
    # Check result indices contain union of sectors
    result_charges = [s.charge for s in C.indices[0].sectors]
    assert sorted(result_charges) == [-1, 0, 1]


def test_subtraction_non_overlapping_sectors():
    """Test subtraction of tensors with non-overlapping sectors."""
    group = U1Group()
    
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    A = Tensor(
        indices=[idx_a, idx_a.flip()],
        itags=["i", "j"],
        data={
            (0, 0): torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            (1, 1): torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]])
        }
    )
    
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    B = Tensor(
        indices=[idx_b, idx_b.flip()],
        itags=["i", "j"],
        data={
            (0, 0): torch.tensor([[0.5, 1.0], [1.5, 2.0]]),
            (-1, -1): torch.tensor([[100.0]])
        }
    )
    
    # Subtract
    C = A - B
    
    # Result should have all blocks
    assert set(C.data.keys()) == {(0, 0), (1, 1), (-1, -1)}
    
    # Check overlapping block
    expected_00 = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    assert torch.allclose(C.data[(0, 0)], expected_00)
    
    # Check A's exclusive block preserved
    assert torch.allclose(C.data[(1, 1)], A.data[(1, 1)])
    
    # Check B's exclusive block negated
    assert torch.allclose(C.data[(-1, -1)], -B.data[(-1, -1)])


def test_addition_partially_overlapping_sectors():
    """Test addition where some sectors overlap and some don't."""
    group = U1Group()
    
    # A has sectors [-1, 0, 1]
    idx_a = Index(Direction.OUT, group, sectors=(
        Sector(-1, 2), Sector(0, 2), Sector(1, 2)
    ))
    A = Tensor(
        indices=[idx_a, idx_a.flip()],
        itags=["i", "j"],
        data={
            (-1, -1): torch.ones((2, 2)),
            (0, 0): torch.ones((2, 2)) * 2,
            (1, 1): torch.ones((2, 2)) * 3
        }
    )
    
    # B has sectors [0, 1, 2] (overlaps at 0 and 1)
    idx_b = Index(Direction.OUT, group, sectors=(
        Sector(0, 2), Sector(1, 2), Sector(2, 3)
    ))
    B = Tensor(
        indices=[idx_b, idx_b.flip()],
        itags=["i", "j"],
        data={
            (0, 0): torch.ones((2, 2)) * 10,
            (1, 1): torch.ones((2, 2)) * 20,
            (2, 2): torch.ones((3, 3)) * 30
        }
    )
    
    C = A + B
    
    # Result should have all sectors [-1, 0, 1, 2]
    assert set(C.data.keys()) == {(-1, -1), (0, 0), (1, 1), (2, 2)}
    
    # Check exclusive blocks
    assert torch.allclose(C.data[(-1, -1)], torch.ones((2, 2)))
    assert torch.allclose(C.data[(2, 2)], torch.ones((3, 3)) * 30)
    
    # Check overlapping blocks
    assert torch.allclose(C.data[(0, 0)], torch.ones((2, 2)) * 12)  # 2 + 10
    assert torch.allclose(C.data[(1, 1)], torch.ones((2, 2)) * 23)  # 3 + 20


def test_addition_empty_blocks():
    """Test addition where one tensor has no blocks in certain charges."""
    group = U1Group()
    
    # A has only charge 0
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    A = Tensor(
        indices=[idx_a, idx_a.flip()],
        itags=["i", "j"],
        data={(0, 0): torch.ones((2, 2))}
        # Note: block (1, 1) is missing (implicitly zero)
    )
    
    # B has only charge 1
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    B = Tensor(
        indices=[idx_b, idx_b.flip()],
        itags=["i", "j"],
        data={(1, 1): torch.ones((2, 2)) * 5}
        # Note: block (0, 0) is missing (implicitly zero)
    )
    
    C = A + B
    
    # Result should have both blocks
    assert set(C.data.keys()) == {(0, 0), (1, 1)}
    assert torch.allclose(C.data[(0, 0)], torch.ones((2, 2)))
    assert torch.allclose(C.data[(1, 1)], torch.ones((2, 2)) * 5)


# SU(2) addition/subtraction tests

def test_addition_su2_same_weights():
    """Test SU(2) tensor addition with same weights (default)."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Create two tensors with default weights
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    B = Tensor.random([idx1, idx2], seed=99, itags=["a", "b"])
    
    C = A + B
    
    # Should use non-Abelian addition
    assert C.intw is not None
    
    # Verify weights are unchanged (same as inputs)
    for key in C.data.keys():
        assert torch.allclose(C.intw[key].weights, A.intw[key].weights, rtol=1e-12, atol=1e-15)
        # Block shape should be unchanged (direct addition)
        assert C.data[key].shape == A.data[key].shape
        # Data should be sum of inputs
        assert torch.allclose(C.data[key], A.data[key] + B.data[key])


def test_addition_su2_different_weights():
    """Test SU(2) tensor addition with different weights."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Create tensors
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    B = Tensor.random([idx1, idx2], seed=99, itags=["a", "b"])
    
    # Modify weights of B to be different
    key = (1, 1)
    bridge_b = B.intw[key]
    om_dim = bridge_b.om_dimension
    new_weights_b = torch.randn(2, om_dim, dtype=torch.float64) * 0.5
    new_bridge_b = dg.Bridge(cgspec=bridge_b.cgspec, weights=new_weights_b)
    
    # Update B with new weights and adjusted block shape
    new_data_b = {key: torch.randn(2, 2, 2, dtype=torch.float64)}
    new_intw_b = {key: new_bridge_b}
    B_modified = Tensor(
        indices=(idx1, idx2),
        itags=("a", "b"),
        data=new_data_b,
        intw=new_intw_b,
        dtype=torch.float64
    )
    
    C = A + B_modified
    
    # Should concatenate along reduced multiplicity dimension
    assert C.intw is not None
    bridge_c = C.intw[key]
    
    # Weights should be concatenated
    expected_weights = torch.cat([A.intw[key].weights, B_modified.intw[key].weights], dim=0)
    assert torch.allclose(bridge_c.weights, expected_weights)
    
    # Block shape should have concatenated trailing dimension
    assert C.data[key].shape[-1] == A.data[key].shape[-1] + B_modified.data[key].shape[-1]
    
    # Data should be concatenated
    expected_data = torch.cat([A.data[key], B_modified.data[key]], dim=-1)
    assert torch.allclose(C.data[key], expected_data)


def test_subtraction_su2_same_weights():
    """Test SU(2) tensor subtraction with same weights (default)."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Create two tensors with default weights
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    B = Tensor.random([idx1, idx2], seed=99, itags=["a", "b"])
    
    C = A - B
    
    # Should use non-Abelian subtraction
    assert C.intw is not None
    
    # Verify weights are unchanged (same as inputs)
    for key in C.data.keys():
        assert torch.allclose(C.intw[key].weights, A.intw[key].weights, rtol=1e-12, atol=1e-15)
        # Block shape should be unchanged (direct subtraction)
        assert C.data[key].shape == A.data[key].shape
        # Data should be difference of inputs
        assert torch.allclose(C.data[key], A.data[key] - B.data[key])


def test_subtraction_su2_different_weights():
    """Test SU(2) tensor subtraction with different weights."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Create tensors
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    B = Tensor.random([idx1, idx2], seed=99, itags=["a", "b"])
    
    # Modify weights of B to be different
    key = (1, 1)
    bridge_b = B.intw[key]
    om_dim = bridge_b.om_dimension
    new_weights_b = torch.randn(2, om_dim, dtype=torch.float64) * 0.5
    new_bridge_b = dg.Bridge(cgspec=bridge_b.cgspec, weights=new_weights_b)
    
    # Update B with new weights and adjusted block shape
    new_data_b = {key: torch.randn(2, 2, 2, dtype=torch.float64)}
    new_intw_b = {key: new_bridge_b}
    B_modified = Tensor(
        indices=(idx1, idx2),
        itags=("a", "b"),
        data=new_data_b,
        intw=new_intw_b,
        dtype=torch.float64
    )
    
    C = A - B_modified
    
    # Should concatenate along reduced multiplicity dimension
    assert C.intw is not None
    bridge_c = C.intw[key]
    
    # Weights should be concatenated
    expected_weights = torch.cat([A.intw[key].weights, B_modified.intw[key].weights], dim=0)
    assert torch.allclose(bridge_c.weights, expected_weights)
    
    # Block shape should have concatenated trailing dimension
    assert C.data[key].shape[-1] == A.data[key].shape[-1] + B_modified.data[key].shape[-1]
    
    # Data should be concatenated
    expected_data = torch.cat([A.data[key], B_modified.data[key]], dim=-1)
    assert torch.allclose(C.data[key], expected_data)


def test_addition_su2_self_doubles():
    """Test that A + A doubles the tensor for SU(2) with same weights."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    C = A + A
    
    # With same weights, should add directly
    for key in C.data.keys():
        assert torch.allclose(C.data[key], 2 * A.data[key])
        assert torch.allclose(C.intw[key].weights, A.intw[key].weights, rtol=1e-12, atol=1e-15)


def test_subtraction_su2_self_gives_zero():
    """Test that A - A gives zero for SU(2) tensors."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    C = A - A
    
    # Result should be zero
    assert C.norm() < 1e-12


def test_addition_su2_collinear_weights():
    """Test SU(2) addition with collinear weights (parallel but scaled)."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    B = Tensor.random([idx1, idx2], seed=99, itags=["a", "b"])
    
    # Scale B's weights to make them collinear with A
    key = (1, 1)
    bridge_a = A.intw[key]
    alpha = 3.0
    scaled_weights = bridge_a.weights * alpha
    B.intw[key] = dg.Bridge(cgspec=bridge_a.cgspec, weights=scaled_weights)
    
    # Store original data for manual verification
    data_a = A.data[key].clone()
    data_b = B.data[key].clone()
    
    C = A + B
    
    # Should detect collinearity and scale+add
    # Result: (R_a + α*R_b) @ w_a
    assert C.intw[key].num_components == 1  # Not concatenated
    expected_data = data_a + data_b * alpha
    assert torch.allclose(C.data[key], expected_data)


def test_subtraction_su2_collinear_weights():
    """Test SU(2) subtraction with collinear weights (parallel but scaled)."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    B = Tensor.random([idx1, idx2], seed=99, itags=["a", "b"])
    
    # Scale B's weights to make them collinear with A
    key = (1, 1)
    bridge_a = A.intw[key]
    alpha = -2.0  # Test negative scaling too
    scaled_weights = bridge_a.weights * alpha
    B.intw[key] = dg.Bridge(cgspec=bridge_a.cgspec, weights=scaled_weights)
    
    # Store original data for manual verification
    data_a = A.data[key].clone()
    data_b = B.data[key].clone()
    
    C = A - B
    
    # Should detect collinearity and scale+subtract
    # Result: (R_a - α*R_b) @ w_a
    assert C.intw[key].num_components == 1  # Not concatenated
    expected_data = data_a - data_b * alpha
    assert torch.allclose(C.data[key], expected_data)


def test_addition_su2_norm_conservation():
    """Test that addition preserves norm properties for SU(2)."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Create two tensors with default weights (orthogonal in OM space)
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    B = Tensor.random([idx1, idx2], seed=99, itags=["a", "b"])
    
    C = A + B
    
    # For same weights, ||A + B||² should be related to ||A||² and ||B||²
    # Since default weights are the same, this is just element-wise addition
    norm_a = A.norm()
    norm_b = B.norm()
    norm_c = C.norm()
    
    # Verify norm is computed correctly (not exact triangle inequality due to structure)
    assert norm_c > 0.0
    assert norm_c >= abs(norm_a - norm_b)


# Compression tests

def test_compress_su2_no_redundancy():
    """Test compression when weights have no redundancy (default case)."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    
    # Store original state
    original_norm = A.norm()
    original_num_components = {key: bridge.num_components for key, bridge in A.intw.items()}
    
    # Default weights have 1 component, so no compression
    A.compress()
    
    # Should be unchanged
    assert math.isclose(A.norm(), original_norm)
    for key in A.data.keys():
        assert A.intw[key].num_components == original_num_components[key]


def test_compress_su2_linearly_dependent():
    """Test compression with linearly dependent weight rows."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Create a tensor with manually constructed redundant weights
    key = (1, 1)
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    
    # Create weights with linearly dependent rows
    bridge_a = A.intw[key]
    om_dim = bridge_a.om_dimension
    
    # Create 3 components where third is linear combination of first two
    base_weight = torch.randn(1, om_dim, dtype=torch.float64)
    weights_redundant = torch.cat([
        base_weight,
        base_weight * 2.0,
        base_weight * 0.5  # Linear combination
    ], dim=0)
    
    # Adjust data block to match
    new_data = {key: torch.randn(2, 2, 3, dtype=torch.float64)}
    new_intw = {key: dg.Bridge(cgspec=bridge_a.cgspec, weights=weights_redundant)}
    
    A_redundant = Tensor(
        indices=(idx1, idx2),
        itags=("a", "b"),
        data=new_data,
        intw=new_intw,
        dtype=torch.float64
    )
    
    # Store original norm
    original_norm = A_redundant.norm()
    
    # Compress
    A_redundant.compress(cutoff=1e-12)
    
    # Should reduce to 1 or 2 components (rank of weight matrix)
    assert A_redundant.intw[key].num_components < 3
    assert A_redundant.intw[key].num_components >= 1
    
    # Norm should be approximately preserved
    assert math.isclose(A_redundant.norm(), original_norm, rel_tol=1e-10)


def test_compress_su2_after_addition_different_weights():
    """Test compression after adding tensors with different weights."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Create two tensors with different weights
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    B = Tensor.random([idx1, idx2], seed=99, itags=["a", "b"])
    
    # Modify B's weights to be different
    key = (1, 1)
    bridge_b = B.intw[key]
    om_dim = bridge_b.om_dimension
    new_weights_b = torch.randn(2, om_dim, dtype=torch.float64)
    new_bridge_b = dg.Bridge(cgspec=bridge_b.cgspec, weights=new_weights_b)
    
    new_data_b = {key: torch.randn(2, 2, 2, dtype=torch.float64)}
    new_intw_b = {key: new_bridge_b}
    B_modified = Tensor(
        indices=(idx1, idx2),
        itags=("a", "b"),
        data=new_data_b,
        intw=new_intw_b,
        dtype=torch.float64
    )
    
    # Add: this concatenates to 3 components (1 + 2)
    C = A + B_modified
    assert C.intw[key].num_components == 3
    
    # Store original state
    original_norm = C.norm()
    original_num_components = C.intw[key].num_components
    
    # Compress: should detect if any redundancy exists
    C.compress(cutoff=1e-13)
    
    # Norm should be preserved
    assert math.isclose(C.norm(), original_norm, rel_tol=1e-10)
    
    # Components should be <= original
    assert C.intw[key].num_components <= original_num_components


def test_compress_su2_specific_keys():
    """Test compression on specific block keys only."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Create tensor with multiple blocks
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    B = Tensor.random([idx1, idx2], seed=99, itags=["a", "b"])
    
    # Modify one block to have different weights
    key1 = (1, 1)
    if key1 in B.data:
        bridge = B.intw[key1]
        om_dim = bridge.om_dimension
        new_weights = torch.randn(2, om_dim, dtype=torch.float64)
        B.intw[key1] = dg.Bridge(cgspec=bridge.cgspec, weights=new_weights)
        B.data[key1] = torch.randn(*B.data[key1].shape[:-1], 2, dtype=torch.float64)
    
    C = A + B
    
    # Store original state for other keys
    original_num_components = {key: bridge.num_components for key, bridge in C.intw.items()}
    
    # Compress only specific key
    if key1 in C.data:
        C.compress(keys=[key1])
        
        # Other keys should remain unchanged
        for key in C.data.keys():
            if key != key1:
                assert C.intw[key].num_components == original_num_components[key]


def test_compress_abelian_no_op():
    """Test that compression on Abelian tensors is a no-op."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    A = Tensor.random([idx, idx.flip()], seed=42, itags=["a", "b"])
    original_norm = A.norm()
    
    # Should be a no-op
    A.compress()
    
    # Should be unchanged
    assert math.isclose(A.norm(), original_norm)
    assert A.intw is None


def test_compress_su2_preserves_tensor_value():
    """Test that compression preserves the physical tensor value."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Create tensor with redundant weights
    A = Tensor.random([idx1, idx2], seed=42, itags=["a", "b"])
    B = Tensor.random([idx1, idx2], seed=99, itags=["a", "b"])
    
    # Make B have parallel weights to A (same direction)
    key = (1, 1)
    bridge_a = A.intw[key]
    # Scale A's weights to create B's weights (linearly dependent)
    new_weights_b = bridge_a.weights * 2.0
    new_bridge_b = dg.Bridge(cgspec=bridge_a.cgspec, weights=new_weights_b)
    
    # Keep B's data the same but with adjusted shape
    new_data_b = {key: B.data[key]}
    new_intw_b = {key: new_bridge_b}
    B_modified = Tensor(
        indices=(idx1, idx2),
        itags=("a", "b"),
        data=new_data_b,
        intw=new_intw_b,
        dtype=torch.float64
    )
    
    C = A + B_modified  # Concatenates to 2 components
    
    # Store original norm
    original_norm = C.norm()
    
    # Compress should reduce to 1 component (since weights are parallel)
    C.compress(cutoff=1e-13)
    
    # Verify norm is preserved (main check that physical tensor is unchanged)
    assert math.isclose(C.norm(), original_norm, rel_tol=1e-10)


# Multi-index and multi-sector tests (non-trivial CG and OM)

def test_addition_su2_three_indices_nontrivial_cg():
    """Test SU(2) addition with 3 indices for non-trivial CG coupling."""
    group = SU2Group()
    # Three spin-1/2 indices
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    B = Tensor.random([idx1, idx2, idx3], seed=99, itags=["a", "b", "c"])
    
    # With default weights, should add directly
    C = A + B
    
    # Verify structure
    assert C.intw is not None
    for key in C.data.keys():
        # Same weights, so components unchanged
        assert C.intw[key].num_components == A.intw[key].num_components
        # Data should be sum
        assert torch.allclose(C.data[key], A.data[key] + B.data[key])


def test_addition_su2_four_indices_nontrivial_om():
    """Test SU(2) addition with 4 indices for non-trivial outer multiplicity."""
    group = SU2Group()
    # Four spin-1/2 indices: creates non-trivial OM dimension
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx4 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx1, idx2, idx3, idx4], seed=42, itags=["a", "b", "c", "d"])
    B = Tensor.random([idx1, idx2, idx3, idx4], seed=99, itags=["a", "b", "c", "d"])
    
    # Check OM dimension is non-trivial
    for key, bridge in A.intw.items():
        # For (1,1,1,1) block, OM dimension should be > 1
        if all(q == 1 for q in key):
            assert bridge.om_dimension > 1, f"Expected non-trivial OM for {key}"
    
    # Addition with same weights
    C = A + B
    
    # Verify structure preserved
    assert C.intw is not None
    for key in C.data.keys():
        assert torch.allclose(C.data[key], A.data[key] + B.data[key])


def test_addition_su2_multiple_sectors():
    """Test SU(2) addition with multiple sectors creating multiple blocks."""
    group = SU2Group()
    # Multiple sectors in each index
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 3)))
    
    A = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    B = Tensor.random([idx1, idx2, idx3], seed=99, itags=["a", "b", "c"])
    
    # Should have multiple blocks
    assert len(A.data) > 1, "Expected multiple blocks"
    
    C = A + B
    
    # All blocks should be summed correctly
    for key in C.data.keys():
        assert torch.allclose(C.data[key], A.data[key] + B.data[key])


def test_addition_su2_three_indices_collinear_weights():
    """Test SU(2) addition with 3 indices and collinear weights (parallel vectors)."""
    group = SU2Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    B = Tensor.random([idx1, idx2, idx3], seed=99, itags=["a", "b", "c"])
    
    # Scale B's weights to be collinear with A's
    alpha = 2.5
    for key in B.intw.keys():
        B.intw[key].weights[:] = A.intw[key].weights * alpha
    
    C = A + B
    
    # Collinear weights: should add with scaling, no OM expansion
    for key in C.data.keys():
        assert C.intw[key].num_components == 1, "Collinear should not expand OM"
        # Verify weights match A's (reference weights)
        assert torch.allclose(C.intw[key].weights, A.intw[key].weights)


def test_addition_su2_four_indices_collinear_weights():
    """Test SU(2) addition with 4 indices (non-trivial OM) and collinear weights."""
    group = SU2Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx4 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx1, idx2, idx3, idx4], seed=42, itags=["a", "b", "c", "d"])
    B = Tensor.random([idx1, idx2, idx3, idx4], seed=99, itags=["a", "b", "c", "d"])
    
    # Scale B's weights to be collinear with A's
    alpha = -1.5
    for key in B.intw.keys():
        B.intw[key].weights[:] = A.intw[key].weights * alpha
    
    C = A + B
    
    # Collinear weights: should add with scaling, no OM expansion
    for key in C.data.keys():
        assert C.intw[key].num_components == 1, "Collinear should not expand OM"
        # Verify non-trivial OM dimension
        if all(q == 1 for q in key):
            assert C.intw[key].om_dimension > 1


def test_addition_su2_multiple_sectors_collinear_weights():
    """Test SU(2) addition with multiple sectors and collinear weights."""
    group = SU2Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 3)))
    
    A = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    B = Tensor.random([idx1, idx2, idx3], seed=99, itags=["a", "b", "c"])
    
    # Scale B's weights to be collinear with A's
    alpha = 3.0
    for key in B.intw.keys():
        B.intw[key].weights[:] = A.intw[key].weights * alpha
    
    # Should have multiple blocks
    assert len(A.data) > 1, "Expected multiple blocks"
    
    C = A + B
    
    # All blocks should use collinear addition (no OM expansion)
    for key in C.data.keys():
        assert C.intw[key].num_components == 1, "Collinear should not expand OM"


def test_subtraction_su2_three_indices_collinear_weights():
    """Test SU(2) subtraction with 3 indices and collinear weights."""
    group = SU2Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    B = Tensor.random([idx1, idx2, idx3], seed=99, itags=["a", "b", "c"])
    
    # Scale B's weights to be collinear with A's
    alpha = 0.75
    for key in B.intw.keys():
        B.intw[key].weights[:] = A.intw[key].weights * alpha
    
    C = A - B
    
    # Collinear weights: should subtract with scaling, no OM expansion
    for key in C.data.keys():
        assert C.intw[key].num_components == 1, "Collinear should not expand OM"
        # Verify weights match A's
        assert torch.allclose(C.intw[key].weights, A.intw[key].weights)


def test_subtraction_su2_four_indices_different_weights():
    """Test SU(2) subtraction with 4 indices and different weights."""
    group = SU2Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx4 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx1, idx2, idx3, idx4], seed=42, itags=["a", "b", "c", "d"])
    B = Tensor.random([idx1, idx2, idx3, idx4], seed=99, itags=["a", "b", "c", "d"])
    
    # Modify B's weights to be different
    key = (1, 1, 1, 1)
    if key in B.intw:
        bridge_b = B.intw[key]
        om_dim = bridge_b.om_dimension
        # Create different weights with 2 components
        new_weights_b = torch.randn(2, om_dim, dtype=torch.float64)
        B.intw[key] = dg.Bridge(cgspec=bridge_b.cgspec, weights=new_weights_b)
        # Adjust data shape
        B.data[key] = torch.randn(*B.data[key].shape[:-1], 2, dtype=torch.float64)
    
    C = A - B
    
    # Should concatenate weights
    if key in C.intw:
        assert C.intw[key].num_components == A.intw[key].num_components + 2


def test_compress_su2_three_indices_multiple_sectors():
    """Test compression with 3 indices and multiple sectors."""
    group = SU2Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 3)))
    
    A = Tensor.random([idx1, idx2, idx3], seed=42, itags=["a", "b", "c"])
    B = Tensor.random([idx1, idx2, idx3], seed=99, itags=["a", "b", "c"])
    
    # Create different weights for some blocks
    for key in B.intw.keys():
        bridge_b = B.intw[key]
        om_dim = bridge_b.om_dimension
        # Add extra components with linearly dependent weights
        base_weight = bridge_b.weights[0:1, :]  # First component
        new_weights = torch.cat([
            bridge_b.weights,
            base_weight * 2.0,  # Linearly dependent
            base_weight * 0.5   # Linearly dependent
        ], dim=0)
        B.intw[key] = dg.Bridge(cgspec=bridge_b.cgspec, weights=new_weights)
        B.data[key] = torch.randn(*B.data[key].shape[:-1], new_weights.shape[0], dtype=torch.float64)
    
    C = A + B
    
    # Should have multiple blocks with increased components
    assert len(C.data) > 1
    original_norm = C.norm()
    original_components = {key: bridge.num_components for key, bridge in C.intw.items()}
    
    # Compress
    C.compress(cutoff=1e-12)
    
    # Norm should be preserved
    assert math.isclose(C.norm(), original_norm, rel_tol=1e-10)
    
    # At least some blocks should be compressed (linearly dependent weights removed)
    compressed_components = {key: bridge.num_components for key, bridge in C.intw.items()}
    compression_occurred = any(
        compressed_components[key] < original_components[key] 
        for key in original_components
    )
    assert compression_occurred, "Expected at least one block to be compressed"


def test_compress_su2_four_indices_nontrivial_om():
    """Test compression with 4 indices and non-trivial OM."""
    group = SU2Group()
    # Four spin-1/2 indices
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx4 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx1, idx2, idx3, idx4], seed=42, itags=["a", "b", "c", "d"])
    B = Tensor.random([idx1, idx2, idx3, idx4], seed=99, itags=["a", "b", "c", "d"])
    
    # Verify non-trivial OM dimension
    key = (1, 1, 1, 1)
    if key in A.intw:
        assert A.intw[key].om_dimension > 1
    
    # Create scenario where weights have redundancy
    if key in B.intw:
        bridge_b = B.intw[key]
        om_dim = bridge_b.om_dimension
        
        # Create weights where some rows are linear combinations
        base1 = torch.randn(1, om_dim, dtype=torch.float64)
        base2 = torch.randn(1, om_dim, dtype=torch.float64)
        redundant_weights = torch.cat([
            base1,
            base2,
            base1 * 0.5 + base2 * 0.3,  # Linear combination
            base1 * 0.2 + base2 * 0.7   # Another linear combination
        ], dim=0)
        
        B.intw[key] = dg.Bridge(cgspec=bridge_b.cgspec, weights=redundant_weights)
        B.data[key] = torch.randn(*B.data[key].shape[:-1], 4, dtype=torch.float64)
    
    C = A + B
    
    original_norm = C.norm()
    if key in C.intw:
        original_components = C.intw[key].num_components
    
    # Compress
    C.compress(cutoff=1e-12)
    
    # Norm should be preserved
    assert math.isclose(C.norm(), original_norm, rel_tol=1e-10)
    
    # Should reduce components (rank should be ~2)
    if key in C.intw:
        assert C.intw[key].num_components < original_components
        assert C.intw[key].num_components >= 2  # Rank of the span


def test_addition_subtraction_compress_workflow():
    """Test realistic workflow: multiple additions/subtractions followed by compression."""
    group = SU2Group()
    # Three indices with multiple sectors
    idx1 = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 3)))
    
    # Create several tensors
    tensors = [Tensor.random([idx1, idx2, idx3], seed=s, itags=["a", "b", "c"]) 
               for s in [10, 20, 30, 40]]
    
    # Modify some to have different weights
    for i, t in enumerate(tensors[1:], 1):
        for key in t.intw.keys():
            if i % 2 == 1:  # Modify odd-indexed tensors
                bridge = t.intw[key]
                om_dim = bridge.om_dimension
                new_weights = torch.randn(i + 1, om_dim, dtype=torch.float64) * 0.1
                t.intw[key] = dg.Bridge(cgspec=bridge.cgspec, weights=new_weights)
                t.data[key] = torch.randn(*t.data[key].shape[:-1], i + 1, dtype=torch.float64)
    
    # Combine: (A + B) - (C + D)
    AB = tensors[0] + tensors[1]
    CD = tensors[2] + tensors[3]
    result = AB - CD
    
    # Should have accumulated many components
    max_components_before = max(bridge.num_components for bridge in result.intw.values())
    assert max_components_before >= 2
    
    original_norm = result.norm()
    
    # Compress to remove redundancy
    result.compress(cutoff=1e-13)
    
    # Norm preserved
    assert math.isclose(result.norm(), original_norm, rel_tol=1e-10)
    
    # Components may be reduced
    max_components_after = max(bridge.num_components for bridge in result.intw.values())
    assert max_components_after <= max_components_before

