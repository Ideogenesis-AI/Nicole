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


"""Tests for Tensor arithmetic operations."""

import numpy as np
import pytest

from nicole import Direction, Index, Sector, Tensor, U1Group, Z2Group
from nicole.symmetry.product import ProductGroup
from .utils import assert_blocks_equal


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
        np.testing.assert_allclose(C.data[key], expected)


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
        np.testing.assert_allclose(sum_tensor.data[key], manual_data[key])


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
        np.testing.assert_allclose(C.data[key], expected)


def test_subtraction_self_gives_zero():
    """Test that A - A gives zero."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    
    result = A - A
    
    for block in result.data.values():
        assert np.allclose(block, 0.0)


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
        np.testing.assert_allclose(result.data[key], A.data[key] * 3)


def test_scalar_multiplication_float():
    """Test scalar multiplication with float."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    
    result = A * 2.5
    
    for key in A.data:
        np.testing.assert_allclose(result.data[key], A.data[key] * 2.5)


def test_scalar_multiplication_complex():
    """Test scalar multiplication with complex number."""
    group = U1Group()
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_b = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))
    tensor = Tensor.random([idx_a, idx_b], seed=123, dtype=np.complex128, itags=["A", "B"])

    scaled = tensor * (2 - 3j)
    for key in tensor.data:
        np.testing.assert_allclose(scaled.data[key], tensor.data[key] * (2 - 3j))


def test_scalar_multiplication_left():
    """Test left scalar multiplication (rmul)."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=["A", "B"])
    
    result = 3.5 * A
    
    for key in A.data:
        np.testing.assert_allclose(result.data[key], A.data[key] * 3.5)


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
        assert np.allclose(block, 0.0)


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
    assert np.isclose(scaled.norm(), tensor.norm() * 5.0)


def test_norm_manual_computation():
    """Test norm against manual computation."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    tensor = Tensor.random([idx, idx.flip()], seed=11, itags=["A", "B"])
    
    manual = np.sqrt(sum(np.sum(np.abs(block) ** 2) for block in tensor.data.values()))
    
    assert np.isclose(tensor.norm(), manual)


def test_norm_complex():
    """Test norm with complex tensors."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    
    tensor = Tensor.random([idx, idx.flip()], dtype=np.complex128, seed=1, itags=["A", "B"])
    
    manual = np.sqrt(sum(np.sum(np.abs(block) ** 2) for block in tensor.data.values()))
    
    assert np.isclose(tensor.norm(), manual)


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
        np.testing.assert_allclose(result.data[key], expected)


def test_dtype_promotion():
    """Test that dtype is promoted correctly."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], dtype=np.float32, seed=1, itags=["A", "B"])
    B = Tensor.random([idx, idx.flip()], dtype=np.float64, seed=2, itags=["A", "B"])
    
    result = A + B
    
    assert result.dtype == np.float64


def test_complex_dtype_promotion():
    """Test dtype promotion with complex types."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], dtype=np.float64, seed=1, itags=["A", "B"])
    
    result = A * (1 + 2j)
    
    assert np.issubdtype(result.dtype, np.complexfloating)


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
        np.testing.assert_allclose(C.data[key], expected)


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
        np.testing.assert_allclose(C.data[key], expected)


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
        np.testing.assert_allclose(B.data[key], expected)
        np.testing.assert_allclose(C.data[key], expected)


def test_addition_non_overlapping_sectors():
    """Test addition of tensors with completely non-overlapping sectors."""
    group = U1Group()
    
    # Tensor A has sectors [0, 1]
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    A = Tensor(
        indices=[idx_a, idx_a.flip()],
        itags=["i", "j"],
        data={
            (0, 0): np.array([[1.0, 2.0], [3.0, 4.0]]),
            (1, 1): np.array([[5.0, 6.0], [7.0, 8.0]])
        }
    )
    
    # Tensor B has sectors [0, -1] (only 0 overlaps)
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    B = Tensor(
        indices=[idx_b, idx_b.flip()],
        itags=["i", "j"],
        data={
            (0, 0): np.array([[10.0, 20.0], [30.0, 40.0]]),
            (-1, -1): np.array([[50.0, 60.0], [70.0, 80.0]])
        }
    )
    
    # Add them
    C = A + B
    
    # Result should have all blocks
    assert set(C.data.keys()) == {(0, 0), (1, 1), (-1, -1)}
    
    # Check overlapping block (0, 0) was added
    expected_00 = np.array([[11.0, 22.0], [33.0, 44.0]])
    assert np.allclose(C.data[(0, 0)], expected_00)
    
    # Check non-overlapping blocks preserved
    assert np.allclose(C.data[(1, 1)], A.data[(1, 1)])
    assert np.allclose(C.data[(-1, -1)], B.data[(-1, -1)])
    
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
            (0, 0): np.array([[1.0, 2.0], [3.0, 4.0]]),
            (1, 1): np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]])
        }
    )
    
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    B = Tensor(
        indices=[idx_b, idx_b.flip()],
        itags=["i", "j"],
        data={
            (0, 0): np.array([[0.5, 1.0], [1.5, 2.0]]),
            (-1, -1): np.array([[100.0]])
        }
    )
    
    # Subtract
    C = A - B
    
    # Result should have all blocks
    assert set(C.data.keys()) == {(0, 0), (1, 1), (-1, -1)}
    
    # Check overlapping block
    expected_00 = np.array([[0.5, 1.0], [1.5, 2.0]])
    assert np.allclose(C.data[(0, 0)], expected_00)
    
    # Check A's exclusive block preserved
    assert np.allclose(C.data[(1, 1)], A.data[(1, 1)])
    
    # Check B's exclusive block negated
    assert np.allclose(C.data[(-1, -1)], -B.data[(-1, -1)])


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
            (-1, -1): np.ones((2, 2)),
            (0, 0): np.ones((2, 2)) * 2,
            (1, 1): np.ones((2, 2)) * 3
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
            (0, 0): np.ones((2, 2)) * 10,
            (1, 1): np.ones((2, 2)) * 20,
            (2, 2): np.ones((3, 3)) * 30
        }
    )
    
    C = A + B
    
    # Result should have all sectors [-1, 0, 1, 2]
    assert set(C.data.keys()) == {(-1, -1), (0, 0), (1, 1), (2, 2)}
    
    # Check exclusive blocks
    assert np.allclose(C.data[(-1, -1)], np.ones((2, 2)))
    assert np.allclose(C.data[(2, 2)], np.ones((3, 3)) * 30)
    
    # Check overlapping blocks
    assert np.allclose(C.data[(0, 0)], np.ones((2, 2)) * 12)  # 2 + 10
    assert np.allclose(C.data[(1, 1)], np.ones((2, 2)) * 23)  # 3 + 20


def test_addition_empty_blocks():
    """Test addition where one tensor has no blocks in certain charges."""
    group = U1Group()
    
    # A has only charge 0
    idx_a = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    A = Tensor(
        indices=[idx_a, idx_a.flip()],
        itags=["i", "j"],
        data={(0, 0): np.ones((2, 2))}
        # Note: block (1, 1) is missing (implicitly zero)
    )
    
    # B has only charge 1
    idx_b = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    B = Tensor(
        indices=[idx_b, idx_b.flip()],
        itags=["i", "j"],
        data={(1, 1): np.ones((2, 2)) * 5}
        # Note: block (0, 0) is missing (implicitly zero)
    )
    
    C = A + B
    
    # Result should have both blocks
    assert set(C.data.keys()) == {(0, 0), (1, 1)}
    assert np.allclose(C.data[(0, 0)], np.ones((2, 2)))
    assert np.allclose(C.data[(1, 1)], np.ones((2, 2)) * 5)

