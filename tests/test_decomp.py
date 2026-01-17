# Copyright (C) 2025 Changkai Zhang.
#
# This file is part of Nicole (TN) library.
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


"""Tests for tensor decomposition operations: SVD."""

import numpy as np
import pytest

from nicole import Direction, Tensor, contract, svd, U1Group, Index, Sector
from .utils import assert_charge_neutral


# Basic SVD tests

def test_svd_basic_reconstruction():
    """Test basic SVD and reconstruction on a simple tensor."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=42)
    original_norm = T.norm()

    # Perform SVD separating axis 0 from the rest
    U, S, Vh = svd(T, axis=0)

    # Check that all tensors are charge neutral
    assert_charge_neutral(U)
    assert_charge_neutral(S)
    assert_charge_neutral(Vh)

    # Reconstruct tensor (automatic detection based on matching itags)
    S_Vh = contract(S, Vh)
    reconstructed = contract(U, S_Vh)

    # Check reconstruction accuracy
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / original_norm
    assert rel_error < 1e-12, f"Reconstruction error {rel_error} too large"
    
    # Check norms are preserved
    np.testing.assert_allclose(reconstructed.norm(), original_norm, rtol=1e-12)


def test_svd_integer_axis():
    """Test SVD with integer axis specification."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=1)
    
    U, S, Vh = svd(T, axis=0)
    
    assert len(U.indices) == 2
    assert len(S.indices) == 2
    assert len(Vh.indices) == 2


def test_svd_string_axis():
    """Test SVD with axis specified by string tag."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=123)

    # Perform SVD by axis name
    U, S, Vh = svd(T, axis="b")

    # Check structure
    assert len(U.indices) == 2
    assert len(S.indices) == 2
    assert len(Vh.indices) == 3
    assert U.itags[0] == "b"
    assert Vh.itags[1] == "a"
    assert Vh.itags[2] == "c"

    # Reconstruct and verify
    S_Vh = contract(S, Vh)
    reconstructed = contract(U, S_Vh)
    
    # Permute reconstructed to match original order (b, a, c) -> (a, b, c)
    reconstructed.permute([1, 0, 2])
    
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / T.norm()
    assert rel_error < 1e-12


def test_svd_different_axis_positions():
    """Test SVD on different axis positions."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 4),))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=111)

    # Test SVD on each axis
    for axis in [0, 1, 2]:
        U, S, Vh = svd(T, axis=axis)
        
        # Reconstruct (automatic detection)
        S_Vh = contract(S, Vh)
        reconstructed = contract(U, S_Vh)
        
        # Permute back to original order
        if axis == 0:
            # Already in correct order (a, b, c)
            pass
        elif axis == 1:
            # Current order (b, a, c) -> (a, b, c)
            reconstructed.permute([1, 0, 2])
        elif axis == 2:
            # Current order (c, a, b) -> (a, b, c)
            reconstructed.permute([1, 2, 0])
        
        # Verify reconstruction
        diff_norm = (T - reconstructed).norm()
        rel_error = diff_norm / T.norm()
        assert rel_error < 1e-12, f"Reconstruction failed for axis {axis}"


# Block handling tests

def test_svd_multiple_blocks_same_charge():
    """Test SVD with multiple blocks sharing the same left charge."""
    group = U1Group()
    
    # Create a tensor where multiple blocks share the same left charge
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))

    # Manually create tensor with specific blocks
    data = {}
    # Both blocks have q_left=0, testing grouped SVD
    data[(0, 0, 0)] = np.random.randn(2, 2, 2)
    data[(0, 1, 1)] = np.random.randn(2, 2, 2)

    T = Tensor(indices=(idx1, idx2, idx3), itags=["a", "b", "c"], data=data, dtype=np.float64)

    # Perform SVD
    U, S, Vh = svd(T, axis=0)

    # Verify that we have grouped the blocks correctly
    # Should have 1 U block, 1 S block, and 2 Vh blocks
    assert len(U.data) == 1, "Should have 1 U block for q_left=0"
    assert len(S.data) == 1, "Should have 1 S block for q_left=0"
    assert len(Vh.data) == 2, "Should have 2 Vh blocks (one per q_right combination)"

    # Check that bond dimension is min(2, 8) = 2
    for key, block in U.data.items():
        assert block.shape[1] == 2, "Bond dimension should be 2"

    # Reconstruct and verify
    S_Vh = contract(S, Vh)
    reconstructed = contract(U, S_Vh)
    
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / T.norm()
    assert rel_error < 1e-12


def test_svd_single_block():
    """Test SVD with single block tensor."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 4),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=999)
    
    U, S, Vh = svd(T, axis=0)
    
    assert len(U.data) == 1
    assert len(S.data) == 1
    assert len(Vh.data) == 1


# Index and direction tests

def test_svd_index_directions():
    """Test that SVD produces correct index directions for contraction."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=456)

    U, S, Vh = svd(T, axis=0)

    # Check index directions
    left_dir = T.indices[0].direction
    bond_dir = U.indices[1].direction
    
    # Bond should be reversed from left
    assert bond_dir == left_dir.reverse(), "Bond index should be reversed from left"
    
    # S should have (bond.flip(), bond)
    assert S.indices[0].direction == bond_dir.reverse(), "S first index should be bond.flip()"
    assert S.indices[1].direction == bond_dir, "S second index should be bond"
    
    # Vh should have (bond.flip(), ...)
    assert Vh.indices[0].direction == bond_dir.reverse(), "Vh first index should be bond.flip()"
    
    # Verify contraction works
    S_Vh = contract(S, Vh)
    reconstructed = contract(U, S_Vh)
    
    # Check that reconstructed has same index directions as original
    for i, idx in enumerate(reconstructed.indices):
        assert idx.direction == T.indices[i].direction


def test_svd_bond_index_structure():
    """Test that bond index has correct structure."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=2)
    
    U, S, Vh = svd(T, axis=0)
    
    bond_index = U.indices[1]
    
    # Bond should have same group as left index
    assert bond_index.group == idx1.group
    
    # Bond charges should be a subset of left charges (only charges that appear in data)
    bond_charges = bond_index.charges()
    left_charges = idx1.charges()
    assert set(bond_charges).issubset(set(left_charges))
    
    # Bond charges should match charges that appear in the left blocks of T
    left_charges_in_data = set(key[0] for key in T.data.keys())
    assert set(bond_charges) == left_charges_in_data


# Singular value tests

def test_svd_singular_values_positive():
    """Test that singular values are positive and sorted."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 4),))

    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=789)

    U, S, Vh = svd(T, axis=0)

    # Extract singular values from S blocks
    for key, block in S.data.items():
        singular_values = np.diag(block)
        # Check all positive
        assert np.all(singular_values >= 0), "Singular values should be non-negative"
        # Check sorted in descending order
        assert np.allclose(singular_values, np.sort(singular_values)[::-1]), \
            "Singular values should be sorted descending"


def test_svd_bond_dimensions():
    """Test that bond dimensions are computed correctly."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5),))

    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=444)

    U, S, Vh = svd(T, axis=0)

    # Bond dimension should be min(3, 5) = 3
    bond_index = U.indices[1]
    assert bond_index.dim == 3, "Bond dimension should be min of left and right dimensions"
    
    # Check S block dimensions match
    for key, block in S.data.items():
        assert block.shape[0] == bond_index.dim
        assert block.shape[1] == bond_index.dim


def test_svd_s_diagonal():
    """Test that S tensor contains diagonal matrices."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 4),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=3)
    
    U, S, Vh = svd(T, axis=0)
    
    # Check that S blocks are diagonal
    for key, block in S.data.items():
        # Off-diagonal elements should be zero
        assert np.allclose(block, np.diag(np.diag(block)))


# Charge conservation tests

def test_svd_preserves_charge_conservation():
    """Test that SVD preserves charge conservation in all output tensors."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=222)
    
    # Original tensor should be charge neutral
    assert_charge_neutral(T)

    # Perform SVD
    U, S, Vh = svd(T, axis=1)

    # All output tensors should be charge neutral
    assert_charge_neutral(U)
    assert_charge_neutral(S)
    assert_charge_neutral(Vh)


def test_svd_charge_conservation_all_axes():
    """Test charge conservation for SVD on all axes."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=4)
    
    for axis in [0, 1, 2]:
        U, S, Vh = svd(T, axis=axis)
        
        assert_charge_neutral(U)
        assert_charge_neutral(S)
        assert_charge_neutral(Vh)


# Error handling tests

def test_svd_invalid_integer_axis():
    """Test that SVD raises error for invalid integer axis."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))

    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=333)

    # Test invalid integer axis
    with pytest.raises(ValueError, match="out of range"):
        svd(T, axis=5)
    
    with pytest.raises(ValueError, match="out of range"):
        svd(T, axis=-1)


def test_svd_invalid_string_axis():
    """Test that SVD raises error for invalid string axis."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))

    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=333)

    # Test invalid string axis
    with pytest.raises(ValueError, match="not found"):
        svd(T, axis="nonexistent")


# dtype tests

def test_svd_complex_dtype():
    """Test SVD with complex dtype."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 4),))
    
    T = Tensor.random([idx1, idx2], dtype=np.complex128, itags=["a", "b"], seed=5)
    
    U, S, Vh = svd(T, axis=0)
    
    # U and Vh should be complex, S should be real
    assert np.issubdtype(U.dtype, np.complexfloating)
    assert np.issubdtype(Vh.dtype, np.complexfloating)
    
    # Reconstruct
    S_Vh = contract(S, Vh)
    reconstructed = contract(U, S_Vh)
    
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / T.norm()
    assert rel_error < 1e-12

