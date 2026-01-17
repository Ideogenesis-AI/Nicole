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

from nicole import Direction, Tensor, contract, decomp, U1Group, Index, Sector
from nicole.decomp import svd
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
    U, S, Vh = decomp(T, axis=0, mode="SVD")

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
    
    U, S, Vh = decomp(T, axis=0, mode="SVD")
    
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
    U, S, Vh = decomp(T, axis="b", mode="SVD")

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

    # Test SVD on each axis (using UR mode for efficiency)
    for axis in [0, 1, 2]:
        U, R = decomp(T, axis=axis, mode="UR")
        
        # Reconstruct using explicit pairs
        reconstructed = contract(U, R, pairs=[(1, 0)])
        
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

    # Perform SVD using decomp
    U, S, Vh = decomp(T, axis=0, mode="SVD")

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
    
    # Test with raw svd() - should return dict for S
    U, S_values, Vh = svd(T, axis=0)
    
    assert len(U.data) == 1
    assert isinstance(S_values, dict)
    assert len(S_values) == 1
    assert len(Vh.data) == 1


# Index and direction tests

def test_svd_index_directions():
    """Test that SVD produces correct index directions for contraction."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=456)

    U, S, Vh = decomp(T, axis=0, mode="SVD")

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
    
    U, S, Vh = decomp(T, axis=0, mode="SVD")
    
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

    # Use raw svd() to get singular values dict
    U, S_values, Vh = svd(T, axis=0)

    # Check singular values directly
    for key, s_array in S_values.items():
        # Check all positive
        assert np.all(s_array >= 0), "Singular values should be non-negative"
        # Check sorted in descending order
        assert np.allclose(s_array, np.sort(s_array)[::-1]), \
            "Singular values should be sorted descending"


def test_svd_bond_dimensions():
    """Test that bond dimensions are computed correctly."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5),))

    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=444)

    U, S, Vh = decomp(T, axis=0, mode="SVD")

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
    
    U, S, Vh = decomp(T, axis=0, mode="SVD")
    
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
    U, S, Vh = decomp(T, axis=1, mode="SVD")

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
        U, S, Vh = decomp(T, axis=axis, mode="SVD")
        
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


def test_svd_ambiguous_string_axis():
    """Test that SVD raises error for ambiguous string axis (duplicate itags)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))

    # Create tensor with duplicate itags
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "a"], seed=444)

    # Test ambiguous string axis
    with pytest.raises(ValueError, match="Ambiguous axis specification"):
        svd(T, axis="a")


# dtype tests

def test_svd_complex_dtype():
    """Test SVD with complex dtype."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 4),))
    
    T = Tensor.random([idx1, idx2], dtype=np.complex128, itags=["a", "b"], seed=5)
    
    # Use UR mode for efficient reconstruction
    U, R = decomp(T, axis=0, mode="UR")
    
    # U and R should be complex
    assert np.issubdtype(U.dtype, np.complexfloating)
    assert np.issubdtype(R.dtype, np.complexfloating)
    
    # Reconstruct using explicit pairs
    reconstructed = contract(U, R, pairs=[(1, 0)])
    
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / T.norm()
    assert rel_error < 1e-12


# Decomp function tests

def test_decomp_ur_mode():
    """Test decomp in UR mode."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=42)
    
    U, R = decomp(T, axis=0, mode="UR")
    
    # Check structure
    assert len(U.indices) == 2
    assert len(R.indices) == 2
    
    # Reconstruct using explicit pairs (bond indices have tags _bond_L and _bond_R)
    reconstructed = contract(U, R, pairs=[(1, 0)])
    
    # Verify accuracy
    diff_norm = (T - reconstructed).norm()
    assert diff_norm / T.norm() < 1e-12


def test_decomp_lv_mode():
    """Test decomp in LV mode."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=42)
    
    L, V = decomp(T, axis=0, mode="LV")
    
    # Check structure
    assert len(L.indices) == 2
    assert len(V.indices) == 2
    
    # Reconstruct using explicit pairs (bond indices have tags _bond_L and _bond_R)
    reconstructed = contract(L, V, pairs=[(1, 0)])
    
    # Verify accuracy
    diff_norm = (T - reconstructed).norm()
    assert diff_norm / T.norm() < 1e-12


def test_decomp_svd_mode():
    """Test decomp in SVD mode returns same as full SVD."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=42)
    
    U, S, Vh = decomp(T, axis=0, mode="SVD")
    
    # Check that S is a proper diagonal tensor
    assert isinstance(S, Tensor)
    assert len(S.indices) == 2
    
    # Verify S blocks are diagonal
    for key, block in S.data.items():
        assert block.ndim == 2
        assert block.shape[0] == block.shape[1]
        # Check it's diagonal (off-diagonal elements are zero)
        np.testing.assert_allclose(block - np.diag(np.diag(block)), 0, atol=1e-14)


def test_decomp_modes_equivalent():
    """Test that all decomp modes give equivalent reconstructions."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=99)
    
    # UR mode
    U_ur, R = decomp(T, axis=0, mode="UR")
    recon_ur = contract(U_ur, R, pairs=[(1, 0)])
    
    # SVD mode
    U_svd, S, Vh_svd = decomp(T, axis=0, mode="SVD")
    S_Vh = contract(S, Vh_svd, pairs=[(1, 0)])
    recon_svd = contract(U_svd, S_Vh, pairs=[(1, 0)])
    
    # LV mode
    L, V_lv = decomp(T, axis=0, mode="LV")
    recon_lv = contract(L, V_lv, pairs=[(1, 0)])
    
    # All reconstructions should be equivalent
    diff_ur_svd = (recon_ur - recon_svd).norm()
    diff_svd_lv = (recon_svd - recon_lv).norm()
    
    assert diff_ur_svd / T.norm() < 1e-12
    assert diff_svd_lv / T.norm() < 1e-12


def test_decomp_invalid_mode():
    """Test that invalid mode raises error."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=1)
    
    with pytest.raises(ValueError, match="Invalid mode"):
        decomp(T, axis=0, mode="invalid")


def test_svd_returns_dict():
    """Test that new svd returns dict of singular values."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=42)
    
    U, S_values, Vh = svd(T, axis=0)
    
    # Check S_values is a dict, not a Tensor
    assert isinstance(S_values, dict)
    assert not isinstance(S_values, Tensor)
    
    # Check all values are 1D arrays
    for key, s_array in S_values.items():
        assert isinstance(s_array, np.ndarray)
        assert s_array.ndim == 1
        assert len(s_array) > 0


def test_decomp_ur_multiindex():
    """Test UR mode with multi-index tensor."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=123)
    
    U, R = decomp(T, axis=1, mode="UR")
    
    # Check structure: U should have 2 indices, R should have 3 indices
    assert len(U.indices) == 2
    assert len(R.indices) == 3
    
    # Reconstruct using explicit pairs
    reconstructed = contract(U, R, pairs=[(1, 0)])
    
    # Permute back to original order (b, a, c) -> (a, b, c)
    reconstructed.permute([1, 0, 2])
    
    # Verify accuracy
    diff_norm = (T - reconstructed).norm()
    assert diff_norm / T.norm() < 1e-12


def test_decomp_lv_multiindex():
    """Test LV mode with multi-index tensor."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=456)
    
    L, V = decomp(T, axis=2, mode="LV")
    
    # Check structure: L should have 2 indices, V should have 3 indices
    assert len(L.indices) == 2
    assert len(V.indices) == 3
    
    # Reconstruct using explicit pairs
    reconstructed = contract(L, V, pairs=[(1, 0)])
    
    # Permute back to original order (c, a, b) -> (a, b, c)
    reconstructed.permute([1, 2, 0])
    
    # Verify accuracy
    diff_norm = (T - reconstructed).norm()
    assert diff_norm / T.norm() < 1e-12


def test_decomp_mode_case_insensitive():
    """Test that decomp mode is case insensitive."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=789)
    
    # Test lowercase
    U1, R1 = decomp(T, axis=0, mode="ur")
    
    # Test uppercase
    U2, R2 = decomp(T, axis=0, mode="UR")
    
    # Test mixed case
    U3, R3 = decomp(T, axis=0, mode="Ur")
    
    # All should give same results
    assert (U1 - U2).norm() < 1e-14
    assert (R1 - R2).norm() < 1e-14
    assert (U1 - U3).norm() < 1e-14
    assert (R1 - R3).norm() < 1e-14


def test_decomp_preserves_charge_neutrality():
    """Test that all decomp modes preserve charge neutrality."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=321)
    
    # Test all modes
    for mode in ["UR", "SVD", "LV"]:
        result = decomp(T, axis=0, mode=mode)
        
        # Check all output tensors are charge neutral
        for tensor in result:
            assert_charge_neutral(tensor)


def test_decomp_ur_efficiency():
    """Test that UR mode doesn't create unnecessary diagonal matrix."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 4),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=111)
    
    U, R = decomp(T, axis=0, mode="UR")
    
    # R should have singular values multiplied in
    # R blocks should not be square (they're rank x dim_right)
    for key, block in R.data.items():
        # R has shape (rank, dim_right) where rank <= min(dim_left, dim_right)
        # For this test: rank <= min(3, 4) = 3, dim_right = 4
        # So block should be (3, 4) which is not square
        assert block.shape[0] <= min(3, 4)
        # Verify it's been scaled by singular values (not all zeros)
        assert np.abs(block).max() > 1e-10


def test_decomp_lv_efficiency():
    """Test that LV mode doesn't create unnecessary diagonal matrix."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 4),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=222)
    
    L, V = decomp(T, axis=0, mode="LV")
    
    # L should have singular values multiplied in
    # L blocks should not be square (they're dim_left x rank)
    for key, block in L.data.items():
        # L has shape (dim_left, rank) where rank <= min(dim_left, dim_right)
        # For this test: rank <= min(3, 4) = 3, dim_left = 3
        # So block should be (3, 3) which happens to be square but scaled
        assert block.shape[1] <= min(3, 4)
        # Verify it's been scaled by singular values (not all zeros)
        assert np.abs(block).max() > 1e-10

