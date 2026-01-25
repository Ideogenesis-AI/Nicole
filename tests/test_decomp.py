# Copyright (C) 2025-2026 Changkai Zhang.
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
from nicole.decomp import svd, eig
from .utils import assert_charge_neutral


# Basic SVD tests

def test_svd_basic_reconstruction():
    """Test basic SVD and reconstruction on a simple tensor."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 3), Sector(2, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 1), Sector(1, 2), Sector(2, 1)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=42)
    original_norm = T.norm()

    # Perform SVD separating axis 0 from the rest
    U, S, Vh = decomp(T, axes=0, mode="SVD")

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
    
    U, S, Vh = decomp(T, axes=0, mode="SVD")
    
    assert len(U.indices) == 2
    assert len(S.indices) == 2
    assert len(Vh.indices) == 2


def test_svd_string_axis():
    """Test SVD with axis specified by string tag."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=123)

    # Perform SVD by axis name
    U, S, Vh = decomp(T, axes="b", mode="SVD")

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
        U, R = decomp(T, axes=axis, mode="UR")
        
        # Reconstruct using explicit pairs
        reconstructed = contract(U, R, axes=(1, 0))
        
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
    """Test SVD with multiple blocks and diverse charge sectors."""
    group = U1Group()
    
    # Create indices with diverse charge sectors
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 2), Sector(2, 1)))

    T = Tensor.random(indices=(idx1, idx2, idx3), itags=["a", "b", "c"], seed=42)

    # Perform SVD using decomp
    U, S, Vh = decomp(T, axes=0, mode="SVD")

    # Verify that we have blocks for multiple charges
    assert len(U.data) >= 2, "Should have blocks for multiple q_left charges"
    assert len(S.data) >= 2, "Should have blocks for multiple q_left charges"
    assert len(Vh.data) >= 2, "Should have multiple Vh blocks"

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
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=456)

    # Test with default flow "><"
    U, S, Vh = decomp(T, axes=0, mode="SVD")

    # With corrected logic, default "><" gives both S indices IN
    assert S.indices[0].direction == Direction.IN, "S first index should be IN for ><"
    assert S.indices[1].direction == Direction.IN, "S second index should be IN for ><"
    
    # Verify contraction works
    S_Vh = contract(S, Vh)
    reconstructed = contract(U, S_Vh)
    
    # Check that reconstructed has same index directions as original
    for i, idx in enumerate(reconstructed.indices):
        assert idx.direction == T.indices[i].direction


def test_svd_bond_index_structure():
    """Test that bond index has correct structure."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 3), Sector(2, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 2), Sector(2, 1)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=123)
    
    U, S, Vh = decomp(T, axes=0, mode="SVD")
    
    bond_index = U.indices[1]
    
    # Bond should have same group as left index
    assert bond_index.group == idx1.group
    
    # Bond charges should be a subset of left charges
    bond_charges = bond_index.charges()
    left_charges = idx1.charges()
    assert set(bond_charges).issubset(set(left_charges))
    
    # Bond should have at least one charge (non-trivial)
    assert len(bond_charges) > 0
    
    # With default flow="><", both U indices are OUT, so bond charges = -left_charges_in_data
    # (charge conservation: q_left + q_bond = 0, so q_bond = -q_left)
    left_charges_in_data = set(key[0] for key in T.data.keys())
    expected_bond_charges = {-q for q in left_charges_in_data}
    assert set(bond_charges) == expected_bond_charges, \
        f"Bond charges {set(bond_charges)} should equal negatives of left data charges {expected_bond_charges}"


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

    U, S, Vh = decomp(T, axes=0, mode="SVD")

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
    
    U, S, Vh = decomp(T, axes=0, mode="SVD")
    
    # Check that S blocks are diagonal
    for key, block in S.data.items():
        # Off-diagonal elements should be zero
        assert np.allclose(block, np.diag(np.diag(block)))


# Charge conservation tests

def test_svd_preserves_charge_conservation():
    """Test that SVD preserves charge conservation in all output tensors."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=222)
    
    # Original tensor should be charge neutral
    assert_charge_neutral(T)

    # Perform SVD
    U, S, Vh = decomp(T, axes=1, mode="SVD")

    # All output tensors should be charge neutral
    assert_charge_neutral(U)
    assert_charge_neutral(S)
    assert_charge_neutral(Vh)


def test_svd_charge_conservation_all_axes():
    """Test charge conservation for SVD on all axes."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 1), Sector(0, 2), Sector(1, 2), Sector(2, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 2), Sector(2, 1)))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=4)
    
    for axis in [0, 1, 2]:
        U, S, Vh = decomp(T, axes=axis, mode="SVD")
        
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
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 3), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-2, 1), Sector(-1, 2), Sector(0, 4), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2], dtype=np.complex128, itags=["a", "b"], seed=5)
    
    # Use UR mode for efficient reconstruction
    U, R = decomp(T, axes=0, mode="UR")
    
    # U and R should be complex
    assert np.issubdtype(U.dtype, np.complexfloating)
    assert np.issubdtype(R.dtype, np.complexfloating)
    
    # Reconstruct using explicit pairs
    reconstructed = contract(U, R, axes=(1, 0))
    
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / T.norm()
    assert rel_error < 1e-12


# Decomp function tests

def test_decomp_ur_mode():
    """Test decomp in UR mode."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=42)
    
    U, R = decomp(T, axes=0, mode="UR")
    
    # Check structure
    assert len(U.indices) == 2
    assert len(R.indices) == 2
    
    # Reconstruct using explicit pairs (bond indices have tags _bond_L and _bond_R)
    reconstructed = contract(U, R, axes=(1, 0))
    
    # Verify accuracy
    diff_norm = (T - reconstructed).norm()
    assert diff_norm / T.norm() < 1e-12


def test_decomp_lv_mode():
    """Test decomp in LV mode."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=42)
    
    L, V = decomp(T, axes=0, mode="LV")
    
    # Check structure
    assert len(L.indices) == 2
    assert len(V.indices) == 2
    
    # Reconstruct using explicit pairs (bond indices have tags _bond_L and _bond_R)
    reconstructed = contract(L, V, axes=(1, 0))
    
    # Verify accuracy
    diff_norm = (T - reconstructed).norm()
    assert diff_norm / T.norm() < 1e-12


def test_decomp_svd_mode():
    """Test decomp in SVD mode returns same as full SVD."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=42)
    
    U, S, Vh = decomp(T, axes=0, mode="SVD")
    
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
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 3), Sector(2, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-2, 1), Sector(-1, 1), Sector(0, 3), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=99)
    
    # UR mode
    U_ur, R = decomp(T, axes=0, mode="UR")
    recon_ur = contract(U_ur, R, axes=(1, 0))
    
    # SVD mode
    U_svd, S, Vh_svd = decomp(T, axes=0, mode="SVD")
    S_Vh = contract(S, Vh_svd, axes=(1, 0))
    recon_svd = contract(U_svd, S_Vh, axes=(1, 0))
    
    # LV mode
    L, V_lv = decomp(T, axes=0, mode="LV")
    recon_lv = contract(L, V_lv, axes=(1, 0))
    
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
        decomp(T, axes=0, mode="invalid")


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
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=123)
    
    U, R = decomp(T, axes=1, mode="UR")
    
    # Check structure: U should have 2 indices, R should have 3 indices
    assert len(U.indices) == 2
    assert len(R.indices) == 3
    
    # Reconstruct using explicit pairs
    reconstructed = contract(U, R, axes=(1, 0))
    
    # Permute back to original order (b, a, c) -> (a, b, c)
    reconstructed.permute([1, 0, 2])
    
    # Verify accuracy
    diff_norm = (T - reconstructed).norm()
    assert diff_norm / T.norm() < 1e-12


def test_decomp_lv_multiindex():
    """Test LV mode with multi-index tensor."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=456)
    
    L, V = decomp(T, axes=2, mode="LV")
    
    # Check structure: L should have 2 indices, V should have 3 indices
    assert len(L.indices) == 2
    assert len(V.indices) == 3
    
    # Reconstruct using explicit pairs
    reconstructed = contract(L, V, axes=(1, 0))
    
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
    U1, R1 = decomp(T, axes=0, mode="ur")
    
    # Test uppercase
    U2, R2 = decomp(T, axes=0, mode="UR")
    
    # Test mixed case
    U3, R3 = decomp(T, axes=0, mode="Ur")
    
    # All should give same results
    assert (U1 - U2).norm() < 1e-14
    assert (R1 - R2).norm() < 1e-14
    assert (U1 - U3).norm() < 1e-14
    assert (R1 - R3).norm() < 1e-14


def test_decomp_preserves_charge_neutrality():
    """Test that all decomp modes preserve charge neutrality."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 1), Sector(0, 2), Sector(1, 2), Sector(2, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 2), Sector(2, 1)))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=321)
    
    # Test all modes
    for mode in ["UR", "SVD", "LV"]:
        result = decomp(T, axes=0, mode=mode)
        
        # Check all output tensors are charge neutral
        for tensor in result:
            assert_charge_neutral(tensor)


def test_decomp_ur_efficiency():
    """Test that UR mode doesn't create unnecessary diagonal matrix."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 4),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=111)
    
    U, R = decomp(T, axes=0, mode="UR")
    
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
    
    L, V = decomp(T, axes=0, mode="LV")
    
    # L should have singular values multiplied in
    # L blocks should not be square (they're dim_left x rank)
    for key, block in L.data.items():
        # L has shape (dim_left, rank) where rank <= min(dim_left, dim_right)
        # For this test: rank <= min(3, 4) = 3, dim_left = 3
        # So block should be (3, 3) which happens to be square but scaled
        assert block.shape[1] <= min(3, 4)
        # Verify it's been scaled by singular values (not all zeros)
        assert np.abs(block).max() > 1e-10


# Truncation tests

def test_svd_truncation_nkeep():
    """Test SVD with nkeep (keep at most N singular values globally)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 10),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 10),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=100)
    
    # Perform SVD with nkeep to keep at most 5 singular values globally
    U, S_blocks, Vh = svd(T, axis=0, trunc={"nkeep": 5})
    
    # Check that total number of kept singular values is at most 5
    total_kept = sum(len(s_array) for s_array in S_blocks.values())
    assert total_kept <= 5, f"Expected at most 5 singular values, got {total_kept}"
    
    # Check bond dimension
    bond_index = U.indices[1]
    assert bond_index.dim == total_kept


def test_svd_truncation_thresh():
    """Test SVD with thresh (keep singular values >= threshold per block)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 10),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 10),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=200)
    
    # Perform SVD with threshold truncation (use higher threshold to ensure truncation)
    threshold = 0.5
    U, S_blocks, Vh = svd(T, axis=0, trunc={"thresh": threshold})
    
    # Check that all kept singular values are >= threshold
    for key, s_array in S_blocks.items():
        assert np.all(s_array >= threshold), f"Block {key} has singular values < {threshold}"
    
    # Verify truncation happened (should have fewer than 10 singular values)
    total_kept = sum(len(s_array) for s_array in S_blocks.values())
    assert total_kept < 10, "Expected truncation to reduce number of singular values"


def test_decomp_truncation_ur_mode():
    """Test decomp with nkeep truncation in UR mode."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 8),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 8),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=300)
    
    # Decomp with nkeep truncation
    U, R = decomp(T, axes=0, mode="UR", trunc={"nkeep": 4})
    
    # Check bond dimension
    bond_index = U.indices[1]
    assert bond_index.dim <= 4
    
    # Verify reconstruction is approximate (not exact due to truncation)
    reconstructed = contract(U, R, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    
    # Error should be non-zero (truncation loses information)
    # but not too large (we kept significant singular values)
    assert rel_error > 1e-10, "Expected non-zero error due to truncation"
    assert rel_error < 0.5, "Truncation error too large"


def test_decomp_truncation_svd_mode():
    """Test decomp with nkeep truncation in SVD mode."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 6),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 6),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=400)
    
    # Decomp with nkeep truncation
    U, S, Vh = decomp(T, axes=0, mode="SVD", trunc={"nkeep": 3})
    
    # Check that total singular values is at most 3
    total_svs = sum(block.shape[0] for block in S.data.values())
    assert total_svs <= 3


def test_svd_truncation_no_truncation():
    """Test that no truncation parameters performs no truncation."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 5),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 7),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=500)
    
    # SVD without truncation
    U, S_blocks, Vh = svd(T, axis=0)
    
    # Should have min(5, 7) = 5 singular values
    total_kept = sum(len(s_array) for s_array in S_blocks.values())
    assert total_kept == 5


def test_svd_truncation_multiblock():
    """Test global truncation with multiple charge blocks."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-2, 3), Sector(-1, 4), Sector(0, 6), Sector(1, 4), Sector(2, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 3), Sector(0, 5), Sector(1, 3), Sector(2, 2)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=600)
    
    # Truncate to 3 singular values globally (across all blocks)
    U, S_blocks, Vh = svd(T, axis=0, trunc={"nkeep": 3})
    
    # Total should be at most 3 singular values across all blocks
    total_kept = sum(len(s_array) for s_array in S_blocks.values())
    assert total_kept <= 3, f"Expected at most 3 singular values globally, got {total_kept}"


def test_svd_truncation_invalid_mode():
    """Test that invalid truncation mode raises error."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 5),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=700)
    
    # Invalid truncation format (not a dict)
    with pytest.raises(ValueError, match="trunc must be a dict"):
        svd(T, axis=0, trunc=("nkeep", 3))
    
    # Invalid truncation mode key
    with pytest.raises(ValueError, match="Invalid truncation mode"):
        svd(T, axis=0, trunc={"invalid_mode": 3})


def test_svd_truncation_combined_thresh_nkeep():
    """Test SVD with both thresh and nkeep truncation modes."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 5), Sector(1, 4)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 6), Sector(1, 5)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=999)
    
    # Apply both truncations: first thresh, then nkeep
    U, S_blocks, Vh = svd(T, axis=0, trunc={"thresh": 0.5, "nkeep": 3})
    
    # Count total singular values
    total_sv = sum(len(s) for s in S_blocks.values())
    
    # Should have at most 3 singular values (nkeep limit)
    assert total_sv <= 3
    
    # All singular values should be >= 0.5 (thresh limit)
    for s_array in S_blocks.values():
        assert np.all(s_array >= 0.5)
    
    # Verify we got exactly 3 (both constraints satisfied)
    assert total_sv == 3


# High-order tensor tests

def test_decomp_4index_tensor():
    """Test decomp on 4-index tensor with reconstruction."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx4 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2, idx3, idx4], itags=["a", "b", "c", "d"], seed=800)
    original_norm = T.norm()
    
    # Test decomposition on different axes
    for axis in [0, 1, 2, 3]:
        U, R = decomp(T, axes=axis, mode="UR")
        
        # Reconstruct
        reconstructed = contract(U, R, axes=(1, 0))
        
        # Permute back to original order
        if axis == 0:
            pass  # (a, b, c, d)
        elif axis == 1:
            reconstructed.permute([1, 0, 2, 3])  # (b, a, c, d) -> (a, b, c, d)
        elif axis == 2:
            reconstructed.permute([1, 2, 0, 3])  # (c, a, b, d) -> (a, b, c, d)
        elif axis == 3:
            reconstructed.permute([1, 2, 3, 0])  # (d, a, b, c) -> (a, b, c, d)
        
        # Verify accuracy
        rel_error = (T - reconstructed).norm() / original_norm
        assert rel_error < 1e-12, f"Reconstruction failed for axis {axis}"


def test_decomp_5index_tensor():
    """Test decomp on 5-index tensor."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d", "e"], seed=900)
    
    # Test SVD mode on middle axis
    U, S, Vh = decomp(T, axes=2, mode="SVD")
    
    # Check structure
    assert len(U.indices) == 2  # (c, bond)
    assert len(S.indices) == 2  # (bond.flip(), bond)
    assert len(Vh.indices) == 5  # (bond.flip(), a, b, d, e)
    
    # Verify charge conservation
    assert_charge_neutral(U)
    assert_charge_neutral(S)
    assert_charge_neutral(Vh)
    
    # Reconstruct
    S_Vh = contract(S, Vh, axes=(1, 0))
    reconstructed = contract(U, S_Vh, axes=(1, 0))
    
    # Permute: (c, a, b, d, e) -> (a, b, c, d, e)
    reconstructed.permute([1, 2, 0, 3, 4])
    
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12


def test_decomp_6index_tensor_with_truncation():
    """Test decomp on 6-index tensor with truncation."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 3),)),
        Index(Direction.IN, group, sectors=(Sector(0, 3),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 3),)),
        Index(Direction.IN, group, sectors=(Sector(0, 3),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 3),)),
        Index(Direction.IN, group, sectors=(Sector(0, 3),))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d", "e", "f"], seed=1000)
    
    # Decompose with truncation (separate first index from rest)
    U, R = decomp(T, axes=0, mode="UR", trunc={"nkeep": 5})
    
    # Check that truncation worked
    bond_dim = U.indices[1].dim
    assert bond_dim <= 5
    
    # Reconstruct (will be approximate due to truncation)
    reconstructed = contract(U, R, axes=(1, 0))
    
    # Verify dimensions
    assert len(reconstructed.indices) == 6
    for i, idx in enumerate(reconstructed.indices):
        assert idx.dim == T.indices[i].dim


def test_high_order_tensor_multiple_charges():
    """Test high-order tensor with multiple charge blocks."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 2))),
        Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 1))),
        Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(-2, 1), Sector(0, 2), Sector(2, 1)))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d"], seed=1100)
    
    # Test LV mode
    L, V = decomp(T, axes=0, mode="LV")
    
    # Check that we have multiple charge sectors
    bond_charges = set(L.indices[1].charges())
    assert len(bond_charges) > 1, "Should have multiple charge sectors"
    
    # Reconstruct
    reconstructed = contract(L, V, axes=(1, 0))
    
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12


def test_high_order_tensor_different_axis_sizes():
    """Test high-order tensor with varying axis dimensions."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 5),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 3),)),
        Index(Direction.IN, group, sectors=(Sector(0, 4),))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d"], seed=1200)
    
    # Decompose on axis 1 (separates axis 1 from axes 0,2,3)
    U, S_blocks, Vh = svd(T, axis=1)
    
    # Bond dimension should be min(dim_axis1, dim_others)
    # dim_axis1 = 5, dim_others = 2*3*4 = 24
    # So bond_dim = min(5, 24) = 5
    total_bond_dim = sum(len(s) for s in S_blocks.values())
    assert total_bond_dim == 5
    
    # Verify singular values are sorted
    for key, s_array in S_blocks.items():
        assert np.all(s_array[:-1] >= s_array[1:]), "Singular values should be sorted descending"


def test_high_order_tensor_all_modes():
    """Test all decomp modes on high-order tensor."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d"], seed=1300)
    
    # Test all three modes give equivalent results
    U_ur, R = decomp(T, axes=1, mode="UR")
    recon_ur = contract(U_ur, R, axes=(1, 0))
    
    U_svd, S, Vh_svd = decomp(T, axes=1, mode="SVD")
    S_Vh = contract(S, Vh_svd, axes=(1, 0))
    recon_svd = contract(U_svd, S_Vh, axes=(1, 0))
    
    L, V_lv = decomp(T, axes=1, mode="LV")
    recon_lv = contract(L, V_lv, axes=(1, 0))
    
    # All reconstructions should match (after permuting to same order)
    # Current order is (b, a, c, d), need (a, b, c, d)
    recon_ur.permute([1, 0, 2, 3])
    recon_svd.permute([1, 0, 2, 3])
    recon_lv.permute([1, 0, 2, 3])
    
    assert (recon_ur - recon_svd).norm() / T.norm() < 1e-12
    assert (recon_svd - recon_lv).norm() / T.norm() < 1e-12


def test_high_order_tensor_bond_structure():
    """Test bond index structure in high-order tensor decomposition."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d"], seed=1400)
    
    U, S_blocks, Vh = svd(T, axis=0)
    
    bond_index = U.indices[1]
    
    # Bond should have charges that appear in the tensor's first index
    left_charges_in_data = set(key[0] for key in T.data.keys())
    bond_charges = set(bond_index.charges())
    
    assert bond_charges == left_charges_in_data
    
    # Bond should have correct group
    assert bond_index.group == indices[0].group
    
    # Bond direction should be opposite of left index
    assert bond_index.direction == indices[0].direction.reverse()


def test_high_order_tensor_thresh_truncation():
    """Test threshold truncation on high-order tensor."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(-1, 3), Sector(0, 4), Sector(1, 3))),
        Index(Direction.IN, group, sectors=(Sector(-2, 2), Sector(-1, 2), Sector(0, 4), Sector(1, 2))),
        Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 4), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 4), Sector(1, 2)))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d"], seed=1500)
    
    # Apply threshold truncation
    threshold = 1.0
    U, S_blocks, Vh = svd(T, axis=0, trunc={"thresh": threshold})
    
    # All kept singular values should be >= threshold
    for key, s_array in S_blocks.items():
        assert np.all(s_array >= threshold)
    
    # Verify we can still reconstruct (approximately)
    U_full = decomp(T, axes=0, mode="UR", trunc={"thresh": threshold})[0]
    assert len(U_full.indices) == 2
    # Left index should be unchanged (sum of all sector dimensions)
    expected_left_dim = sum(s.dim for s in indices[0].sectors)
    assert U_full.indices[0].dim == expected_left_dim


# Flow parameter tests

def test_decomp_flow_svd_default():
    """Test SVD mode with default flow ><."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=42)
    
    # Default flow should be "><" (both arrows incoming)
    U, S, Vh = decomp(T, axes=0, mode="SVD")

    # Default "><" flow produces S with (IN, IN)
    assert S.indices[0].direction == Direction.IN
    assert S.indices[1].direction == Direction.IN
    
    # Verify reconstruction
    S_Vh = contract(S, Vh, axes=(1, 0))
    reconstructed = contract(U, S_Vh, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12


def test_decomp_flow_svd_outward():
    """Test SVD mode with flow >> (both arrows outward)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=43)
    
    U, S, Vh = decomp(T, axes=0, mode="SVD", flow=">>")
    
    # ">>" flow produces S with (IN, OUT)
    assert S.indices[0].direction == Direction.IN
    assert S.indices[1].direction == Direction.OUT
    
    # Verify reconstruction
    S_Vh = contract(S, Vh, axes=(1, 0))
    reconstructed = contract(U, S_Vh, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12


def test_decomp_flow_svd_inward():
    """Test SVD mode with flow << (both arrows inward)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=44)
    
    U, S, Vh = decomp(T, axes=0, mode="SVD", flow="<<")
    
    # "<<" flow produces S with (OUT, IN)
    assert S.indices[0].direction == Direction.OUT
    assert S.indices[1].direction == Direction.IN
    
    # Verify reconstruction
    S_Vh = contract(S, Vh, axes=(1, 0))
    reconstructed = contract(U, S_Vh, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12


def test_decomp_flow_svd_with_in_index():
    """Test SVD mode with left_index IN (natural flow is <<)."""
    group = U1Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=45)
    
    # Test all three flows - same results as OUT left_index
    U_default, S_default, Vh_default = decomp(T, axes=0, mode="SVD")
    assert S_default.indices[0].direction == Direction.IN
    assert S_default.indices[1].direction == Direction.IN
    
    U_out, S_out, Vh_out = decomp(T, axes=0, mode="SVD", flow=">>")
    assert S_out.indices[0].direction == Direction.IN
    assert S_out.indices[1].direction == Direction.OUT
    
    U_in, S_in, Vh_in = decomp(T, axes=0, mode="SVD", flow="<<")
    assert S_in.indices[0].direction == Direction.OUT
    assert S_in.indices[1].direction == Direction.IN
    
    # All should reconstruct correctly
    for U, S, Vh in [(U_default, S_default, Vh_default), 
                      (U_out, S_out, Vh_out), 
                      (U_in, S_in, Vh_in)]:
        S_Vh = contract(S, Vh, axes=(1, 0))
        reconstructed = contract(U, S_Vh, axes=(1, 0))
        rel_error = (T - reconstructed).norm() / T.norm()
        assert rel_error < 1e-12


def test_decomp_flow_ur_mode_default():
    """Test UR mode with default flow (should normalize to >>)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=46)
    
    # Default flow "><" should normalize to ">>" for UR mode
    U, R = decomp(T, axes=0, mode="UR")
    
    # UR mode with "><" and ">>" both produce (OUT, IN) bonds
    assert U.indices[1].direction == Direction.OUT
    assert R.indices[0].direction == Direction.IN
    
    # Verify reconstruction
    reconstructed = contract(U, R, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12


def test_decomp_flow_ur_mode_explicit():
    """Test UR mode with explicit flow values."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=47)
    
    # Test with ">>" and "><" flow (both normalize to >> for UR mode)
    U_out, R_out = decomp(T, axes=0, mode="UR", flow=">>")
    assert U_out.indices[1].direction == Direction.OUT
    assert R_out.indices[0].direction == Direction.IN
    
    # Test with "<<" flow (different from >>)
    U_in, R_in = decomp(T, axes=0, mode="UR", flow="<<")
    assert U_in.indices[1].direction == Direction.IN
    assert R_in.indices[0].direction == Direction.OUT
    
    # Both should reconstruct correctly
    recon_out = contract(U_out, R_out, axes=(1, 0))
    recon_in = contract(U_in, R_in, axes=(1, 0))
    
    assert (T - recon_out).norm() / T.norm() < 1e-12
    assert (T - recon_in).norm() / T.norm() < 1e-12


def test_decomp_flow_ur_mode_in_index():
    """Test UR mode with left_index IN (natural flow is <<)."""
    group = U1Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=48)
    
    # Default flow "><" normalizes to ">>" for UR mode
    U, R = decomp(T, axes=0, mode="UR")
    assert U.indices[1].direction == Direction.OUT
    assert R.indices[0].direction == Direction.IN
    
    # Verify reconstruction
    reconstructed = contract(U, R, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12


def test_decomp_flow_lv_mode_default():
    """Test LV mode with default flow (should normalize to <<)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=49)
    
    # Default flow "><" normalizes to "<<" for LV mode
    L, V = decomp(T, axes=0, mode="LV")
    
    # LV mode with "><" and "<<" both produce (IN, OUT) bonds
    assert L.indices[1].direction == Direction.IN
    assert V.indices[0].direction == Direction.OUT
    
    # Verify reconstruction
    reconstructed = contract(L, V, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12


def test_decomp_flow_lv_mode_explicit():
    """Test LV mode with explicit flow values."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=50)
    
    # Test with "<<" and "><" flow (both normalize to << for LV mode)
    L_in, V_in = decomp(T, axes=0, mode="LV", flow="<<")
    assert L_in.indices[1].direction == Direction.IN
    assert V_in.indices[0].direction == Direction.OUT
    
    # Test with ">>" flow (different from <<)
    L_out, V_out = decomp(T, axes=0, mode="LV", flow=">>")
    assert L_out.indices[1].direction == Direction.OUT
    assert V_out.indices[0].direction == Direction.IN
    
    # Both should reconstruct correctly
    recon_in = contract(L_in, V_in, axes=(1, 0))
    recon_out = contract(L_out, V_out, axes=(1, 0))
    
    assert (T - recon_in).norm() / T.norm() < 1e-12
    assert (T - recon_out).norm() / T.norm() < 1e-12


def test_decomp_flow_lv_mode_in_index():
    """Test LV mode with left_index IN (natural flow is <<)."""
    group = U1Group()
    idx1 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=51)
    
    # Default flow "><" normalizes to "<<"
    L, V = decomp(T, axes=0, mode="LV")
    assert L.indices[1].direction == Direction.IN
    assert V.indices[0].direction == Direction.OUT
    
    # Verify reconstruction
    reconstructed = contract(L, V, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12


def test_decomp_flow_multiindex_svd():
    """Test flow parameter with multi-index tensor in SVD mode."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c"], seed=52)
    
    # Test all three flows
    for flow in ["><", ">>", "<<"]:
        U, S, Vh = decomp(T, axes=0, mode="SVD", flow=flow)
        
        # Verify reconstruction
        S_Vh = contract(S, Vh, axes=(1, 0))
        reconstructed = contract(U, S_Vh, axes=(1, 0))
        rel_error = (T - reconstructed).norm() / T.norm()
        assert rel_error < 1e-12


def test_decomp_flow_multiindex_ur_lv():
    """Test flow parameter with multi-index tensor in UR and LV modes."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c"], seed=53)
    
    # Test UR mode with different flows
    for flow in ["><", ">>", "<<"]:
        U, R = decomp(T, axes=1, mode="UR", flow=flow)
        reconstructed = contract(U, R, axes=(1, 0))
        # Permute reconstructed to match T's index order (b, a, c) -> (a, b, c)
        reconstructed.permute([1, 0, 2])
        rel_error = (T - reconstructed).norm() / T.norm()
        assert rel_error < 1e-12
    
    # Test LV mode with different flows
    for flow in ["><", ">>", "<<"]:
        L, V = decomp(T, axes=1, mode="LV", flow=flow)
        reconstructed = contract(L, V, axes=(1, 0))
        # Permute reconstructed to match T's index order (b, a, c) -> (a, b, c)
        reconstructed.permute([1, 0, 2])
        rel_error = (T - reconstructed).norm() / T.norm()
        assert rel_error < 1e-12


def test_decomp_flow_invalid():
    """Test that invalid flow values raise errors."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=54)
    
    with pytest.raises(ValueError, match="Invalid flow"):
        decomp(T, axes=0, mode="SVD", flow="<>")
    
    with pytest.raises(ValueError, match="Invalid flow"):
        decomp(T, axes=0, mode="UR", flow="->")


def test_decomp_flow_charge_conservation():
    """Test that flow parameter preserves charge conservation."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 3), Sector(2, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 3), Sector(2, 1)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=55)
    
    # Test all modes and flows preserve charge neutrality
    for mode in ["SVD", "UR", "LV"]:
        for flow in ["><", ">>", "<<"]:
            result = decomp(T, axes=0, mode=mode, flow=flow)
            
            # Check charge neutrality of all output tensors
            for tensor in result:
                assert_charge_neutral(tensor)


# Itag parameter tests

def test_decomp_itag_svd_single_string():
    """Test SVD mode with single string itag (both bonds use same tag)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=56)
    
    U, S, Vh = decomp(T, axes=0, mode="SVD", itag="bond")
    
    # Both U and Vh should have "bond" as their bond tag
    assert U.itags[1] == "bond"
    assert Vh.itags[0] == "bond"
    assert S.itags[0] == "bond"
    assert S.itags[1] == "bond"
    
    # Verify reconstruction (need explicit pairs since both bonds have same tag)
    S_Vh = contract(S, Vh, axes=(1, 0))
    reconstructed = contract(U, S_Vh, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12


def test_decomp_itag_svd_tuple():
    """Test SVD mode with tuple itag (different tags for left and right)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=57)
    
    U, S, Vh = decomp(T, axes=0, mode="SVD", itag=("bond_u", "bond_vh"))
    
    # U should have "bond_u", Vh should have "bond_vh"
    assert U.itags[1] == "bond_u"
    assert Vh.itags[0] == "bond_vh"
    assert S.itags[0] == "bond_u"
    assert S.itags[1] == "bond_vh"
    
    # Verify reconstruction with integer pairs
    S_Vh = contract(S, Vh, axes=(1, 0))
    reconstructed = contract(U, S_Vh, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12


def test_decomp_itag_ur_mode():
    """Test UR mode with custom itag."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=58)
    
    # Test with single string itag
    U, R = decomp(T, axes=0, mode="UR", itag="k")
    assert U.itags[1] == "k"
    assert R.itags[0] == "k"
    
    # Verify reconstruction (explicit pairs needed since both have same tag)
    reconstructed = contract(U, R, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12
    
    # Test with tuple itag
    U2, R2 = decomp(T, axes=0, mode="UR", itag=("i", "j"))
    assert U2.itags[1] == "i"
    assert R2.itags[0] == "i"  # R uses left tag for bond
    
    # Verify reconstruction
    reconstructed2 = contract(U2, R2, axes=(1, 0))
    rel_error2 = (T - reconstructed2).norm() / T.norm()
    assert rel_error2 < 1e-12


def test_decomp_itag_lv_mode():
    """Test LV mode with custom itag."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=59)
    
    # Test with single string itag
    L, V = decomp(T, axes=0, mode="LV", itag="m")
    assert L.itags[1] == "m"
    assert V.itags[0] == "m"
    
    # Verify reconstruction (explicit pairs needed since both have same tag)
    reconstructed = contract(L, V, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12
    
    # Test with tuple itag
    L2, V2 = decomp(T, axes=0, mode="LV", itag=("p", "q"))
    assert L2.itags[1] == "q"  # L uses right tag for bond
    assert V2.itags[0] == "q"
    
    # Verify reconstruction
    reconstructed2 = contract(L2, V2, axes=(1, 0))
    rel_error2 = (T - reconstructed2).norm() / T.norm()
    assert rel_error2 < 1e-12


def test_decomp_itag_multiindex():
    """Test itag parameter with multi-index tensor."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c"], seed=60)
    
    # Test SVD mode with custom tags
    # Decomposing on axis 1 ("b") separates "b" from "a" and "c"
    U, S, Vh = decomp(T, axes=1, mode="SVD", itag=("left", "right"))
    assert U.itags == ("b", "left")  # U has (separated_index, bond)
    assert S.itags == ("left", "right")
    assert Vh.itags == ("right", "a", "c")  # Vh has (bond, *rest)
    
    # Verify reconstruction with integer pairs
    S_Vh = contract(S, Vh, axes=(1, 0))
    reconstructed = contract(U, S_Vh, axes=(1, 0))
    reconstructed.permute([1, 0, 2])  # Reorder (b, a, c) to (a, b, c)
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12


def test_decomp_itag_invalid():
    """Test that invalid itag values raise errors."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=61)
    
    # Invalid itag: tuple with wrong length
    with pytest.raises(ValueError, match="itag must be"):
        decomp(T, axes=0, mode="SVD", itag=("a", "b", "c"))
    
    # Invalid itag: wrong type
    with pytest.raises(ValueError, match="itag must be"):
        decomp(T, axes=0, mode="SVD", itag=123)


# Eigenvalue decomposition tests

def test_eig_basic():
    """Test basic eigenvalue decomposition."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 1)))
    idx_in = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 1)))
    
    # Create a symmetric matrix for real eigenvalues
    T = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=62)
    # Make it Hermitian by averaging with its transpose
    T_data_sym = {}
    for key, arr in T.data.items():
        T_data_sym[key] = (arr + arr.T.conj()) / 2
    T = Tensor(
        indices=(idx_out, idx_in),
        itags=("i", "j"),
        data=T_data_sym,
        dtype=T.dtype
    )
    
    U, D = eig(T)
    
    # Check dimensions
    assert len(U.indices) == 2
    assert U.itags[0] == "i"
    assert U.itags[1] == "_bond_eig"
    
    # Check eigenvalues are real for Hermitian matrix
    for key, eigvals in D.items():
        assert np.allclose(eigvals.imag, 0, atol=1e-10)
    
    # Verify eigendecomposition: T @ U = U @ diag(D) for each block
    for key in T.data.keys():
        q_row, q_col = key
        if q_row != q_col:
            continue  # Skip off-diagonal blocks
        
        T_block = T.data[key]
        U_block = U.data[(q_row, q_row)]
        D_block = D[(q_row, q_row)]
        
        # T @ U
        T_U = T_block @ U_block
        # U @ diag(D)
        U_D = U_block @ np.diag(D_block)
        
        assert np.allclose(T_U, U_D, atol=1e-10)


def test_eig_reconstruction():
    """Test that eigendecomposition can reconstruct the original matrix."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 2), Sector(0, 3), Sector(1, 2), Sector(2, 1)))
    idx_in = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 3), Sector(1, 2)))
    
    # Create a symmetric matrix
    T = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=63)
    T_data_sym = {}
    for key, arr in T.data.items():
        T_data_sym[key] = (arr + arr.T.conj()) / 2
    T = Tensor(
        indices=(idx_out, idx_in),
        itags=("i", "j"),
        data=T_data_sym,
        dtype=T.dtype
    )
    
    U, D = eig(T)
    
    # Verify eigendecomposition block-by-block
    # T @ U = U @ diag(D)
    for key in T.data.keys():
        q_row, q_col = key
        if q_row != q_col:
            continue
        
        T_block = T.data[key]
        U_block = U.data[(q_row, q_row)]
        D_block = D[(q_row, q_row)]
        
        # Reconstruct T from eigendecomposition: T = U @ diag(D) @ U^{-1}
        # For symmetric matrices, U is orthogonal: U^{-1} = U^T
        D_diag = np.diag(D_block)
        T_reconstructed = U_block @ D_diag @ U_block.T.conj()
        
        rel_error = np.linalg.norm(T_block - T_reconstructed) / np.linalg.norm(T_block)
        assert rel_error < 1e-10


def test_eig_truncation_nkeep():
    """Test eigenvalue decomposition with nkeep truncation."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(-2, 2), Sector(-1, 2), Sector(0, 4), Sector(1, 3), Sector(2, 2)))
    idx_in = Index(Direction.IN, group, sectors=(Sector(-2, 2), Sector(-1, 2), Sector(0, 4), Sector(1, 3), Sector(2, 2)))
    
    T = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=64)
    T_data_sym = {}
    for key, arr in T.data.items():
        T_data_sym[key] = (arr + arr.T.conj()) / 2
    T = Tensor(
        indices=(idx_out, idx_in),
        itags=("i", "j"),
        data=T_data_sym,
        dtype=T.dtype
    )
    
    # Keep only top 3 eigenvalues
    U, D = eig(T, trunc={"nkeep": 3})
    
    # Count total eigenvalues
    total_eigvals = sum(len(eigvals) for eigvals in D.values())
    assert total_eigvals == 3
    
    # Verify they are the largest magnitude ones
    all_eigvals_full = []
    for key in T.data.keys():
        q_row, q_col = key
        if q_row != q_col:
            continue
        eigvals_full, _ = np.linalg.eig(T.data[key])
        all_eigvals_full.extend(np.abs(eigvals_full))
    
    all_eigvals_full.sort(reverse=True)
    top_3_expected = all_eigvals_full[:3]
    
    all_eigvals_truncated = []
    for eigvals in D.values():
        all_eigvals_truncated.extend(np.abs(eigvals))
    
    all_eigvals_truncated.sort(reverse=True)
    
    assert np.allclose(all_eigvals_truncated, top_3_expected, atol=1e-10)


def test_eig_truncation_thresh():
    """Test eigenvalue decomposition with threshold truncation."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 3), Sector(1, 3), Sector(2, 2)))
    idx_in = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 3), Sector(1, 3), Sector(2, 2)))
    
    T = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=65)
    T_data_sym = {}
    for key, arr in T.data.items():
        T_data_sym[key] = (arr + arr.T.conj()) / 2
    T = Tensor(
        indices=(idx_out, idx_in),
        itags=("i", "j"),
        data=T_data_sym,
        dtype=T.dtype
    )
    
    # Keep eigenvalues with |λ| >= 1.0
    U, D = eig(T, trunc={"thresh": 1.0})
    
    # Verify all kept eigenvalues satisfy threshold
    for eigvals in D.values():
        assert np.all(np.abs(eigvals) >= 1.0)


def test_eig_truncation_combined_thresh_nkeep():
    """Test eigenvalue decomposition with both thresh and nkeep truncation."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(-1, 3), Sector(0, 4), Sector(1, 3)))
    idx_in = Index(Direction.IN, group, sectors=(Sector(-1, 3), Sector(0, 4), Sector(1, 3)))
    
    # Create symmetric tensor
    T = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=888)
    T_data_sym = {}
    for key, arr in T.data.items():
        T_data_sym[key] = (arr + arr.T.conj()) / 2
    T = Tensor(
        indices=(idx_out, idx_in),
        itags=("i", "j"),
        data=T_data_sym,
        dtype=T.dtype
    )
    
    # Apply both truncations: first thresh >= 0.8, then nkeep top 4
    U, D = eig(T, trunc={"thresh": 0.8, "nkeep": 4})
    
    # Count total eigenvalues
    total_eig = sum(len(eigvals) for eigvals in D.values())
    
    # Should have at most 4 eigenvalues (nkeep limit)
    assert total_eig <= 4
    
    # All eigenvalues should have |λ| >= 0.8 (thresh limit)
    for eigvals in D.values():
        assert np.all(np.abs(eigvals) >= 0.8)


def test_eig_non_square_error():
    """Test that eig raises error for non-square matrices."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))  # Different size
    
    T = Tensor.random([idx1, idx2], itags=["i", "j"], seed=66)
    
    # This should work (dimensions can differ as long as charge structure matches)
    # But let's test with mismatched directions
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    T2 = Tensor.random([idx1, idx3], itags=["i", "j"], seed=67)
    
    with pytest.raises(ValueError, match="opposite directions"):
        eig(T2)


def test_eig_non_2d_error():
    """Test that eig raises error for non-2D tensors."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    ]
    
    T = Tensor.random(indices, itags=["i", "j", "k"], seed=68)
    
    with pytest.raises(ValueError, match="square matrix"):
        eig(T)


def test_eig_complex_matrix():
    """Test eigenvalue decomposition of complex matrix."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    # Create a complex matrix
    data = {(0, 0): np.array([[1+1j, 2-1j], [2+1j, 3-2j]], dtype=np.complex128)}
    T = Tensor(
        indices=(idx_out, idx_in),
        itags=("i", "j"),
        data=data,
        dtype=np.complex128
    )
    
    U, D = eig(T)
    
    # Verify eigendecomposition
    T_block = T.data[(0, 0)]
    U_block = U.data[(0, 0)]
    D_block = D[(0, 0)]
    
    # T @ U = U @ diag(D)
    T_U = T_block @ U_block
    U_D = U_block @ np.diag(D_block)
    
    assert np.allclose(T_U, U_D, atol=1e-10)


def test_eig_charge_conservation():
    """Test that eigenvalue decomposition preserves charge structure."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 3), Sector(2, 2)))
    idx_in = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 3), Sector(2, 2)))
    
    T = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=69)
    
    U, D = eig(T)
    
    # Check that U is charge-neutral
    for key in U.data.keys():
        q_row, q_bond = key
        # For charge neutrality: q_row + q_bond = 0 (considering directions)
        # Since U has (OUT, IN) structure, we expect q_row == q_bond
        assert q_row == q_bond
    
    # Check that D has same structure
    for key in D.keys():
        q_left, q_right = key
        assert q_left == q_right


def test_eig_itag():
    """Test that itag parameter customizes the bond index tag."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=70)
    
    # Test with custom itag
    U, D = eig(T, itag="eig_bond")
    
    assert U.itags[0] == "i"
    assert U.itags[1] == "eig_bond"
    
    # Test with default itag
    U_default, D_default = eig(T)
    assert U_default.itags[1] == "_bond_eig"


# Multi-axis decomposition tests

def test_decomp_multi_axis_svd():
    """Test decomp with multiple axes in SVD mode."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(-1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 1)))
    idx4 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2, idx3, idx4], seed=42, itags=['a', 'b', 'c', 'd'])
    
    # Decompose on multiple axes
    U, S, Vh = decomp(T, axes=['a', 'b', 'c'], mode='SVD')
    
    # U should have the 3 original axes plus bond
    assert len(U.indices) == 4
    assert set(['a', 'b', 'c']).issubset(set(U.itags))
    
    # Vh should have bond and remaining axis
    assert len(Vh.indices) == 2
    assert 'd' in Vh.itags
    
    # S should have 2 indices
    assert len(S.indices) == 2
    
    # Check charge neutrality
    assert_charge_neutral(U)
    assert_charge_neutral(S)
    assert_charge_neutral(Vh)


def test_decomp_multi_axis_ur():
    """Test decomp with multiple axes in UR mode."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2, idx3], seed=1, itags=['a', 'b', 'c'])
    
    # Decompose first two axes
    U, R = decomp(T, axes=[0, 1], mode='UR')
    
    # U should have 2 original axes plus bond
    assert len(U.indices) == 3
    assert 'a' in U.itags and 'b' in U.itags
    
    # R should have bond and remaining axis
    assert len(R.indices) == 2
    assert 'c' in R.itags
    
    # Verify reconstruction
    reconstructed = contract(U, R)
    assert set(reconstructed.itags) == {'a', 'b', 'c'}


def test_decomp_multi_axis_lv():
    """Test decomp with multiple axes in LV mode."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx4 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2, idx3, idx4], seed=2, itags=['a', 'b', 'c', 'd'])
    
    # Decompose on 3 axes
    L, V = decomp(T, axes=['a', 'b', 'c'], mode='LV')
    
    # L should have 3 original axes plus bond
    assert len(L.indices) == 4
    assert set(['a', 'b', 'c']).issubset(set(L.itags))
    
    # V should have bond and remaining axis
    assert len(V.indices) == 2
    assert 'd' in V.itags


def test_decomp_multi_axis_by_positions():
    """Test decomp with multiple axes specified by integer positions."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx, idx.flip(), idx, idx.flip()], seed=3, itags=['a', 'b', 'c', 'd'])
    
    # Decompose using positions
    U, S, Vh = decomp(T, axes=[0, 2], mode='SVD')
    
    # U should have positions 0, 2 plus bond
    assert len(U.indices) == 3
    assert 'a' in U.itags and 'c' in U.itags
    
    # Vh should have bond and positions 1, 3
    assert len(Vh.indices) == 3
    assert 'b' in Vh.itags and 'd' in Vh.itags


def test_decomp_multi_axis_reconstruction():
    """Test that multi-axis decomposition can reconstruct the original tensor."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 1), Sector(2, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 1), Sector(0, 1), Sector(1, 1)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 1)))
    
    T = Tensor.random([idx1, idx2, idx3], seed=123, itags=['a', 'b', 'c'])
    
    # Decompose
    U, R = decomp(T, axes=['a', 'b'], mode='UR')
    
    # Reconstruct
    reconstructed = contract(U, R)
    
    # Should have same indices
    assert len(reconstructed.indices) == 3
    assert set(reconstructed.itags) == {'a', 'b', 'c'}
    
    # Permute to match original order
    tag_to_pos_orig = {tag: i for i, tag in enumerate(T.itags)}
    tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
    perm = [tag_to_pos_recon[tag] for tag in T.itags]
    reconstructed.permute(perm)
    
    # Data should match (up to numerical precision)
    for key in T.data.keys():
        if key in reconstructed.data:
            np.testing.assert_allclose(T.data[key], reconstructed.data[key], rtol=1e-10, atol=1e-12)


def test_decomp_multi_axis_too_few_raises():
    """Test that decomp raises error when sequence has fewer than 2 axes."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    T = Tensor.random([idx, idx.flip()], seed=1, itags=['a', 'b'])
    
    with pytest.raises(ValueError, match="at least 2 axes"):
        decomp(T, axes=['a'], mode='SVD')


def test_decomp_single_axis_unchanged():
    """Test that single-axis behavior is unchanged."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(-1, 2)))
    
    T = Tensor.random([idx1, idx2], seed=50, itags=['a', 'b'])
    
    # Single axis by tag
    U1, S1, Vh1 = decomp(T, axes='a', mode='SVD')
    assert len(U1.indices) == 2
    assert 'a' in U1.itags
    
    # Single axis by position
    U2, S2, Vh2 = decomp(T, axes=0, mode='SVD')
    assert len(U2.indices) == 2
    assert 'a' in U2.itags


def test_decomp_multi_axis_with_flow():
    """Test multi-axis decomp with different flow parameters."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx, idx.flip(), idx, idx.flip()], seed=4, itags=['a', 'b', 'c', 'd'])
    
    # Test different flows
    for flow in ["><", ">>", "<<"]:
        U, S, Vh = decomp(T, axes=[0, 1], mode='SVD', flow=flow)
        assert len(U.indices) == 3
        assert len(S.indices) == 2
        assert len(Vh.indices) == 3


def test_decomp_multi_axis_with_itag():
    """Test multi-axis decomp with custom bond tags."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx, idx.flip(), idx], seed=5, itags=['a', 'b', 'c'])
    
    # Custom bond tag
    U, S, Vh = decomp(T, axes=[0, 1], mode='SVD', itag='custom_bond')
    
    assert 'custom_bond' in U.itags
    assert 'custom_bond' in S.itags
    assert 'custom_bond' in Vh.itags


def test_decomp_multi_axis_with_truncation():
    """Test multi-axis decomp with truncation."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    
    T = Tensor.random([idx, idx.flip(), idx], seed=6, itags=['a', 'b', 'c'])
    
    # Decompose with truncation
    U, S, Vh = decomp(T, axes=[0, 1], mode='SVD', trunc={"nkeep": 2})
    
    # Should still have correct structure
    assert len(U.indices) == 3
    assert 'a' in U.itags and 'b' in U.itags


def test_decomp_multi_axis_duplicate_itags():
    """Test multi-axis decomposition with duplicate itags using integer positions."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, (Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, (Sector(0, 2),))
    idx3 = Index(Direction.OUT, group, (Sector(0, 3),))
    
    # Create tensor with duplicate itags
    T = Tensor.random([idx1, idx2, idx3], itags=['x', 'x', 'y'], seed=100)
    
    # Decompose using integer positions (avoids itag ambiguity)
    # This tests that explicit axes work even with duplicate itags
    U, S, Vh = decomp(T, axes=[0, 1], mode='SVD')
    
    # Check structure
    assert len(U.indices) == 3  # 2 merged axes + bond
    assert len(S.indices) == 2  # bond indices
    assert len(Vh.indices) == 2  # 1 remaining axis + bond
    
    # Check that U has the two original axes (with their duplicate itags)
    assert U.itags[0] == 'x'  # First merged axis
    assert U.itags[1] == 'x'  # Second merged axis
    assert U.itags[2] == '_bond_L'  # Bond index
    
    # Check that Vh has the remaining axis
    assert 'y' in Vh.itags
    
    # Verify reconstruction
    S_Vh = contract(S, Vh, axes=(1, 0))
    reconstructed = contract(U, S_Vh, axes=(2, 0))
    
    # Check that all blocks match
    for key in T.data:
        np.testing.assert_allclose(reconstructed.data[key], T.data[key], atol=1e-10)


def test_decomp_multi_axis_preserves_index_order():
    """Test that multi-axis decomposition preserves the original index ordering."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, (Sector(0, 2),))
    idx2 = Index(Direction.IN, group, (Sector(0, 2),))
    idx3 = Index(Direction.OUT, group, (Sector(0, 3),))
    idx4 = Index(Direction.IN, group, (Sector(0, 2),))
    
    # Create a 4-index tensor
    T = Tensor.random([idx1, idx2, idx3, idx4], itags=['a', 'b', 'c', 'd'], seed=200)
    
    # Test 1: Merge axes [0, 2] (non-adjacent)
    U1, S1, Vh1 = decomp(T, axes=[0, 2], mode='SVD')
    
    # U should have indices in order: original axis 0, original axis 2, bond
    assert U1.itags[0] == 'a'  # Original axis 0
    assert U1.itags[1] == 'c'  # Original axis 2
    assert U1.itags[2] == '_bond_L'  # Bond
    
    # Vh should have remaining indices in order: original axes 1, 3, bond
    assert Vh1.itags[0] == '_bond_R'  # Bond
    assert Vh1.itags[1] == 'b'  # Original axis 1
    assert Vh1.itags[2] == 'd'  # Original axis 3
    
    # Test 2: Merge axes [1, 3] (non-adjacent)
    U2, R2 = decomp(T, axes=[1, 3], mode='UR')
    
    # U should preserve order of merged axes
    assert U2.itags[0] == 'b'  # Original axis 1
    assert U2.itags[1] == 'd'  # Original axis 3
    assert U2.itags[2] == '_bond_L'  # Bond
    
    # R should have remaining axes in order
    # UR mode uses '_bond_L' for both U and R
    assert R2.itags[0] == '_bond_L'  # Bond
    assert R2.itags[1] == 'a'  # Original axis 0
    assert R2.itags[2] == 'c'  # Original axis 2
    
    # Test 3: Merge three axes [0, 1, 3]
    L3, V3 = decomp(T, axes=[0, 1, 3], mode='LV')
    
    # L should preserve order of merged axes: 0, 1, 3
    assert L3.itags[0] == 'a'  # Original axis 0
    assert L3.itags[1] == 'b'  # Original axis 1
    assert L3.itags[2] == 'd'  # Original axis 3
    assert L3.itags[3] == '_bond_R'  # Bond (LV mode uses _bond_R)
    
    # V should have remaining axis
    # LV mode uses '_bond_R' for both L and V
    assert V3.itags[0] == '_bond_R'  # Bond
    assert V3.itags[1] == 'c'  # Original axis 2
    
    # Verify reconstruction for Test 1
    S1_Vh1 = contract(S1, Vh1, axes=(1, 0))
    recon1 = contract(U1, S1_Vh1, axes=(2, 0))
    # recon1 has itags ('a', 'c', 'b', 'd'), need to permute to ('a', 'b', 'c', 'd')
    recon1.permute([0, 2, 1, 3])
    for key in T.data:
        np.testing.assert_allclose(recon1.data[key], T.data[key], atol=1e-10)


def test_decomp_truncation_combined_thresh_nkeep():
    """Test decomp with both thresh and nkeep truncation modes (SVD)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 3), Sector(0, 5), Sector(1, 4)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 4), Sector(0, 6), Sector(1, 5)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=777)
    
    # Apply both: first filter singular values >= 0.5, then keep top 5
    U, S, Vh = decomp(T, axes=0, mode='SVD', trunc={"thresh": 0.5, "nkeep": 5})
    
    # Extract singular values from diagonal S tensor
    all_sv = []
    for key, block in S.data.items():
        sv = np.diag(block)
        all_sv.extend(sv)
    
    # Should have at most 5 values (nkeep)
    assert len(all_sv) <= 5
    
    # All should be >= 0.5 (thresh)
    assert np.all(np.array(all_sv) >= 0.5)
    
    # Verify reconstruction still works
    S_Vh = contract(S, Vh, axes=(1, 0))
    reconstructed = contract(U, S_Vh, axes=(1, 0))
    
    # Check dimensions match
    assert len(reconstructed.indices) == len(T.indices)


def test_decomp_truncation_combined_ur_mode():
    """Test decomp UR mode with both thresh and nkeep truncation."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 4), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(1, 4)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=888)
    
    # Apply both truncations in UR mode
    U, R = decomp(T, axes=0, mode='UR', trunc={"thresh": 0.3, "nkeep": 6})
    
    # Check that bond dimension is at most 6
    bond_dim = sum(sector.dim for sector in U.indices[1].sectors)
    assert bond_dim <= 6
    
    # Verify reconstruction works
    reconstructed = contract(U, R, axes=(1, 0))
    assert len(reconstructed.indices) == len(T.indices)
