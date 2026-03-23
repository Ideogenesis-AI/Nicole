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


"""Tests for tensor factorization operations: SVD, QR, and EIG."""

import math
import torch
import pytest

from nicole import Direction, Index, Sector, Tensor, U1Group, SU2Group
from nicole import contract, decomp, diag
from nicole.decomp import svd, qr, eig
from ..utils import assert_charge_neutral, populate_random_weights
from ..utils import assert_blocks_equal, assert_physical_tensors_equal


# =============================================================================
#  Singular Value Decomposition Tests
# =============================================================================

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
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)
    
    # Check norms are preserved
    assert math.isclose(reconstructed.norm(), original_norm)


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
    reconstructed.permute([1, 0, 2], in_place=True)
    
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / T.norm()
    assert rel_error < 1e-12
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


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
            reconstructed.permute([1, 0, 2], in_place=True)
        elif axis == 2:
            # Current order (c, a, b) -> (a, b, c)
            reconstructed.permute([1, 2, 0], in_place=True)
        
        # Verify reconstruction
        diff_norm = (T - reconstructed).norm()
        rel_error = diff_norm / T.norm()
        assert rel_error < 1e-12, f"Reconstruction failed for axis {axis}"
        assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


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
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


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
        assert torch.all(s_array >= 0).item(), "Singular values should be non-negative"
        # Check sorted in descending order
        sorted_s = torch.sort(s_array)[0]
        reversed_s = torch.flip(sorted_s, dims=[0])
        assert torch.allclose(s_array, reversed_s), \
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
        assert torch.allclose(block, torch.diag(torch.diag(block)))


# U and Vd isometry tests

def test_svd_u1_u_blocks_are_isometric():
    """Each U block satisfies U^† @ U = I (column-orthonormal)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 3), Sector(0, 4), Sector(1, 3)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(-1, 2), Sector(0, 5), Sector(1, 4)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 3), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=50)

    U, _S, _Vh = svd(T, axis=0)

    for key, block in U.data.items():
        should_be_I = block.T.conj() @ block
        rank = block.shape[1]
        assert torch.allclose(should_be_I, torch.eye(rank, dtype=block.dtype), atol=1e-12), (
            f"U block {key}: U^† @ U should be identity, max deviation="
            f"{(should_be_I - torch.eye(rank, dtype=block.dtype)).abs().max().item():.2e}"
        )


def test_svd_u1_vh_blocks_are_isometric():
    """Vh blocks, concatenated per left charge, satisfy Vh @ Vh^† = I (row-orthonormal)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 3), Sector(0, 4), Sector(1, 3)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(-1, 2), Sector(0, 5), Sector(1, 4)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 3), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=50)

    _U, _S, Vh = svd(T, axis=0)

    # Group Vh blocks by q_left (position 0 of key), then concatenate horizontally
    vh_by_q_left: dict = {}
    for key, block in Vh.data.items():
        q_left = key[0]
        row = block.reshape(block.shape[0], -1)  # (bond_dim, prod_right)
        vh_by_q_left.setdefault(q_left, []).append(row)

    for q_left, rows in vh_by_q_left.items():
        Vh_full = torch.cat(rows, dim=1)  # (bond_dim, total_right_dim)
        should_be_I = Vh_full @ Vh_full.T.conj()
        rank = Vh_full.shape[0]
        assert torch.allclose(should_be_I, torch.eye(rank, dtype=Vh_full.dtype), atol=1e-12), (
            f"Vh for q_left={q_left}: Vh @ Vh^† should be identity, max deviation="
            f"{(should_be_I - torch.eye(rank, dtype=Vh_full.dtype)).abs().max().item():.2e}"
        )


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
    
    T = Tensor.random([idx1, idx2], dtype=torch.complex128, itags=["a", "b"], seed=5)
    
    # Use UR mode for efficient reconstruction
    U, R = decomp(T, axes=0, mode="UR")
    
    # U and R should be complex
    assert U.dtype.is_complex
    assert R.dtype.is_complex
    
    # Reconstruct using explicit pairs
    reconstructed = contract(U, R, axes=(1, 0))
    
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / T.norm()
    assert rel_error < 1e-12
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


# Truncation tests

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
        assert isinstance(s_array, torch.Tensor)
        assert s_array.ndim == 1
        assert len(s_array) > 0


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
        assert torch.all(s_array >= threshold).item(), f"Block {key} has singular values < {threshold}"
    
    # Verify truncation happened (should have fewer than 10 singular values)
    total_kept = sum(len(s_array) for s_array in S_blocks.values())
    assert total_kept < 10, "Expected truncation to reduce number of singular values"


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
        assert torch.all(s_array >= 0.5).item()
    
    # Verify we got exactly 3 (both constraints satisfied)
    assert total_sv == 3


# High-order tensor tests

def test_svd_high_order_different_axis_sizes():
    """Test SVD on high-order tensor with varying axis dimensions."""
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

    # Verify singular values are sorted descending
    for key, s_array in S_blocks.items():
        assert torch.all(s_array[:-1] >= s_array[1:]).item(), "Singular values should be sorted descending"


def test_svd_high_order_bond_structure():
    """Test bond index structure from SVD of a high-order tensor."""
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


def test_svd_high_order_thresh_truncation():
    """Test threshold truncation on high-order tensor SVD."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(-1, 3), Sector(0, 4), Sector(1, 3))),
        Index(Direction.IN, group, sectors=(Sector(-2, 2), Sector(-1, 2), Sector(0, 4), Sector(1, 2))),
        Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 4), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 4), Sector(1, 2)))
    ]

    T = Tensor.random(indices, itags=["a", "b", "c", "d"], seed=1500)

    threshold = 1.0
    U, S_blocks, Vh = svd(T, axis=0, trunc={"thresh": threshold})

    # All kept singular values should be >= threshold
    for key, s_array in S_blocks.items():
        assert torch.all(s_array >= threshold).item()

    # Left index of U should be unchanged
    expected_left_dim = sum(s.dim for s in indices[0].sectors)
    assert U.indices[0].dim == expected_left_dim


# SU(2) SVD tests

def _make_su2_3rd_order(seed: int = 1):
    """3rd-order SU(2) tensor with (OUT, IN, OUT) index structure."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1), Sector(2, 2)))
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=seed)
    populate_random_weights(T, seed=seed + 1000)
    return T


def _make_su2_2nd_order(seed: int = 42):
    """2nd-order SU(2) tensor (OUT x IN)."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    return Tensor.random([idx1, idx2], itags=["a", "b"], seed=seed)


def _make_su2_4th_order(seed: int = 1):
    """4th-order SU(2) tensor with (OUT, IN, OUT, IN) index structure."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 1)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1), Sector(2, 2)))
    idx4 = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 1)))
    T = Tensor.random([idx1, idx2, idx3, idx4], itags=["a", "b", "c", "d"], seed=seed)
    populate_random_weights(T, seed=seed + 1000)
    return T


def _make_su2_5th_order(seed: int = 1):
    """5th-order SU(2) tensor with (OUT, IN, OUT, IN, OUT) index structure."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 1), Sector(2, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 1)))
    idx4 = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 1), Sector(2, 1)))
    idx5 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    T = Tensor.random([idx1, idx2, idx3, idx4, idx5], itags=["a", "b", "c", "d", "e"], seed=seed)
    populate_random_weights(T, seed=seed + 1000)
    return T


def _make_su2_6th_order(seed: int = 1):
    """6th-order SU(2) tensor with (OUT, IN, OUT, IN, OUT, IN) index structure."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx2 = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 1), Sector(2, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 1)))
    idx4 = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 1), Sector(2, 1)))
    idx5 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 2)))
    idx6 = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 1)))
    T = Tensor.random([idx1, idx2, idx3, idx4, idx5, idx6], itags=["a", "b", "c", "d", "e", "f"], seed=seed)
    populate_random_weights(T, seed=seed + 1000)
    return T


def test_svd_su2_reconstruction_2nd_order():
    """SVD of a 2nd-order SU(2) tensor reconstructs the original for all decomposed axes."""
    T = _make_su2_2nd_order(seed=11)

    for axis in range(len(T.indices)):
        U, S_tensor, Vh = decomp(T, axes=axis, mode="SVD")
        reconstructed = contract(U, contract(S_tensor, Vh))

        tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
        perm = [tag_to_pos_recon[tag] for tag in T.itags]
        reconstructed.permute(perm, in_place=True)

        assert_physical_tensors_equal(T, reconstructed, atol=1e-10,
                                      msg=f"SU(2) 2nd-order SVD reconstruction axis={axis}")


def test_svd_su2_reconstruction_3rd_order():
    """SVD of a 3rd-order SU(2) tensor reconstructs the original for all decomposed axes."""
    T = _make_su2_3rd_order(seed=10)

    for axis in range(len(T.indices)):
        U, S_tensor, Vh = decomp(T, axes=axis, mode="SVD")
        reconstructed = contract(U, contract(S_tensor, Vh))

        tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
        perm = [tag_to_pos_recon[tag] for tag in T.itags]
        reconstructed.permute(perm, in_place=True)

        assert_physical_tensors_equal(T, reconstructed, atol=1e-10,
                                      msg=f"SU(2) 3rd-order SVD reconstruction axis={axis}")


def test_svd_su2_reconstruction_4th_order():
    """SVD of a 4th-order SU(2) tensor reconstructs the original for all decomposed axes."""
    T = _make_su2_4th_order(seed=12)

    for axis in range(len(T.indices)):
        U, S_tensor, Vh = decomp(T, axes=axis, mode="SVD")
        reconstructed = contract(U, contract(S_tensor, Vh))

        tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
        perm = [tag_to_pos_recon[tag] for tag in T.itags]
        reconstructed.permute(perm, in_place=True)

        assert_physical_tensors_equal(T, reconstructed, atol=1e-10,
                                      msg=f"SU(2) 4th-order SVD reconstruction axis={axis}")


def test_svd_su2_reconstruction_5th_order():
    """SVD of a 5th-order SU(2) tensor reconstructs the original for all decomposed axes."""
    T = _make_su2_5th_order(seed=13)

    for axis in range(len(T.indices)):
        U, S_tensor, Vh = decomp(T, axes=axis, mode="SVD")
        reconstructed = contract(U, contract(S_tensor, Vh))

        tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
        perm = [tag_to_pos_recon[tag] for tag in T.itags]
        reconstructed.permute(perm, in_place=True)

        assert_physical_tensors_equal(T, reconstructed, atol=1e-10,
                                      msg=f"SU(2) 5th-order SVD reconstruction axis={axis}")


def test_svd_su2_reconstruction_6th_order():
    """SVD of a 6th-order SU(2) tensor reconstructs the original for all decomposed axes."""
    T = _make_su2_6th_order(seed=14)

    for axis in range(len(T.indices)):
        U, S_tensor, Vh = decomp(T, axes=axis, mode="SVD")
        reconstructed = contract(U, contract(S_tensor, Vh))

        tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
        perm = [tag_to_pos_recon[tag] for tag in T.itags]
        reconstructed.permute(perm, in_place=True)

        assert_physical_tensors_equal(T, reconstructed, atol=1e-10,
                                      msg=f"SU(2) 6th-order SVD reconstruction axis={axis}")


def test_svd_su2_u_intertwiner_identity_like():
    """U from SU(2) SVD has identity-like intertwiner: one component, weights[0,0]=sqrt(irrep_dim)."""
    group = SU2Group()
    T = _make_su2_3rd_order(seed=20)

    U, _S, _Vh = svd(T, axis=0)

    assert U.intw is not None, "U must have intw for SU(2)"
    for key, bridge in U.intw.items():
        q = key[0]
        expected_weight = math.sqrt(group.irrep_dim(q))
        assert bridge.num_components == 1, f"U intertwiner at {key} must have 1 component"
        assert bridge.om_dimension == 1, f"U intertwiner OM dim must be 1 (2nd-order identity-like)"
        assert math.isclose(bridge.weights[0, 0].item(), expected_weight, rel_tol=1e-10), (
            f"U intertwiner weight at {key} should be sqrt(irrep_dim)={expected_weight:.6f}, "
            f"got {bridge.weights[0, 0].item():.6f}"
        )


def test_svd_su2_s_intertwiner_identity_like():
    """S from SU(2) SVD has identity-like intertwiner."""
    group = SU2Group()
    T = _make_su2_3rd_order(seed=21)

    _U, S_tensor, _Vh = decomp(T, axes=0, mode="SVD")

    assert S_tensor.intw is not None, "S must have intw for SU(2)"
    for key, bridge in S_tensor.intw.items():
        q = key[0]
        expected_weight = math.sqrt(group.irrep_dim(q))
        assert bridge.num_components == 1, f"S intertwiner at {key} must have 1 component"
        assert bridge.om_dimension == 1, f"S intertwiner OM dim must be 1 (2nd-order identity-like)"
        assert math.isclose(bridge.weights[0, 0].item(), expected_weight, rel_tol=1e-10), (
            f"S intertwiner weight at {key} should be sqrt(irrep_dim)={expected_weight:.6f}, "
            f"got {bridge.weights[0, 0].item():.6f}"
        )


def test_svd_su2_u_data_blocks_have_trailing_component_dim():
    """U data blocks from SU(2) SVD have shape (d_left, rank, 1)."""
    T = _make_su2_3rd_order(seed=22)

    U, _S, _Vh = svd(T, axis=0)

    for key, block in U.data.items():
        assert block.ndim == 3, f"U block {key} must be 3D for SU(2), got ndim={block.ndim}"
        assert block.shape[-1] == 1, f"U block {key} trailing dim must be 1, got {block.shape[-1]}"


def test_svd_su2_u_reduced_blocks_are_isometric():
    """The reduced matrix U_mat = U.data[key][:, :, 0] satisfies U_mat^T @ U_mat = I."""
    T = _make_su2_3rd_order(seed=23)

    U, _S, _Vh = svd(T, axis=0)

    for key, block in U.data.items():
        U_mat = block[:, :, 0]  # (d_left, rank)
        should_be_I = U_mat.T @ U_mat
        rank = U_mat.shape[1]
        assert torch.allclose(should_be_I, torch.eye(rank, dtype=U_mat.dtype), atol=1e-10), (
            f"U reduced block {key}: U^T @ U should be identity, max deviation="
            f"{(should_be_I - torch.eye(rank, dtype=U_mat.dtype)).abs().max().item():.2e}"
        )


def test_svd_su2_vh_blocks_are_isometric():
    """Vh reduced blocks, concatenated per left charge, satisfy Vh @ Vh^† = I (row-orthonormal)."""
    T = _make_su2_3rd_order(seed=24)

    _U, _S, Vh = svd(T, axis=0)

    # Group Vh blocks by q_left (position 0 of key), then concatenate horizontally
    vh_by_q_left: dict = {}
    for key, block in Vh.data.items():
        q_left = key[0]
        row = block.reshape(block.shape[0], -1)  # (bond_dim, prod_right_with_component)
        vh_by_q_left.setdefault(q_left, []).append(row)

    for q_left, rows in vh_by_q_left.items():
        Vh_full = torch.cat(rows, dim=1)  # (bond_dim, total_right_dim)
        should_be_I = Vh_full @ Vh_full.T.conj()
        rank = Vh_full.shape[0]
        assert torch.allclose(should_be_I, torch.eye(rank, dtype=Vh_full.dtype), atol=1e-10), (
            f"Vh for q_left={q_left}: Vh @ Vh^† should be identity, max deviation="
            f"{(should_be_I - torch.eye(rank, dtype=Vh_full.dtype)).abs().max().item():.2e}"
        )


def test_svd_su2_vd_intertwiner_matches_t_axis0():
    """For left_axis=0 the R-symbol is identity: Vd.intw weights equal T.intw weights."""
    T = _make_su2_3rd_order(seed=30)

    _U, _S, Vh = svd(T, axis=0)

    assert Vh.intw is not None, "Vd must have intw for SU(2)"
    assert T.intw is not None

    for t_key, t_bridge in T.intw.items():
        vh_key = t_key  # keys are identical when left_axis=0
        assert vh_key in Vh.intw, f"Vd.intw missing key {vh_key}"
        assert torch.allclose(Vh.intw[vh_key].weights, t_bridge.weights, atol=1e-14), (
            f"Vd.intw[{vh_key}].weights differ from T.intw[{t_key}].weights"
        )


def test_svd_su2_vd_has_correct_intw_keys():
    """Vd has exactly the same intertwiner key set as T when left_axis=0."""
    T = _make_su2_3rd_order(seed=31)

    _U, _S, Vh = svd(T, axis=0)

    assert set(Vh.intw.keys()) == set(T.intw.keys()), (
        f"Vd.intw keys {set(Vh.intw.keys())} differ from T.intw keys {set(T.intw.keys())}"
    )


def test_svd_su2_non_zero_axis_reconstruction():
    """SVD with left_axis=1 (R-symbol path) reconstructs the SU(2) tensor."""
    T = _make_su2_3rd_order(seed=40)

    U, S_tensor, Vh = decomp(T, axes=1, mode="SVD")
    reconstructed = contract(U, contract(S_tensor, Vh))

    tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
    perm = [tag_to_pos_recon[tag] for tag in T.itags]
    reconstructed.permute(perm, in_place=True)

    assert_physical_tensors_equal(T, reconstructed, atol=1e-10,
                                  msg="SU(2) left_axis=1 SVD reconstruction")


def test_svd_su2_non_zero_axis_vh_intw_not_none():
    """Vd has a populated intw even when left_axis != 0."""
    T = _make_su2_3rd_order(seed=41)

    _U, _S, Vh = svd(T, axis=1)

    assert Vh.intw is not None, "Vd must have intw for SU(2) regardless of left_axis"
    assert len(Vh.intw) > 0, "Vd.intw must be non-empty"


def test_svd_su2_truncation_nkeep():
    """nkeep truncation on SU(2) tensor produces tensors with valid intw."""
    T = _make_su2_3rd_order(seed=50)

    U, S_tensor, Vh = decomp(T, axes=0, mode="SVD", trunc={"nkeep": 2})

    assert U.intw is not None, "U must have intw after nkeep truncation"
    assert Vh.intw is not None, "Vh must have intw after nkeep truncation"
    assert S_tensor.intw is not None, "S must have intw after nkeep truncation"
    assert len(U.data) > 0, "U must have non-empty blocks after truncation"


def test_svd_su2_truncation_thresh():
    """thresh truncation (small thresh) on SU(2) tensor still reconstructs accurately."""
    T = _make_su2_3rd_order(seed=51)

    U, S_tensor, Vh = decomp(T, axes=0, mode="SVD", trunc={"thresh": 1e-10})
    reconstructed = contract(U, contract(S_tensor, Vh))

    tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
    perm = [tag_to_pos_recon[tag] for tag in T.itags]
    reconstructed.permute(perm, in_place=True)

    assert_physical_tensors_equal(T, reconstructed, atol=1e-8,
                                  msg="SU(2) thresh=1e-10 reconstruction")


def test_svd_su2_truncation_nkeep_s_blocks_3d():
    """S blocks are 3D with trailing dim 1 after truncation."""
    T = _make_su2_3rd_order(seed=52)

    _U, S_tensor, _Vh = decomp(T, axes=0, mode="SVD", trunc={"nkeep": 3})

    for key, block in S_tensor.data.items():
        assert block.ndim == 3, f"S block {key} must be 3D for SU(2)"
        assert block.shape[-1] == 1, f"S block {key} trailing dim must be 1"


def test_svd_su2_charge_neutral():
    """U, S, Vh from SU(2) SVD are all charge neutral."""
    T = _make_su2_3rd_order(seed=60)

    U, S_tensor, Vh = decomp(T, axes=0, mode="SVD")

    assert_charge_neutral(U)
    assert_charge_neutral(S_tensor)
    assert_charge_neutral(Vh)


# =============================================================================
#  QR Decomposition Tests
# =============================================================================

# Basic QR tests

def test_qr_basic_reconstruction():
    """Test basic QR decomposition and reconstruction on a simple tensor."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 3), Sector(2, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 3)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 1), Sector(1, 2), Sector(2, 1)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=42)
    original_norm = T.norm()

    # Perform QR separating axis 0 from the rest
    Q, R = qr(T, axis=0)

    # Check that all tensors are charge neutral
    assert_charge_neutral(Q)
    assert_charge_neutral(R)

    # Reconstruct tensor (automatic detection based on matching itags)
    reconstructed = contract(Q, R)

    # Check reconstruction accuracy
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / original_norm
    assert rel_error < 1e-12, f"Reconstruction error {rel_error} too large"
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)
    
    # Check norms are preserved
    assert abs(reconstructed.norm() - original_norm) < 1e-12


def test_qr_integer_axis():
    """Test QR with integer axis specification."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=1)
    
    Q, R = qr(T, axis=0)
    
    assert len(Q.indices) == 2
    assert len(R.indices) == 2


def test_qr_string_axis():
    """Test QR with axis specified by string tag."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=123)

    # Perform QR by axis name
    Q, R = qr(T, axis="b")

    # Check structure
    assert len(Q.indices) == 2
    assert len(R.indices) == 3
    assert Q.itags[0] == "b"
    assert R.itags[1] == "a"
    assert R.itags[2] == "c"

    # Reconstruct and verify
    reconstructed = contract(Q, R)
    
    # Permute reconstructed to match original order (b, a, c) -> (a, b, c)
    reconstructed.permute([1, 0, 2], in_place=True)
    
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / T.norm()
    assert rel_error < 1e-12
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


def test_qr_different_axis_positions():
    """Test QR on different axis positions."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 4),))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=111)

    # Test QR on each axis
    for axis in [0, 1, 2]:
        Q, R = qr(T, axis=axis)
        
        # Reconstruct using explicit pairs
        reconstructed = contract(Q, R, axes=(1, 0))
        
        # Permute back to original order
        if axis == 0:
            # Already in correct order (a, b, c)
            pass
        elif axis == 1:
            # Current order (b, a, c) -> (a, b, c)
            reconstructed.permute([1, 0, 2], in_place=True)
        elif axis == 2:
            # Current order (c, a, b) -> (a, b, c)
            reconstructed.permute([1, 2, 0], in_place=True)
        
        # Verify reconstruction
        diff_norm = (T - reconstructed).norm()
        rel_error = diff_norm / T.norm()
        assert rel_error < 1e-12, f"Reconstruction failed for axis {axis}"
        assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


# Block handling tests

def test_qr_multiple_blocks_same_charge():
    """Test QR with multiple blocks and diverse charge sectors."""
    group = U1Group()
    
    # Create indices with diverse charge sectors
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 2), Sector(2, 1)))

    T = Tensor.random(indices=(idx1, idx2, idx3), itags=["a", "b", "c"], seed=42)

    # Perform QR
    Q, R = qr(T, axis=0)

    # Verify that we have blocks for multiple charges
    assert len(Q.data) >= 2, "Should have blocks for multiple q_left charges"
    assert len(R.data) >= 2, "Should have multiple R blocks"

    # Reconstruct and verify
    reconstructed = contract(Q, R)
    
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / T.norm()
    assert rel_error < 1e-12
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


def test_qr_single_block():
    """Test QR with single block tensor."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 4),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=999)
    
    # Test with raw qr()
    Q, R = qr(T, axis=0)
    
    assert len(Q.data) == 1
    assert len(R.data) == 1


# Index and direction tests

def test_qr_index_directions():
    """Test that QR produces correct index directions for contraction."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=456)

    Q, R = qr(T, axis=0)

    # Q's bond index should have opposite direction of left index
    assert Q.indices[1].direction == idx1.direction.reverse()
    
    # R's bond index should be flipped from Q's bond index
    assert R.indices[0].direction == Q.indices[1].direction.reverse()
    
    # Verify contraction works
    reconstructed = contract(Q, R)
    
    # Check that reconstructed has same index directions as original
    for i, idx in enumerate(reconstructed.indices):
        assert idx.direction == T.indices[i].direction


def test_qr_bond_index_structure():
    """Test that bond index has correct structure."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 3), Sector(2, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 2), Sector(2, 1)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=123)
    
    Q, R = qr(T, axis=0)
    
    bond_index = Q.indices[1]
    
    # Bond should have same group as left index
    assert bond_index.group == idx1.group
    
    # Bond charges should be a subset of left charges
    bond_charges = bond_index.charges()
    left_charges = idx1.charges()
    assert set(bond_charges).issubset(set(left_charges))
    
    # Bond should have at least one charge (non-trivial)
    assert len(bond_charges) > 0


def test_qr_bond_dimensions():
    """Test that bond dimensions are computed correctly."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5),))

    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=444)

    Q, R = qr(T, axis=0)

    # Bond dimension should be min(3, 5) = 3 (reduced QR)
    bond_index = Q.indices[1]
    assert bond_index.dim == 3, "Bond dimension should be min of left and right dimensions"


# Q orthogonality tests

def test_qr_q_orthogonality():
    """Test that Q matrix is orthogonal (Q^H @ Q = I)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 4),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 6),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=789)
    
    Q, R = qr(T, axis=0)
    
    # Check Q^H @ Q = I for each block
    for key, q_block in Q.data.items():
        # Q^H @ Q should be identity
        identity = q_block.T.conj() @ q_block
        eye = torch.eye(identity.shape[0], dtype=identity.dtype)
        assert torch.allclose(identity, eye, atol=1e-12), "Q is not orthogonal"


def test_qr_q_orthonormal_columns():
    """Test that Q has orthonormal columns."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 3), Sector(0, 4), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 5), Sector(1, 4)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=555)
    
    Q, R = qr(T, axis=0)
    
    # For each block, check that columns are orthonormal
    for key, q_block in Q.data.items():
        # Compute Gram matrix (Q^H @ Q)
        gram = q_block.T.conj() @ q_block
        # Should be identity
        eye = torch.eye(gram.shape[0], dtype=gram.dtype)
        assert torch.allclose(gram, eye, atol=1e-12), "Q columns are not orthonormal"


# R upper triangular tests

def test_qr_r_upper_triangular():
    """Test that R matrix is upper triangular (when viewed as matrix)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 5),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=333)
    
    Q, R = qr(T, axis=0)
    
    # For single block, check that R is upper triangular
    for key, r_block in R.data.items():
        # r_block should be a matrix (rank, dim)
        # For square or reduced case, first dimension is the bond
        if r_block.ndim == 2:
            # Extract the square part if possible
            rank = r_block.shape[0]
            # Check lower triangle is zero
            lower_tri = torch.tril(r_block[:rank, :rank], diagonal=-1)
            assert torch.allclose(lower_tri, torch.zeros_like(lower_tri), atol=1e-12), \
                "R is not upper triangular"


# Charge conservation tests

def test_qr_preserves_charge_conservation():
    """Test that QR preserves charge conservation in all output tensors."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))

    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=222)
    
    # Original tensor should be charge neutral
    assert_charge_neutral(T)

    # Perform QR
    Q, R = qr(T, axis=1)

    # All output tensors should be charge neutral
    assert_charge_neutral(Q)
    assert_charge_neutral(R)


def test_qr_charge_conservation_all_axes():
    """Test charge conservation for QR on all axes."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 1), Sector(0, 2), Sector(1, 2), Sector(2, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(0, 2), Sector(2, 1)))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=4)
    
    for axis in [0, 1, 2]:
        Q, R = qr(T, axis=axis)
        
        assert_charge_neutral(Q)
        assert_charge_neutral(R)


# Error handling tests

def test_qr_invalid_integer_axis():
    """Test that QR raises error for invalid integer axis."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))

    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=333)

    # Test invalid integer axis
    with pytest.raises(ValueError, match="out of range"):
        qr(T, axis=5)
    
    with pytest.raises(ValueError, match="out of range"):
        qr(T, axis=-1)


def test_qr_invalid_string_axis():
    """Test that QR raises error for invalid string axis."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))

    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=333)

    # Test invalid string axis
    with pytest.raises(ValueError, match="not found"):
        qr(T, axis="nonexistent")


def test_qr_ambiguous_string_axis():
    """Test that QR raises error for ambiguous string axis (duplicate itags)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))

    # Create tensor with duplicate itags
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "a"], seed=444)

    # Test ambiguous string axis
    with pytest.raises(ValueError, match="Ambiguous axis specification"):
        qr(T, axis="a")


# dtype tests

def test_qr_complex_dtype():
    """Test QR with complex dtype."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 3), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-2, 1), Sector(-1, 2), Sector(0, 4), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2], dtype=torch.complex128, itags=["a", "b"], seed=5)
    
    Q, R = qr(T, axis=0)
    
    # Q and R should be complex
    assert Q.dtype.is_complex
    assert R.dtype.is_complex
    
    # Reconstruct using explicit pairs
    reconstructed = contract(Q, R, axes=(1, 0))
    
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / T.norm()
    assert rel_error < 1e-12
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


def test_qr_float32_dtype():
    """Test QR with float32 dtype."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 4),))
    
    T = Tensor.random([idx1, idx2], dtype=torch.float32, itags=["a", "b"], seed=666)
    
    Q, R = qr(T, axis=0)
    
    # Q and R should be float32
    assert Q.dtype == torch.float32
    assert R.dtype == torch.float32
    
    # Reconstruct and verify
    reconstructed = contract(Q, R, axes=(1, 0))
    
    diff_norm = (T - reconstructed).norm()
    rel_error = diff_norm / T.norm()
    assert rel_error < 1e-5  # Lower precision for float32
    assert_blocks_equal(T, reconstructed)  # Default tolerances for float32


# High-order tensor tests

def test_qr_4th_order_tensor():
    """Test QR on 4-index tensor with reconstruction."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx4 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2, idx3, idx4], itags=["a", "b", "c", "d"], seed=800)
    original_norm = T.norm()
    
    # Test decomposition on different axes
    for axis in [0, 1, 2, 3]:
        Q, R = qr(T, axis=axis)
        
        # Reconstruct
        reconstructed = contract(Q, R, axes=(1, 0))
        
        # Permute back to original order
        if axis == 0:
            pass  # (a, b, c, d)
        elif axis == 1:
            reconstructed.permute([1, 0, 2, 3], in_place=True)  # (b, a, c, d) -> (a, b, c, d)
        elif axis == 2:
            reconstructed.permute([1, 2, 0, 3], in_place=True)  # (c, a, b, d) -> (a, b, c, d)
        elif axis == 3:
            reconstructed.permute([1, 2, 3, 0], in_place=True)  # (d, a, b, c) -> (a, b, c, d)
        
        # Verify accuracy
        rel_error = (T - reconstructed).norm() / original_norm
        assert rel_error < 1e-12, f"Reconstruction failed for axis {axis}"
        assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


def test_qr_5th_order_tensor():
    """Test QR on 5-index tensor."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d", "e"], seed=900)
    
    # Test QR on middle axis
    Q, R = qr(T, axis=2)
    
    # Check structure
    assert len(Q.indices) == 2  # (c, bond)
    assert len(R.indices) == 5  # (bond.flip(), a, b, d, e)
    
    # Verify charge conservation
    assert_charge_neutral(Q)
    assert_charge_neutral(R)
    
    # Reconstruct
    reconstructed = contract(Q, R, axes=(1, 0))
    
    # Permute: (c, a, b, d, e) -> (a, b, c, d, e)
    reconstructed.permute([1, 2, 0, 3, 4], in_place=True)
    
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


def test_qr_6th_order_tensor():
    """Test QR on 6-index tensor."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),)),
        Index(Direction.OUT, group, sectors=(Sector(0, 2),)),
        Index(Direction.IN, group, sectors=(Sector(0, 2),))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d", "e", "f"], seed=1000)
    
    # Decompose (separate first index from rest)
    Q, R = qr(T, axis=0)
    
    # Check structure
    assert len(Q.indices) == 2  # (a, bond)
    assert len(R.indices) == 6  # (bond.flip(), b, c, d, e, f)
    
    # Reconstruct
    reconstructed = contract(Q, R, axes=(1, 0))
    
    # Verify dimensions
    assert len(reconstructed.indices) == 6
    for i, idx in enumerate(reconstructed.indices):
        assert idx.dim == T.indices[i].dim


def test_qr_high_order_multiple_charges():
    """Test high-order tensor with multiple charge blocks."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 2))),
        Index(Direction.IN, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 1))),
        Index(Direction.OUT, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(-2, 1), Sector(0, 2), Sector(2, 1)))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d"], seed=1100)
    
    # Test QR
    Q, R = qr(T, axis=0)
    
    # Check that we have multiple charge sectors
    bond_charges = set(Q.indices[1].charges())
    assert len(bond_charges) > 1, "Should have multiple charge sectors"
    
    # Reconstruct
    reconstructed = contract(Q, R, axes=(1, 0))
    
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


def test_qr_high_order_different_axis_sizes():
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
    Q, R = qr(T, axis=1)
    
    # Bond dimension should be min(dim_axis1, dim_others)
    # dim_axis1 = 5, dim_others = 2*3*4 = 24
    # So bond_dim = min(5, 24) = 5
    bond_dim = Q.indices[1].dim
    assert bond_dim == 5


def test_qr_high_order_bond_structure():
    """Test bond index structure in high-order tensor QR."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d"], seed=1400)
    
    Q, R = qr(T, axis=0)
    
    bond_index = Q.indices[1]
    
    # Bond should have charges that appear in the tensor's first index
    left_charges_in_data = set(key[0] for key in T.data.keys())
    bond_charges = set(bond_index.charges())
    
    assert bond_charges == left_charges_in_data
    
    # Bond should have correct group
    assert bond_index.group == indices[0].group
    
    # Bond direction should be opposite of left index
    assert bond_index.direction == indices[0].direction.reverse()


# Comparison with standard QR tests

def test_qr_matches_torch_qr():
    """Test that our QR matches torch.linalg.qr for single block."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 4),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 6),))
    
    # Create tensor with known data
    data = {(0, 0): torch.randn(4, 6, dtype=torch.float64)}
    T = Tensor(indices=(idx1, idx2), itags=("a", "b"), data=data, dtype=torch.float64)
    
    # Our QR
    Q, R = qr(T, axis=0)
    
    # PyTorch QR
    Q_torch, R_torch = torch.linalg.qr(data[(0, 0)], mode='reduced')
    
    # Compare (up to sign ambiguity in columns)
    Q_block = Q.data[(0, 0)]
    R_block = R.data[(0, 0)]
    
    # Check Q^H @ Q = I
    identity = Q_block.T @ Q_block
    assert torch.allclose(identity, torch.eye(identity.shape[0], dtype=identity.dtype), atol=1e-12)
    
    # Check Q @ R = original matrix
    reconstructed = Q_block @ R_block
    assert torch.allclose(reconstructed, data[(0, 0)], atol=1e-12)


def test_qr_thin_matrix():
    """Test QR on thin matrix (more rows than columns)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 10),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=111)
    
    Q, R = qr(T, axis=0)
    
    # Q should be 10x3, R should be 3x3
    Q_block = Q.data[(0, 0)]
    R_block = R.data[(0, 0)]
    
    assert Q_block.shape == (10, 3)
    assert R_block.shape == (3, 3)
    
    # Q should have orthonormal columns
    gram = Q_block.T @ Q_block
    assert torch.allclose(gram, torch.eye(3, dtype=gram.dtype), atol=1e-12)
    
    # Reconstruct
    reconstructed = contract(Q, R, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


def test_qr_wide_matrix():
    """Test QR on wide matrix (more columns than rows)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 10),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=222)
    
    Q, R = qr(T, axis=0)
    
    # Q should be 3x3, R should be 3x10
    Q_block = Q.data[(0, 0)]
    R_block = R.data[(0, 0)]
    
    assert Q_block.shape == (3, 3)
    assert R_block.shape == (3, 10)
    
    # Q should be orthogonal
    identity = Q_block.T @ Q_block
    assert torch.allclose(identity, torch.eye(3, dtype=identity.dtype), atol=1e-12)
    
    # Reconstruct
    reconstructed = contract(Q, R, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


def test_qr_square_matrix():
    """Test QR on square matrix."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 5),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=333)
    
    Q, R = qr(T, axis=0)
    
    # Both Q and R should be 5x5
    Q_block = Q.data[(0, 0)]
    R_block = R.data[(0, 0)]
    
    assert Q_block.shape == (5, 5)
    assert R_block.shape == (5, 5)
    
    # Q should be orthogonal (unitary)
    identity = Q_block.T @ Q_block
    assert torch.allclose(identity, torch.eye(5, dtype=identity.dtype), atol=1e-12)
    
    # Reconstruct
    reconstructed = contract(Q, R, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


# Edge cases

def test_qr_minimal_tensor():
    """Test QR on minimal 1x1 tensor."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 1),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=999)
    
    Q, R = qr(T, axis=0)
    
    # Should still work
    reconstructed = contract(Q, R, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


def test_qr_preserves_itags():
    """Test that QR preserves original itags in output tensors."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 4),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["physical", "virtual", "ancilla"], seed=555)
    
    Q, R = qr(T, axis="physical")
    
    # Q should have original itag for separated axis
    assert Q.itags[0] == "physical"
    assert Q.itags[1] == "_bond"
    
    # R should have remaining original itags
    assert "_bond" in R.itags
    assert "virtual" in R.itags
    assert "ancilla" in R.itags


def test_qr_multiblock_reconstruction_accuracy():
    """Test reconstruction accuracy with multiple charge blocks."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-3, 2), Sector(-1, 3), Sector(0, 4), Sector(1, 3), Sector(3, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-2, 2), Sector(-1, 3), Sector(0, 5), Sector(1, 3), Sector(2, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 3), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=777)
    
    Q, R = qr(T, axis=0)
    
    # Reconstruct
    reconstructed = contract(Q, R, axes=(1, 0))
    
    # Check each block individually
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


def test_qr_no_truncation():
    """Test that QR does not perform any truncation."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 10),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 10),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=888)
    
    Q, R = qr(T, axis=0)
    
    # Bond dimension should be full (no truncation)
    bond_dim = Q.indices[1].dim
    assert bond_dim == 10, "QR should not truncate"
    
    # Reconstruction should be exact
    reconstructed = contract(Q, R, axes=(1, 0))
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-12
    assert_blocks_equal(T, reconstructed, rtol=1e-10, atol=1e-12)


def test_qr_consistency_across_axes():
    """Test that QR gives consistent results when applied to different axes."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    
    T = Tensor.random([idx, idx.flip(), idx], itags=["a", "b", "c"], seed=999)
    
    # Apply QR to each axis
    results = []
    for axis in [0, 1, 2]:
        Q, R = qr(T, axis=axis)
        reconstructed = contract(Q, R, axes=(1, 0))
        
        # Permute back to original order
        if axis == 0:
            pass
        elif axis == 1:
            reconstructed.permute([1, 0, 2], in_place=True)
        elif axis == 2:
            reconstructed.permute([1, 2, 0], in_place=True)
        
        results.append(reconstructed)
    
    # All reconstructions should match the original
    for i, recon in enumerate(results):
        rel_error = (T - recon).norm() / T.norm()
        assert rel_error < 1e-12, f"Axis {i} reconstruction failed"
        assert_blocks_equal(T, recon, rtol=1e-10, atol=1e-12)


# SU(2) QR Tests

def test_qr_su2_reconstruction_2nd_order():
    """QR of a 2nd-order SU(2) tensor reconstructs the original for all decomposed axes."""
    T = _make_su2_2nd_order(seed=11)

    for axis in range(len(T.indices)):
        Q, R = qr(T, axis=axis)
        reconstructed = contract(Q, R)

        tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
        perm = [tag_to_pos_recon[tag] for tag in T.itags]
        reconstructed.permute(perm, in_place=True)

        assert_physical_tensors_equal(T, reconstructed, atol=1e-10,
                                      msg=f"SU(2) 2nd-order QR reconstruction axis={axis}")


def test_qr_su2_reconstruction_3rd_order():
    """QR of a 3rd-order SU(2) tensor reconstructs the original for all decomposed axes."""
    T = _make_su2_3rd_order(seed=10)

    for axis in range(len(T.indices)):
        Q, R = qr(T, axis=axis)
        reconstructed = contract(Q, R)

        tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
        perm = [tag_to_pos_recon[tag] for tag in T.itags]
        reconstructed.permute(perm, in_place=True)

        assert_physical_tensors_equal(T, reconstructed, atol=1e-10,
                                      msg=f"SU(2) 3rd-order QR reconstruction axis={axis}")


def test_qr_su2_reconstruction_4th_order():
    """QR of a 4th-order SU(2) tensor reconstructs the original for all decomposed axes."""
    T = _make_su2_4th_order(seed=12)

    for axis in range(len(T.indices)):
        Q, R = qr(T, axis=axis)
        reconstructed = contract(Q, R)

        tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
        perm = [tag_to_pos_recon[tag] for tag in T.itags]
        reconstructed.permute(perm, in_place=True)

        assert_physical_tensors_equal(T, reconstructed, atol=1e-10,
                                      msg=f"SU(2) 4th-order QR reconstruction axis={axis}")


def test_qr_su2_reconstruction_5th_order():
    """QR of a 5th-order SU(2) tensor reconstructs the original for all decomposed axes."""
    T = _make_su2_5th_order(seed=13)

    for axis in range(len(T.indices)):
        Q, R = qr(T, axis=axis)
        reconstructed = contract(Q, R)

        tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
        perm = [tag_to_pos_recon[tag] for tag in T.itags]
        reconstructed.permute(perm, in_place=True)

        assert_physical_tensors_equal(T, reconstructed, atol=1e-10,
                                      msg=f"SU(2) 5th-order QR reconstruction axis={axis}")


def test_qr_su2_reconstruction_6th_order():
    """QR of a 6th-order SU(2) tensor reconstructs the original for all decomposed axes."""
    T = _make_su2_6th_order(seed=14)

    for axis in range(len(T.indices)):
        Q, R = qr(T, axis=axis)
        reconstructed = contract(Q, R)

        tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
        perm = [tag_to_pos_recon[tag] for tag in T.itags]
        reconstructed.permute(perm, in_place=True)

        assert_physical_tensors_equal(T, reconstructed, atol=1e-10,
                                      msg=f"SU(2) 6th-order QR reconstruction axis={axis}")


def test_qr_su2_q_intertwiner_identity_like():
    """Q from SU(2) QR has identity-like intertwiner: one component, weights[0,0]=sqrt(irrep_dim)."""
    T = _make_su2_3rd_order(seed=20)

    Q, _R = qr(T, axis=0)

    assert Q.intw is not None, "Q must have intw for SU(2)"
    for key, bridge in Q.intw.items():
        q = key[0]
        assert bridge.weights.shape == (1, 1), \
            f"Q bridge weights should be (1,1), got {bridge.weights.shape}"
        expected = math.sqrt(T.group.irrep_dim(q))
        assert abs(bridge.weights[0, 0].item() - expected) < 1e-6, \
            f"Q bridge weights[0,0] should be sqrt(irrep_dim({q}))={expected:.4f}"


def test_qr_su2_q_data_blocks_have_trailing_component_dim():
    """Q blocks from SU(2) QR have a trailing OM axis of size 1."""
    T = _make_su2_3rd_order(seed=22)

    Q, _R = qr(T, axis=0)

    for key, arr in Q.data.items():
        assert arr.ndim == 3, f"Q block {key} should be 3D, got {arr.ndim}D"
        assert arr.shape[-1] == 1, \
            f"Q block {key} trailing dim should be 1, got {arr.shape[-1]}"


def test_qr_su2_q_reduced_blocks_are_isometric():
    """Q from SU(2) QR: reduced blocks satisfy Q^†Q = I."""
    T = _make_su2_3rd_order(seed=23)

    Q, _R = qr(T, axis=0)

    import torch
    for key, arr in Q.data.items():
        q_mat = arr.squeeze(-1)  # (d_left, rank)
        prod = q_mat.conj().T @ q_mat  # (rank, rank)
        eye = torch.eye(prod.shape[0], dtype=T.dtype)
        assert torch.allclose(prod, eye, atol=1e-10), \
            f"Q block {key}: Q^†Q deviates from identity, max err={( prod - eye).abs().max():.2e}"


def test_qr_su2_r_intertwiner_matches_t_axis0():
    """For axis=0, R from SU(2) QR has intertwiner with same keys and weights as T."""
    T = _make_su2_3rd_order(seed=30)

    _Q, R = qr(T, axis=0)

    assert R.intw is not None, "R must have intw for SU(2)"
    assert set(R.intw.keys()) == set(T.intw.keys()), \
        "R intw keys should match T intw keys when axis=0 (identity permutation)"
    for key, bridge in R.intw.items():
        t_weights = T.intw[key].weights.to(dtype=bridge.weights.dtype)
        assert torch.allclose(bridge.weights, t_weights, atol=1e-10), \
            f"R intw weights differ from T intw weights at key {key}"


def test_qr_su2_r_has_correct_intw_keys():
    """R from SU(2) QR has intertwiner keys matching its index count."""
    T = _make_su2_3rd_order(seed=31)

    _Q, R = qr(T, axis=0)

    assert R.intw is not None
    for key in R.intw.keys():
        assert len(key) == len(R.indices), \
            f"R intw key length {len(key)} should match R index count {len(R.indices)}"


def test_qr_su2_non_zero_axis_r_intw_not_none():
    """R from SU(2) QR has a non-None intertwiner even when axis != 0."""
    T = _make_su2_3rd_order(seed=41)

    for axis in range(1, len(T.indices)):
        _Q, R = qr(T, axis=axis)
        assert R.intw is not None, f"R.intw must not be None for axis={axis}"


def test_qr_su2_charge_neutral():
    """Q and R from SU(2) QR are charge neutral."""
    T = _make_su2_3rd_order(seed=60)

    Q, R = qr(T, axis=0)

    assert_charge_neutral(Q)
    assert_charge_neutral(R)


# =============================================================================
#  Eigen-decomposition Tests
# =============================================================================

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
    
    U, D = eig(T, is_hermitian=True)
    
    # Check dimensions
    assert len(U.indices) == 2
    assert U.itags[0] == "i"
    assert U.itags[1] == "_bond_eig"
    
    # Check eigenvalues are real for Hermitian matrix
    for key, eigvals in D.items():
        assert eigvals.is_floating_point(), \
            f"Expected real eigenvalues from eigh, got {eigvals.dtype} at {key}"
    
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
        U_D = U_block @ torch.diag(D_block)
        
        assert torch.allclose(T_U, U_D, atol=1e-10)


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
        D_diag = torch.diag(D_block)
        T_reconstructed = U_block @ D_diag @ U_block.T.conj()
        
        rel_error = torch.linalg.norm(T_block - T_reconstructed).item() / torch.linalg.norm(T_block).item()
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
    
    # Keep only top 3 eigenvalues (largest/most positive)
    U, D = eig(T, order="descend", trunc={"nkeep": 3})
    
    # Count total eigenvalues
    total_eigvals = sum(len(eigvals) for eigvals in D.values())
    assert total_eigvals == 3
    
    # Verify they are the largest (most positive) ones
    all_eigvals_full = []
    for key in T.data.keys():
        q_row, q_col = key
        if q_row != q_col:
            continue
        eigvals_full, _ = torch.linalg.eig(T.data[key])
        # For Hermitian matrix, eigenvalues are real - use real part for comparison
        all_eigvals_full.append(torch.real(eigvals_full))
    
    all_eigvals_full = torch.cat(all_eigvals_full)
    all_eigvals_full, _ = torch.sort(all_eigvals_full, descending=True)
    top_3_expected = all_eigvals_full[:3]
    
    # Collect truncated eigenvalues as tensors
    all_eigvals_truncated = torch.cat([torch.real(eigvals) for eigvals in D.values()])
    all_eigvals_truncated, _ = torch.sort(all_eigvals_truncated, descending=True)
    
    assert torch.allclose(all_eigvals_truncated, top_3_expected, atol=1e-10)


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
    
    # Keep eigenvalues >= 1.0 (using descending order)
    U, D = eig(T, order="descend", trunc={"thresh": 1.0})
    
    # Verify all kept eigenvalues satisfy threshold
    for eigvals in D.values():
        assert torch.all(torch.real(eigvals) >= 1.0).item()


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
    
    # Apply both truncations: first thresh >= 0.8, then nkeep top 4 (largest)
    U, D = eig(T, order="descend", trunc={"thresh": 0.8, "nkeep": 4})
    
    # Count total eigenvalues
    total_eig = sum(len(eigvals) for eigvals in D.values())
    
    # Should have at most 4 eigenvalues (nkeep limit)
    assert total_eig <= 4
    
    # All eigenvalues should satisfy >= 0.8 (thresh limit with descending order)
    for eigvals in D.values():
        assert torch.all(torch.real(eigvals) >= 0.8).item()


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
    data = {(0, 0): torch.tensor([[1+1j, 2-1j], [2+1j, 3-2j]], dtype=torch.complex128)}
    T = Tensor(
        indices=(idx_out, idx_in),
        itags=("i", "j"),
        data=data,
        dtype=torch.complex128
    )
    
    U, D = eig(T)
    
    # Verify eigendecomposition
    T_block = T.data[(0, 0)]
    U_block = U.data[(0, 0)]
    D_block = D[(0, 0)]
    
    # T @ U = U @ diag(D)
    T_U = T_block @ U_block
    U_D = U_block @ torch.diag(D_block)
    
    assert torch.allclose(T_U, U_D)


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


def test_eig_order_ascending_mixed_signs():
    """Test ascending order with mixed positive and negative eigenvalues."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 5),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 5),))
    
    # Create diagonal matrix with known eigenvalues: [-5, -2, 0, 3, 7]
    data = {(0, 0): torch.diag(torch.tensor([-5.0, -2.0, 0.0, 3.0, 7.0], dtype=torch.float64))}
    T = Tensor(indices=(idx_out, idx_in), itags=("i", "j"), data=data, dtype=torch.float64)
    
    U, D = eig(T, order="ascend")
    eigvals = torch.real(D[(0, 0)])
    
    # Should be sorted from smallest to largest
    expected = torch.tensor([-5.0, -2.0, 0.0, 3.0, 7.0], dtype=eigvals.dtype)
    assert torch.allclose(eigvals, expected)


def test_eig_order_descending_mixed_signs():
    """Test descending order with mixed positive and negative eigenvalues."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 5),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 5),))
    
    # Create diagonal matrix with known eigenvalues: [-5, -2, 0, 3, 7]
    data = {(0, 0): torch.diag(torch.tensor([-5.0, -2.0, 0.0, 3.0, 7.0], dtype=torch.float64))}
    T = Tensor(indices=(idx_out, idx_in), itags=("i", "j"), data=data, dtype=torch.float64)
    
    U, D = eig(T, order="descend")
    eigvals = torch.real(D[(0, 0)])
    
    # Should be sorted from largest to smallest
    expected = torch.tensor([7.0, 3.0, 0.0, -2.0, -5.0], dtype=eigvals.dtype)
    assert torch.allclose(eigvals, expected)


def test_eig_thresh_ascending_negative():
    """Test thresh mode with ascending order for negative eigenvalues."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 6),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 6),))
    
    # Eigenvalues: [-8, -5, -3, -1, 2, 4]
    data = {(0, 0): torch.diag(torch.tensor([-8.0, -5.0, -3.0, -1.0, 2.0, 4.0], dtype=torch.float64))}
    T = Tensor(indices=(idx_out, idx_in), itags=("i", "j"), data=data, dtype=torch.float64)
    
    # Keep eigenvalues <= -2 (should keep: -8, -5, -3)
    U, D = eig(T, order="ascend", trunc={"thresh": -2.0})
    eigvals = torch.real(D[(0, 0)])
    
    expected = torch.tensor([-8.0, -5.0, -3.0], dtype=eigvals.dtype)
    assert torch.allclose(eigvals, expected)


def test_eig_thresh_descending_positive():
    """Test thresh mode with descending order for positive eigenvalues."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 6),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 6),))
    
    # Eigenvalues: [-4, -2, 1, 3, 5, 8]
    data = {(0, 0): torch.diag(torch.tensor([-4.0, -2.0, 1.0, 3.0, 5.0, 8.0], dtype=torch.float64))}
    T = Tensor(indices=(idx_out, idx_in), itags=("i", "j"), data=data, dtype=torch.float64)
    
    # Keep eigenvalues >= 2 (should keep: 8, 5, 3)
    U, D = eig(T, order="descend", trunc={"thresh": 2.0})
    eigvals = torch.real(D[(0, 0)])
    
    expected = torch.tensor([8.0, 5.0, 3.0], dtype=eigvals.dtype)
    assert torch.allclose(eigvals, expected)


def test_eig_thresh_zero_boundary():
    """Test thresh mode at zero boundary for filtering positive/negative."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 5),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 5),))
    
    # Eigenvalues: [-3, -1, 0, 2, 4]
    data = {(0, 0): torch.diag(torch.tensor([-3.0, -1.0, 0.0, 2.0, 4.0], dtype=torch.float64))}
    T = Tensor(indices=(idx_out, idx_in), itags=("i", "j"), data=data, dtype=torch.float64)
    
    # Keep only non-negative eigenvalues (>= 0)
    U_pos, D_pos = eig(T, order="descend", trunc={"thresh": 0.0})
    eigvals_pos = torch.real(D_pos[(0, 0)])
    assert torch.allclose(eigvals_pos, torch.tensor([4.0, 2.0, 0.0], dtype=eigvals_pos.dtype))
    
    # Keep only non-positive eigenvalues (<= 0)
    U_neg, D_neg = eig(T, order="ascend", trunc={"thresh": 0.0})
    eigvals_neg = torch.real(D_neg[(0, 0)])
    assert torch.allclose(eigvals_neg, torch.tensor([-3.0, -1.0, 0.0], dtype=eigvals_neg.dtype))


def test_eig_nkeep_ascending_ground_states():
    """Test nkeep with ascending order to get ground states."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 6),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 6),))
    
    # Eigenvalues: [-10, -5, -2, 1, 3, 8]
    data = {(0, 0): torch.diag(torch.tensor([-10.0, -5.0, -2.0, 1.0, 3.0, 8.0], dtype=torch.float64))}
    T = Tensor(indices=(idx_out, idx_in), itags=("i", "j"), data=data, dtype=torch.float64)
    
    # Keep 3 smallest (ground state + 2 excited states)
    U, D = eig(T, order="ascend", trunc={"nkeep": 3})
    eigvals = torch.real(D[(0, 0)])
    
    expected = torch.tensor([-10.0, -5.0, -2.0], dtype=eigvals.dtype)
    assert torch.allclose(eigvals, expected)


def test_eig_nkeep_descending_excited_states():
    """Test nkeep with descending order to get highest excited states."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 6),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 6),))
    
    # Eigenvalues: [-10, -5, -2, 1, 3, 8]
    data = {(0, 0): torch.diag(torch.tensor([-10.0, -5.0, -2.0, 1.0, 3.0, 8.0], dtype=torch.float64))}
    T = Tensor(indices=(idx_out, idx_in), itags=("i", "j"), data=data, dtype=torch.float64)
    
    # Keep 3 largest (highest excited states)
    U, D = eig(T, order="descend", trunc={"nkeep": 3})
    eigvals = torch.real(D[(0, 0)])
    
    expected = torch.tensor([8.0, 3.0, 1.0], dtype=eigvals.dtype)
    assert torch.allclose(eigvals, expected)


def test_eig_combined_thresh_nkeep_ascending():
    """Test combined thresh and nkeep with ascending order."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 8),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 8),))
    
    # Eigenvalues: [-10, -8, -6, -4, -2, 0, 2, 4]
    data = {(0, 0): torch.diag(torch.tensor([-10.0, -8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0], dtype=torch.float64))}
    T = Tensor(indices=(idx_out, idx_in), itags=("i", "j"), data=data, dtype=torch.float64)
    
    # Keep eigenvalues <= -3, then keep 2 smallest
    # After thresh: [-10, -8, -6, -4]
    # After nkeep: [-10, -8]
    U, D = eig(T, order="ascend", trunc={"thresh": -3.0, "nkeep": 2})
    eigvals = torch.real(D[(0, 0)])
    
    expected = torch.tensor([-10.0, -8.0], dtype=eigvals.dtype)
    assert torch.allclose(eigvals, expected)


def test_eig_combined_thresh_nkeep_descending():
    """Test combined thresh and nkeep with descending order."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 8),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 8),))
    
    # Eigenvalues: [-4, -2, 0, 2, 4, 6, 8, 10]
    data = {(0, 0): torch.diag(torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=torch.float64))}
    T = Tensor(indices=(idx_out, idx_in), itags=("i", "j"), data=data, dtype=torch.float64)
    
    # Keep eigenvalues >= 3, then keep 2 largest
    # After thresh: [10, 8, 6, 4]
    # After nkeep: [10, 8]
    U, D = eig(T, order="descend", trunc={"thresh": 3.0, "nkeep": 2})
    eigvals = torch.real(D[(0, 0)])
    
    expected = torch.tensor([10.0, 8.0], dtype=eigvals.dtype)
    assert torch.allclose(eigvals, expected)


def test_eig_order_with_complex_eigenvalues():
    """Test order parameter with complex eigenvalues (sorts by real part)."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    
    # Create a non-Hermitian matrix with complex eigenvalues
    # Using a simple matrix that we know has complex eigenvalues
    mat = torch.tensor([[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]], dtype=torch.complex128)
    data = {(0, 0): mat}
    T = Tensor(indices=(idx_out, idx_in), itags=("i", "j"), data=data, dtype=torch.complex128)
    
    U_asc, D_asc = eig(T, order="ascend")
    eigvals_asc = D_asc[(0, 0)]
    
    # Check that sorting is by real part (ascending)
    real_parts_asc = torch.real(eigvals_asc)
    assert torch.all(real_parts_asc[:-1] <= real_parts_asc[1:]).item()
    
    U_desc, D_desc = eig(T, order="descend")
    eigvals_desc = D_desc[(0, 0)]
    
    # Check that sorting is by real part (descending)
    real_parts_desc = torch.real(eigvals_desc)
    assert torch.all(real_parts_desc[:-1] >= real_parts_desc[1:]).item()


def test_eig_thresh_filters_all_eigenvalues():
    """Test thresh mode when all eigenvalues are filtered out."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 4),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 4),))
    
    # Eigenvalues: [1, 2, 3, 4]
    data = {(0, 0): torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64))}
    T = Tensor(indices=(idx_out, idx_in), itags=("i", "j"), data=data, dtype=torch.float64)
    
    # Keep eigenvalues <= 0 (should filter all)
    U, D = eig(T, order="ascend", trunc={"thresh": 0.0})
    
    # Should have no eigenvalues
    assert len(D) == 0


def test_eig_order_multiple_blocks():
    """Test that order parameter works correctly with multiple charge blocks."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2)))
    
    # Create blocks with known eigenvalues
    # Block (0,0): eigenvalues [-2, 0, 5]
    # Block (1,1): eigenvalues [-3, 1]
    data = {
        (0, 0): torch.diag(torch.tensor([-2.0, 0.0, 5.0], dtype=torch.float64)),
        (1, 1): torch.diag(torch.tensor([-3.0, 1.0], dtype=torch.float64))
    }
    T = Tensor(indices=(idx_out, idx_in), itags=("i", "j"), data=data, dtype=torch.float64)
    
    # Test ascending - each block should be sorted independently
    U_asc, D_asc = eig(T, order="ascend")
    eigvals_asc_00 = torch.real(D_asc[(0, 0)])
    eigvals_asc_11 = torch.real(D_asc[(1, 1)])
    assert torch.allclose(eigvals_asc_00, torch.tensor([-2.0, 0.0, 5.0], dtype=eigvals_asc_00.dtype))
    assert torch.allclose(eigvals_asc_11, torch.tensor([-3.0, 1.0], dtype=eigvals_asc_11.dtype))
    
    # Test descending - each block should be sorted independently
    U_desc, D_desc = eig(T, order="descend")
    eigvals_desc_00 = torch.real(D_desc[(0, 0)])
    eigvals_desc_11 = torch.real(D_desc[(1, 1)])
    assert torch.allclose(eigvals_desc_00, torch.tensor([5.0, 0.0, -2.0], dtype=eigvals_desc_00.dtype))
    assert torch.allclose(eigvals_desc_11, torch.tensor([1.0, -3.0], dtype=eigvals_desc_11.dtype))
    
    # Test nkeep across blocks - should keep 3 smallest globally
    U_nkeep, D_nkeep = eig(T, order="ascend", trunc={"nkeep": 3})
    all_eigvals = torch.cat([torch.real(eigvals) for eigvals in D_nkeep.values()])
    all_eigvals, _ = torch.sort(all_eigvals)
    assert torch.allclose(all_eigvals, torch.tensor([-3.0, -2.0, 0.0], dtype=all_eigvals.dtype))


# SU(2) Eigendecomposition Tests

def test_eig_su2_u_has_intw():
    """U from SU(2) eig has a non-None intertwiner."""
    T = _make_su2_2nd_order(seed=70)

    U, _D = eig(T)

    assert U.intw is not None, "U must have intw for SU(2)"


def test_eig_su2_u_intertwiner_identity_like():
    """U from SU(2) eig has identity-like intertwiner: one component, weights[0,0]=sqrt(irrep_dim)."""
    T = _make_su2_2nd_order(seed=71)

    U, _D = eig(T)

    assert U.intw is not None
    for key, bridge in U.intw.items():
        q = key[0]
        assert bridge.weights.shape == (1, 1), \
            f"U bridge weights should be (1,1), got {bridge.weights.shape}"
        expected = math.sqrt(T.group.irrep_dim(q))
        assert abs(bridge.weights[0, 0].item() - expected) < 1e-6, \
            f"U bridge weights[0,0] should be sqrt(irrep_dim({q}))={expected:.4f}"


def test_eig_su2_u_data_blocks_have_trailing_component_dim():
    """U blocks from SU(2) eig have a trailing OM axis of size 1."""
    T = _make_su2_2nd_order(seed=72)

    U, _D = eig(T)

    for key, arr in U.data.items():
        assert arr.ndim == 3, f"U block {key} should be 3D, got {arr.ndim}D"
        assert arr.shape[-1] == 1, \
            f"U block {key} trailing dim should be 1, got {arr.shape[-1]}"


def test_eig_su2_eigendecomposition_relation():
    """For each block: T_block @ U_block = U_block @ diag(D_block) (reduced matrix elements)."""
    T = _make_su2_2nd_order(seed=73)

    U, D = eig(T)

    for key, t_arr in T.data.items():
        q = key[0]
        u_key = (q, q)
        if u_key not in U.data:
            continue

        T_mat = t_arr.squeeze(-1).to(dtype=U.dtype)      # (d_q, d_q)
        U_mat = U.data[u_key].squeeze(-1)                 # (d_q, rank)
        D_vec = D[u_key]                                  # (rank,)

        lhs = T_mat @ U_mat
        rhs = U_mat @ torch.diag(D_vec.to(dtype=U.dtype))
        assert torch.allclose(lhs, rhs, atol=1e-8), \
            f"Eigendecomposition relation T@U = U@diag(D) failed for charge {q}, " \
            f"max err={( lhs - rhs).abs().max():.2e}"


def test_eig_su2_reconstruction():
    """Physical tensor T = U @ diag(D) @ U† reconstructed via the tensor API."""
    T = _make_su2_2nd_order(seed=76)

    # Symmetrize so U is orthonormal and T = U @ diag(D) @ U† holds exactly.
    for key in list(T.data.keys()):
        mat = T.data[key].squeeze(-1)
        T.data[key] = ((mat + mat.T.conj()) / 2).unsqueeze(-1)

    # Clone BEFORE eig, which mutates T in-place via regularize().
    # T_ref holds the physical values we want to reconstruct.
    T_ref = T.clone()

    U, D = eig(T, is_hermitian=True)

    # Build diagonal matrix tensor from eigenvalues.
    # Give both bond axes the same itag as U's bond so contract's itag check passes.
    bond_index = U.indices[1]
    bond_tag = U.itags[1]
    D_diag = diag(D, bond_index, itags=(bond_tag, bond_tag))

    # Reconstruct: T = U @ D_diag @ U†
    U_dag = U.conj()
    D_Udag = contract(D_diag, U_dag, axes=(1, 1))
    T_recon = contract(U, D_Udag, axes=(1, 0))
    T_recon.retag(T_ref.itags)

    assert_physical_tensors_equal(T_ref, T_recon, atol=1e-8,
                                  msg="SU(2) eig physical reconstruction T = U@diag(D)@U†")


def test_eig_su2_charge_neutral():
    """U from SU(2) eig is charge neutral."""
    T = _make_su2_2nd_order(seed=74)

    U, _D = eig(T)

    assert_charge_neutral(U)


def test_eig_su2_truncation_nkeep():
    """SU(2) eig with nkeep truncation produces the correct number of eigenvalues."""
    T = _make_su2_2nd_order(seed=75)

    U, D = eig(T, order="descend", trunc={"nkeep": 2})

    total = sum(len(vals) for vals in D.values())
    assert total == 2, f"Expected 2 eigenvalues after nkeep=2, got {total}"
    assert U.intw is not None, "U.intw must not be None after truncation"
    # intw keys must match U.data keys exactly
    assert set(U.intw.keys()) == set(U.data.keys()), \
        "U.intw keys must match U.data keys after nkeep truncation"


# is_hermitian flag tests (Abelian / non-Abelian) x (Hermitian / non-Hermitian)

def test_eig_is_hermitian_abelian_hermitian():
    """Abelian (U1) + is_hermitian=True: eigenvalues are real, eigenvectors orthonormal."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(0, 3), Sector(1, 2)))
    T = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=80)
    # Symmetrize each block so T is Hermitian
    for key in list(T.data.keys()):
        T.data[key] = (T.data[key] + T.data[key].T.conj()) / 2

    U, D = eig(T, is_hermitian=True)

    # eigh returns real eigenvalues
    for key, vals in D.items():
        assert vals.is_floating_point(), \
            f"Expected real eigenvalues from eigh, got {vals.dtype} at {key}"

    # eigenvectors must be orthonormal per block
    for key, arr in U.data.items():
        U_mat = arr  # (d, rank) for Abelian
        prod = U_mat.T.conj() @ U_mat
        eye  = torch.eye(prod.shape[0], dtype=prod.dtype)
        assert torch.allclose(prod, eye, atol=1e-10), \
            f"U†U ≠ I at block {key}, max err={( prod - eye).abs().max():.2e}"


def test_eig_is_hermitian_abelian_nonhermitian():
    """Abelian (U1) + is_hermitian=False: decomposition holds, eigenvectors not orthonormal."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(0, 3), Sector(1, 2)))
    T = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=81)
    # Do NOT symmetrize — blocks are generically asymmetric

    U, D = eig(T)  # is_hermitian=False by default

    # eigendecomposition relation must hold: T @ U = U @ diag(D)
    for key, t_arr in T.data.items():
        if key not in U.data:
            continue
        T_mat = t_arr.to(dtype=U.dtype)
        U_mat = U.data[key]
        D_vec = D[key]
        lhs = T_mat @ U_mat
        rhs = U_mat @ torch.diag(D_vec)
        assert torch.allclose(lhs, rhs, atol=1e-8), \
            f"T@U = U@diag(D) failed at {key}, max err={( lhs - rhs).abs().max():.2e}"

    # eigenvectors from the general path are NOT guaranteed to be orthonormal
    any_non_orthonormal = False
    for key, arr in U.data.items():
        prod = arr.T.conj() @ arr
        eye  = torch.eye(prod.shape[0], dtype=prod.dtype)
        if not torch.allclose(prod, eye, atol=1e-10):
            any_non_orthonormal = True
            break
    assert any_non_orthonormal, \
        "Expected at least one non-orthonormal eigenvector block for a non-symmetric matrix"


def test_eig_is_hermitian_su2_hermitian():
    """SU(2) + is_hermitian=True: eigenvalues are real, reduced eigenvectors orthonormal."""
    T = _make_su2_2nd_order(seed=82)
    # Symmetrize the reduced data blocks so T is Hermitian
    for key in list(T.data.keys()):
        mat = T.data[key].squeeze(-1)
        T.data[key] = ((mat + mat.T.conj()) / 2).unsqueeze(-1)

    U, D = eig(T, is_hermitian=True)

    # eigh returns real eigenvalues
    for key, vals in D.items():
        assert vals.is_floating_point(), \
            f"Expected real eigenvalues from eigh, got {vals.dtype} at {key}"

    # reduced eigenvector blocks must be orthonormal
    for key, arr in U.data.items():
        U_mat = arr.squeeze(-1)  # (d, rank)
        prod  = U_mat.T.conj() @ U_mat
        eye   = torch.eye(prod.shape[0], dtype=prod.dtype)
        assert torch.allclose(prod, eye, atol=1e-10), \
            f"U†U ≠ I at SU(2) block {key}, max err={( prod - eye).abs().max():.2e}"


def test_eig_is_hermitian_su2_nonhermitian():
    """SU(2) + is_hermitian=False: decomposition holds, reduced eigenvectors not orthonormal."""
    T = _make_su2_2nd_order(seed=83)
    # Do NOT symmetrize — blocks are generically asymmetric

    U, D = eig(T)  # is_hermitian=False by default

    # eigendecomposition relation on reduced blocks: T_mat @ U_mat = U_mat @ diag(D_vec)
    for key, t_arr in T.data.items():
        q = key[0]
        u_key = (q, q)
        if u_key not in U.data:
            continue
        T_mat = t_arr.squeeze(-1).to(dtype=U.dtype)
        U_mat = U.data[u_key].squeeze(-1)
        D_vec = D[u_key]
        lhs = T_mat @ U_mat
        rhs = U_mat @ torch.diag(D_vec.to(dtype=U.dtype))
        assert torch.allclose(lhs, rhs, atol=1e-8), \
            f"T@U = U@diag(D) failed at SU(2) block {key}, max err={( lhs - rhs).abs().max():.2e}"

    # eigenvectors from the general path are NOT guaranteed to be orthonormal
    any_non_orthonormal = False
    for key, arr in U.data.items():
        U_mat = arr.squeeze(-1)
        prod  = U_mat.T.conj() @ U_mat
        eye   = torch.eye(prod.shape[0], dtype=prod.dtype)
        if not torch.allclose(prod, eye, atol=1e-10):
            any_non_orthonormal = True
            break
    assert any_non_orthonormal, \
        "Expected at least one non-orthonormal reduced eigenvector block for a non-symmetric SU(2) tensor"
