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


"""Tests for tensor decomposition operations: decomp() high-level function."""

import torch
import pytest

from nicole import Direction, Tensor, contract, decomp, U1Group, Index, Sector
from nicole.decomp import svd
from ..utils import assert_charge_neutral


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
        off_diag = block - torch.diag(torch.diag(block))
        assert torch.allclose(off_diag, torch.zeros_like(off_diag), atol=1e-14)


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


def test_decomp_qr_mode():
    """Test decomp in QR mode."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=42)
    
    Q, R = decomp(T, axes=0, mode="QR")
    
    # Check structure
    assert len(Q.indices) == 2
    assert len(R.indices) == 2
    
    # Reconstruct using explicit pairs
    reconstructed = contract(Q, R, axes=(1, 0))
    
    # Verify accuracy
    diff_norm = (T - reconstructed).norm()
    assert diff_norm / T.norm() < 1e-12


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


def test_decomp_qr_multiindex():
    """Test QR mode with multi-index tensor."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=789)
    
    Q, R = decomp(T, axes=1, mode="QR")
    
    # Check structure: Q should have 2 indices, R should have 3 indices
    assert len(Q.indices) == 2
    assert len(R.indices) == 3
    
    # Reconstruct using explicit pairs
    reconstructed = contract(Q, R, axes=(1, 0))
    
    # Permute back to original order (b, a, c) -> (a, b, c)
    reconstructed.permute([1, 0, 2])
    
    # Verify accuracy
    diff_norm = (T - reconstructed).norm()
    assert diff_norm / T.norm() < 1e-12


def test_decomp_mode_case_insensitive():
    """Test that decomp mode is case-insensitive."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=789)
    
    # All of these should work
    U1, R1 = decomp(T, axes=0, mode="UR")
    U2, R2 = decomp(T, axes=0, mode="ur")
    U3, R3 = decomp(T, axes=0, mode="Ur")
    
    # Should give same results
    assert (U1 - U2).norm() / U1.norm() < 1e-15
    assert (R1 - R2).norm() / R1.norm() < 1e-15


def test_decomp_preserves_charge_neutrality():
    """Test that decomposition preserves charge neutrality."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(-1, 2), Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(-1, 1), Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=111)
    
    # Original should be charge neutral
    assert_charge_neutral(T)
    
    # After decomposition
    U, R = decomp(T, axes=0, mode="UR")
    
    # U and R should each be charge neutral
    assert_charge_neutral(U)
    assert_charge_neutral(R)


def test_decomp_ur_efficiency():
    """Test UR mode is more efficient than SVD for reconstruction."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 10),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 10),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=1000)
    
    # UR mode returns 2 tensors
    result_ur = decomp(T, axes=0, mode="UR")
    assert len(result_ur) == 2
    
    # SVD mode returns 3 tensors
    result_svd = decomp(T, axes=0, mode="SVD")
    assert len(result_svd) == 3


def test_decomp_lv_efficiency():
    """Test LV mode is more efficient than SVD for reconstruction."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 10),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 10),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=2000)
    
    # LV mode returns 2 tensors
    result_lv = decomp(T, axes=0, mode="LV")
    assert len(result_lv) == 2
    
    # SVD mode returns 3 tensors
    result_svd = decomp(T, axes=0, mode="SVD")
    assert len(result_svd) == 3


def test_decomp_truncation_ur_mode():
    """Test truncation in UR mode."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 10),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 10),))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=3000)
    
    # Truncate to 3 values
    U, R = decomp(T, axes=0, mode="UR", trunc={"nkeep": 3})
    
    # Bond dimension should be at most 3
    bond_dim = U.indices[1].dim
    assert bond_dim <= 3


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


def test_decomp_flow_qr_mode_default():
    """Test QR mode with default flow (should normalize to >>)."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=300)
    
    # QR with default flow "><" (should normalize to ">>")
    Q, R = decomp(T, axes=0, mode="QR")
    
    # Check flow: "><" normalizes to ">>" which means bonds point outward
    # Q's bond is OUT (pointing out from Q), R's bond is IN (pointing into R from outside)
    assert Q.indices[1].direction == Direction.OUT
    assert R.indices[0].direction == Direction.IN
    
    # Reconstruct
    reconstructed = contract(Q, R, axes=(1, 0))
    
    # Verify accuracy
    assert (T - reconstructed).norm() / T.norm() < 1e-12


def test_decomp_flow_qr_mode_explicit():
    """Test QR mode with explicit flow values."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=301)
    
    # QR with flow ">>" (bonds point outward: Q bond is OUT, R bond is IN)
    Q1, R1 = decomp(T, axes=0, mode="QR", flow=">>")
    assert Q1.indices[1].direction == Direction.OUT
    assert R1.indices[0].direction == Direction.IN
    
    # QR with flow "<<" (bonds point inward: Q bond is IN, R bond is OUT)
    Q2, R2 = decomp(T, axes=0, mode="QR", flow="<<")
    assert Q2.indices[1].direction == Direction.IN
    assert R2.indices[0].direction == Direction.OUT
    
    # Both should reconstruct correctly
    recon1 = contract(Q1, R1, axes=(1, 0))
    recon2 = contract(Q2, R2, axes=(1, 0))
    
    assert (T - recon1).norm() / T.norm() < 1e-12
    assert (T - recon2).norm() / T.norm() < 1e-12


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


def test_decomp_itag_qr_mode():
    """Test QR mode with custom itag."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=302)
    
    # Test with single string itag
    Q, R = decomp(T, axes=0, mode="QR", itag="qr_bond")
    assert Q.itags[1] == "qr_bond"
    assert R.itags[0] == "qr_bond"
    
    # Verify reconstruction
    reconstructed = contract(Q, R, axes=(1, 0))
    assert (T - reconstructed).norm() / T.norm() < 1e-12


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


def test_decomp_multi_axis_qr():
    """Test decomp with multiple axes in QR mode."""
    group = U1Group()
    indices = [
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2))),
        Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2)))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d"], seed=401)
    
    # QR on axes [0, 2] (merges them into Q)
    Q, R = decomp(T, axes=[0, 2], mode="QR")
    
    # Q should have 3 indices: (a, c, bond)
    # R should have 3 indices: (bond, b, d)
    assert len(Q.indices) == 3
    assert len(R.indices) == 3
    assert 'a' in Q.itags and 'c' in Q.itags
    assert 'b' in R.itags and 'd' in R.itags
    
    # Reconstruct
    reconstructed = contract(Q, R)
    assert set(reconstructed.itags) == {'a', 'b', 'c', 'd'}


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
            assert torch.allclose(T.data[key], reconstructed.data[key], rtol=1e-10, atol=1e-12)


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
        assert torch.allclose(reconstructed.data[key], T.data[key], atol=1e-10)


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
        assert torch.allclose(recon1.data[key], T.data[key], rtol=1e-10, atol=1e-12)


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
        sv = torch.diag(block)
        all_sv.extend(sv)
    
    # Should have at most 5 values (nkeep)
    assert len(all_sv) <= 5
    
    # All should be >= 0.5 (thresh)
    assert torch.all(torch.tensor(all_sv) >= 0.5).item()
    
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


# ===== High-Order Tensor Tests =====

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
        assert torch.all(s_array[:-1] >= s_array[1:]).item(), "Singular values should be sorted descending"


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
        assert torch.all(s_array >= threshold).item()
    
    # Verify we can still reconstruct (approximately)
    U_full = decomp(T, axes=0, mode="UR", trunc={"thresh": threshold})[0]
    assert len(U_full.indices) == 2
    # Left index should be unchanged (sum of all sector dimensions)
    expected_left_dim = sum(s.dim for s in indices[0].sectors)
    assert U_full.indices[0].dim == expected_left_dim


# ===== High-Order SVD-Related Mode Stress Tests =====

def test_decomp_svd_ultra_high_order_complex_charges():
    """Stress test: SVD mode with 6-index tensor and complex charge structure."""
    group = U1Group()
    # Create a very high-order tensor (6 indices)
    indices = [
        Index(Direction.OUT, group, sectors=(
            Sector(-3, 1), Sector(-2, 2), Sector(-1, 2), Sector(0, 3), Sector(1, 2), Sector(2, 2), Sector(3, 1)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-2, 2), Sector(-1, 3), Sector(0, 4), Sector(1, 3), Sector(2, 2)
        )),
        Index(Direction.OUT, group, sectors=(
            Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 1)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-3, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(3, 1)
        )),
        Index(Direction.OUT, group, sectors=(
            Sector(-2, 1), Sector(0, 3), Sector(2, 1)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-2, 1), Sector(-1, 1), Sector(0, 2), Sector(1, 1), Sector(2, 1)
        ))
    ]
    
    T = Tensor.random(indices, itags=["w", "x", "y", "z", "u", "v"], seed=8000)
    
    # SVD decomposition on axes [0, 2, 4]
    U, S, Vh = decomp(T, axes=[0, 2, 4], mode="SVD", flow=">>")
    
    # Verify structure
    assert len(U.indices) == 4  # w, y, u, bond_L
    assert len(S.data) > 0  # S is a dict-like with charge blocks
    assert len(Vh.indices) == 4  # bond_R, x, z, v
    assert 'w' in U.itags and 'y' in U.itags and 'u' in U.itags
    assert 'x' in Vh.itags and 'z' in Vh.itags and 'v' in Vh.itags
    
    # Check bond directions for flow=">>"
    assert U.indices[-1].direction == Direction.OUT
    assert Vh.indices[0].direction == Direction.IN
    
    # Verify bond has many charge sectors
    bond_charges_u = set(U.indices[-1].charges())
    bond_charges_vh = set(Vh.indices[0].charges())
    assert len(bond_charges_u) >= 4, f"Should have at least 4 charge sectors, got {len(bond_charges_u)}"
    
    # Verify that many blocks exist
    assert len(U.data) >= 8, f"U should have at least 8 blocks, got {len(U.data)}"
    assert len(Vh.data) >= 8, f"Vh should have at least 8 blocks, got {len(Vh.data)}"
    assert len(S.data) >= 8, f"S should have at least 8 blocks, got {len(S.data)}"
    
    # Verify charge neutrality
    assert_charge_neutral(U)
    assert_charge_neutral(Vh)
    
    # Test with custom itags
    U2, S2, Vh2 = decomp(T, axes=[0, 2, 4], mode="SVD", itag=("_bond_L_", "_bond_R_"))
    assert "_bond_L_" in U2.itags
    assert "_bond_R_" in Vh2.itags
    
    # Reconstruct using S as diagonal tensor and verify accuracy
    from nicole import contract
    # Use automatic contraction (bond tags match)
    US = contract(U, S)
    reconstructed = contract(US, Vh)
    
    # Permute to match original order
    tag_to_pos_orig = {tag: i for i, tag in enumerate(T.itags)}
    tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
    perm = [tag_to_pos_recon[tag] for tag in T.itags]
    reconstructed.permute(perm)
    
    # Check reconstruction accuracy
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-11, f"Reconstruction error too large: {rel_error}"


def test_decomp_ur_ultra_high_order_complex_charges():
    """Stress test: UR mode with 6-index tensor and complex charge structure."""
    group = U1Group()
    # Create a very high-order tensor (6 indices)
    indices = [
        Index(Direction.OUT, group, sectors=(
            Sector(-3, 1), Sector(-2, 2), Sector(-1, 3), Sector(0, 3), Sector(1, 2), Sector(2, 1), Sector(3, 1)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-2, 2), Sector(-1, 2), Sector(0, 4), Sector(1, 2), Sector(2, 2)
        )),
        Index(Direction.OUT, group, sectors=(
            Sector(-2, 1), Sector(-1, 2), Sector(0, 3), Sector(1, 2), Sector(2, 1)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-3, 1), Sector(-1, 1), Sector(0, 3), Sector(1, 1), Sector(3, 1)
        )),
        Index(Direction.OUT, group, sectors=(
            Sector(-2, 1), Sector(-1, 1), Sector(0, 2), Sector(1, 1), Sector(2, 1)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 1)
        ))
    ]
    
    T = Tensor.random(indices, itags=["p", "q", "r", "s", "t", "u"], seed=8100)
    
    # UR decomposition on axes [1, 3, 5] (specified axes go into U)
    U, R = decomp(T, axes=[1, 3, 5], mode="UR", flow="<<")
    
    # Verify structure
    assert len(U.indices) == 4  # q, s, u, bond_L
    assert len(R.indices) == 4  # bond_R, p, r, t
    assert 'q' in U.itags and 's' in U.itags and 'u' in U.itags
    assert 'p' in R.itags and 'r' in R.itags and 't' in R.itags
    
    # Check bond directions for flow="<<"
    assert U.indices[-1].direction == Direction.IN
    assert R.indices[0].direction == Direction.OUT
    
    # Verify bond has many charge sectors
    bond_charges_u = set(U.indices[-1].charges())
    bond_charges_r = set(R.indices[0].charges())
    assert bond_charges_u == bond_charges_r, "Bond charges should match"
    assert len(bond_charges_u) >= 4, f"Should have at least 4 charge sectors, got {len(bond_charges_u)}"
    
    # Verify that many blocks exist
    assert len(U.data) >= 8, f"U should have at least 8 blocks, got {len(U.data)}"
    assert len(R.data) >= 8, f"R should have at least 8 blocks, got {len(R.data)}"
    
    # Verify charge neutrality
    assert_charge_neutral(U)
    assert_charge_neutral(R)
    
    # Test with truncation
    U_trunc, R_trunc = decomp(T, axes=[1, 3, 5], mode="UR", trunc={"nkeep": 50})
    bond_dim_trunc = sum(sector.dim for sector in U_trunc.indices[-1].sectors)
    assert bond_dim_trunc <= 50, f"Truncated bond dimension should be <= 50, got {bond_dim_trunc}"
    
    # Reconstruct and verify accuracy
    reconstructed = contract(U, R)
    
    # Permute to match original order
    tag_to_pos_orig = {tag: i for i, tag in enumerate(T.itags)}
    tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
    perm = [tag_to_pos_recon[tag] for tag in T.itags]
    reconstructed.permute(perm)
    
    # Check reconstruction accuracy
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-11, f"Reconstruction error too large: {rel_error}"
    
    # Spot-check a few blocks
    block_count = 0
    for key in T.data.keys():
        if key in reconstructed.data and block_count < 5:
            assert torch.allclose(T.data[key], reconstructed.data[key], rtol=1e-10, atol=1e-12)
            block_count += 1


def test_decomp_lv_ultra_high_order_complex_charges():
    """Stress test: LV mode with 6-index tensor and complex charge structure."""
    group = U1Group()
    # Create a very high-order tensor (6 indices)
    indices = [
        Index(Direction.OUT, group, sectors=(
            Sector(-3, 1), Sector(-2, 2), Sector(-1, 2), Sector(0, 3), Sector(1, 3), Sector(2, 2), Sector(3, 1)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-3, 1), Sector(-2, 2), Sector(-1, 2), Sector(0, 4), Sector(1, 2), Sector(2, 2), Sector(3, 1)
        )),
        Index(Direction.OUT, group, sectors=(
            Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 1)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 1)
        )),
        Index(Direction.OUT, group, sectors=(
            Sector(-2, 1), Sector(0, 3), Sector(2, 1)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-3, 1), Sector(-1, 1), Sector(0, 2), Sector(1, 1), Sector(3, 1)
        ))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d", "e", "f"], seed=8200)
    
    # LV decomposition on axes [0, 2, 4] (specified go into L, complement into Vh)
    L, Vh = decomp(T, axes=[0, 2, 4], mode="LV", flow=">>")
    
    # Verify structure
    assert len(L.indices) == 4  # a, c, e, bond_L
    assert len(Vh.indices) == 4  # bond_R, b, d, f
    assert 'a' in L.itags and 'c' in L.itags and 'e' in L.itags
    assert 'b' in Vh.itags and 'd' in Vh.itags and 'f' in Vh.itags
    
    # Check bond directions for flow=">>"
    assert L.indices[-1].direction == Direction.OUT
    assert Vh.indices[0].direction == Direction.IN
    
    # Verify bond has many charge sectors
    bond_charges_l = set(L.indices[-1].charges())
    bond_charges_vh = set(Vh.indices[0].charges())
    assert bond_charges_l == bond_charges_vh, "Bond charges should match"
    assert len(bond_charges_l) >= 4, f"Should have at least 4 charge sectors, got {len(bond_charges_l)}"
    
    # Verify that many blocks exist
    assert len(L.data) >= 8, f"L should have at least 8 blocks, got {len(L.data)}"
    assert len(Vh.data) >= 8, f"Vh should have at least 8 blocks, got {len(Vh.data)}"
    
    # Verify charge neutrality
    assert_charge_neutral(L)
    assert_charge_neutral(Vh)
    
    # Test different flow
    L2, Vh2 = decomp(T, axes=[0, 2, 4], mode="LV", flow="<<")
    assert L2.indices[-1].direction == Direction.IN
    assert Vh2.indices[0].direction == Direction.OUT
    
    # Test with custom itag (single string)
    L3, Vh3 = decomp(T, axes=[0, 2, 4], mode="LV", itag="_custom_lv_bond_")
    assert "_custom_lv_bond_" in L3.itags
    assert "_custom_lv_bond_" in Vh3.itags
    
    # Reconstruct and verify accuracy
    reconstructed = contract(L, Vh)
    
    # Permute to match original order
    tag_to_pos_orig = {tag: i for i, tag in enumerate(T.itags)}
    tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
    perm = [tag_to_pos_recon[tag] for tag in T.itags]
    reconstructed.permute(perm)
    
    # Check reconstruction accuracy
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-11, f"Reconstruction error too large: {rel_error}"
    
    # Verify individual blocks match
    block_count = 0
    for key in T.data.keys():
        if key in reconstructed.data and block_count < 5:
            assert torch.allclose(T.data[key], reconstructed.data[key], rtol=1e-10, atol=1e-12)
            block_count += 1


# ===== High-Order QR Mode Stress Tests =====

def test_decomp_qr_high_order_multi_charge_stress():
    """Stress test: QR mode with high-order tensor and many charge sectors."""
    group = U1Group()
    # Create indices with many charge sectors
    indices = [
        Index(Direction.OUT, group, sectors=(
            Sector(-2, 2), Sector(-1, 3), Sector(0, 4), Sector(1, 3), Sector(2, 2)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-2, 1), Sector(-1, 2), Sector(0, 3), Sector(1, 2), Sector(2, 1)
        )),
        Index(Direction.OUT, group, sectors=(
            Sector(-1, 2), Sector(0, 3), Sector(1, 2)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-2, 1), Sector(-1, 1), Sector(0, 2), Sector(1, 1), Sector(2, 1)
        )),
        Index(Direction.OUT, group, sectors=(
            Sector(0, 2), Sector(1, 2), Sector(2, 1)
        ))
    ]
    
    T = Tensor.random(indices, itags=["a", "b", "c", "d", "e"], seed=9000)
    
    # QR decomposition on axes [0, 2, 4] (3 axes into Q)
    Q, R = decomp(T, axes=[0, 2, 4], mode="QR")
    
    # Verify structure
    assert len(Q.indices) == 4  # a, c, e, bond
    assert len(R.indices) == 3  # bond, b, d
    assert 'a' in Q.itags and 'c' in Q.itags and 'e' in Q.itags
    assert 'b' in R.itags and 'd' in R.itags
    
    # Check that we have multiple charge sectors in the bond
    bond_charges = set(Q.indices[-1].charges())
    assert len(bond_charges) >= 3, f"Should have at least 3 charge sectors, got {len(bond_charges)}"
    
    # Verify that blocks exist for different charges
    assert len(Q.data) >= 5, f"Q should have at least 5 blocks, got {len(Q.data)}"
    assert len(R.data) >= 5, f"R should have at least 5 blocks, got {len(R.data)}"
    
    # Verify charge neutrality
    assert_charge_neutral(Q)
    assert_charge_neutral(R)
    
    # Reconstruct and verify accuracy
    reconstructed = contract(Q, R)
    
    # Permute to match original order
    tag_to_pos_orig = {tag: i for i, tag in enumerate(T.itags)}
    tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
    perm = [tag_to_pos_recon[tag] for tag in T.itags]
    reconstructed.permute(perm)
    
    # Check reconstruction accuracy
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-11, f"Reconstruction error too large: {rel_error}"
    
    # Verify individual blocks match
    for key in T.data.keys():
        if key in reconstructed.data:
            assert torch.allclose(T.data[key], reconstructed.data[key], rtol=1e-10, atol=1e-12)


def test_decomp_qr_ultra_high_order_complex_charges():
    """Stress test: QR mode with 6-index tensor and complex charge structure."""
    group = U1Group()
    # Create a very high-order tensor (6 indices)
    indices = [
        Index(Direction.OUT, group, sectors=(
            Sector(-3, 1), Sector(-2, 2), Sector(-1, 2), Sector(0, 3), Sector(1, 2), Sector(2, 2), Sector(3, 1)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-2, 2), Sector(-1, 3), Sector(0, 4), Sector(1, 3), Sector(2, 2)
        )),
        Index(Direction.OUT, group, sectors=(
            Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 1)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-3, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(3, 1)
        )),
        Index(Direction.OUT, group, sectors=(
            Sector(-2, 1), Sector(0, 3), Sector(2, 1)
        )),
        Index(Direction.IN, group, sectors=(
            Sector(-2, 1), Sector(-1, 1), Sector(0, 2), Sector(1, 1), Sector(2, 1)
        ))
    ]
    
    T = Tensor.random(indices, itags=["w", "x", "y", "z", "u", "v"], seed=9999)
    
    # QR decomposition with mixed axes [1, 3, 5] into R, rest into Q
    Q, R = decomp(T, axes=[0, 2, 4], mode="QR", flow="<<")
    
    # Verify structure
    assert len(Q.indices) == 4  # w, y, u, bond
    assert len(R.indices) == 4  # bond, x, z, v
    assert 'w' in Q.itags and 'y' in Q.itags and 'u' in Q.itags
    assert 'x' in R.itags and 'z' in R.itags and 'v' in R.itags
    
    # Check bond directions for flow="<<" (bonds point inward)
    assert Q.indices[-1].direction == Direction.IN
    assert R.indices[0].direction == Direction.OUT
    
    # Verify bond has many charge sectors
    bond_charges_q = set(Q.indices[-1].charges())
    bond_charges_r = set(R.indices[0].charges())
    assert bond_charges_q == bond_charges_r, "Bond charges should match"
    assert len(bond_charges_q) >= 4, f"Should have at least 4 charge sectors, got {len(bond_charges_q)}"
    
    # Verify that many blocks exist
    assert len(Q.data) >= 8, f"Q should have at least 8 blocks, got {len(Q.data)}"
    assert len(R.data) >= 8, f"R should have at least 8 blocks, got {len(R.data)}"
    
    # Verify charge neutrality
    assert_charge_neutral(Q)
    assert_charge_neutral(R)
    
    # Test reconstruction with explicit custom itag
    Q2, R2 = decomp(T, axes=[0, 2, 4], mode="QR", itag="_custom_qr_")
    assert "_custom_qr_" in Q2.itags
    assert "_custom_qr_" in R2.itags
    
    # Reconstruct and verify accuracy
    reconstructed = contract(Q, R)
    
    # Permute to match original order
    tag_to_pos_orig = {tag: i for i, tag in enumerate(T.itags)}
    tag_to_pos_recon = {tag: i for i, tag in enumerate(reconstructed.itags)}
    perm = [tag_to_pos_recon[tag] for tag in T.itags]
    reconstructed.permute(perm)
    
    # Check reconstruction accuracy
    rel_error = (T - reconstructed).norm() / T.norm()
    assert rel_error < 1e-11, f"Reconstruction error too large: {rel_error}"
    
    # Spot-check a few blocks
    block_count = 0
    for key in T.data.keys():
        if key in reconstructed.data and block_count < 5:
            assert torch.allclose(T.data[key], reconstructed.data[key], rtol=1e-10, atol=1e-12)
            block_count += 1
