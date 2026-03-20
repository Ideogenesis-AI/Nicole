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


"""Tests for physical space and operator construction."""

import math
import torch
import pytest

from nicole import load_space
from nicole import Direction, Index, Tensor
from nicole import U1Group, Z2Group, SU2Group,ProductGroup


class TestLoadSpaceBasic:
    """Test basic functionality of load_space."""
    
    def test_spin_half(self):
        """Test spin-1/2 space creation."""
        Spc, Op = load_space("Spin", "U1", {"J": 0.5})
        
        # Check space properties
        assert Spc.direction == Direction.IN
        assert isinstance(Spc.group, U1Group)
        assert Spc.dim == 2
        assert len(Spc.sectors) == 2
        
        # Check sectors: m_z = -1/2, +1/2 → charges = -1, +1
        charges = [s.charge for s in Spc.sectors]
        assert charges == [-1, 1]
        for sector in Spc.sectors:
            assert sector.dim == 1
        
        # Check operators exist (including vacuum index)
        assert set(Op.keys()) == {"Sz", "Sp", "Sm", "vac"}
    
    def test_spin_zero(self):
        """Test spin-0 space (single state)."""
        Spc, Op = load_space("Spin", "U1", {"J": 0})
        
        assert Spc.dim == 1
        assert len(Spc.sectors) == 1
        assert Spc.sectors[0].charge == 0
        assert Spc.sectors[0].dim == 1
        
        # Sz should have single diagonal block
        assert len(Op["Sz"].data) == 1
        assert (0, 0) in Op["Sz"].data
        assert Op["Sz"].data[(0, 0)][0, 0] == 0.0
        
        # Sp and Sm should be empty (no transitions)
        assert len(Op["Sp"].data) == 0
        assert len(Op["Sm"].data) == 0
    
    def test_spin_one(self):
        """Test spin-1 space creation."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.0})
        
        assert Spc.dim == 3
        assert len(Spc.sectors) == 3
        
        # Check sectors: m_z = -1, 0, +1 → charges = -2, 0, +2
        charges = [s.charge for s in Spc.sectors]
        assert charges == [-2, 0, 2]
    
    def test_spin_three_halves(self):
        """Test spin-3/2 space creation."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.5})
        
        assert Spc.dim == 4
        assert len(Spc.sectors) == 4
        
        # Check sectors: m_z = -3/2, -1/2, +1/2, +3/2 → charges = -3, -1, +1, +3
        charges = [s.charge for s in Spc.sectors]
        assert charges == [-3, -1, 1, 3]
    
    def test_spin_two(self):
        """Test spin-2 space creation."""
        Spc, Op = load_space("Spin", "U1", {"J": 2})
        
        assert Spc.dim == 5
        assert len(Spc.sectors) == 5
        
        # Check sectors: m_z = -2, -1, 0, +1, +2 → charges = -4, -2, 0, +2, +4
        charges = [s.charge for s in Spc.sectors]
        assert charges == [-4, -2, 0, 2, 4]
    
    def test_integer_and_float_j(self):
        """Test that both integer and float J values work."""
        Spc_int, Op_int = load_space("Spin", "U1", {"J": 1})
        Spc_float, Op_float = load_space("Spin", "U1", {"J": 1.0})
        
        # Should produce identical results
        assert Spc_int.dim == Spc_float.dim
        assert len(Spc_int.sectors) == len(Spc_float.sectors)
        assert len(Op_int["Sz"].data) == len(Op_float["Sz"].data)


class TestOperatorStructure:
    """Test operator tensor structure."""
    
    def test_sz_structure(self):
        """Test Sz operator structure."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.0})
        Sz = Op["Sz"]
        
        # Should be 2-index tensor
        assert len(Sz.indices) == 2
        assert Sz.itags == ("_init_", "_init_")
        
        # Directions: (IN, OUT)
        assert Sz.indices[0].direction == Direction.IN
        assert Sz.indices[1].direction == Direction.OUT
        
        # Should have diagonal blocks only
        assert len(Sz.data) == 3  # 3 sectors for J=1
        for (q_out, q_in) in Sz.data.keys():
            assert q_out == q_in  # Diagonal
    
    def test_sp_structure(self):
        """Test Sp operator structure."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.0})
        Sp = Op["Sp"]
        
        # Should be 3-index tensor
        assert len(Sp.indices) == 3
        assert Sp.itags == ("_init_", "_init_", "_aux_")
        
        # Directions: (IN, OUT, OUT)
        assert Sp.indices[0].direction == Direction.IN
        assert Sp.indices[1].direction == Direction.OUT
        assert Sp.indices[2].direction == Direction.OUT
        
        # Auxiliary index should have single sector with charge +2
        aux_idx = Sp.indices[2]
        assert len(aux_idx.sectors) == 1
        assert aux_idx.sectors[0].charge == 2
        assert aux_idx.sectors[0].dim == 1
        
        # Should have 2 blocks for J=1 (excluding highest m_z)
        assert len(Sp.data) == 2
    
    def test_sm_structure(self):
        """Test Sm operator structure."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.0})
        Sm = Op["Sm"]
        
        # Should be 3-index tensor
        assert len(Sm.indices) == 3
        assert Sm.itags == ("_init_", "_init_", "_aux_")
        
        # Directions: (IN, OUT, OUT)
        assert Sm.indices[0].direction == Direction.IN
        assert Sm.indices[1].direction == Direction.OUT
        assert Sm.indices[2].direction == Direction.OUT
        
        # Auxiliary index should have single sector with charge -2
        aux_idx = Sm.indices[2]
        assert len(aux_idx.sectors) == 1
        assert aux_idx.sectors[0].charge == -2
        assert aux_idx.sectors[0].dim == 1
        
        # Should have 2 blocks for J=1 (excluding lowest m_z)
        assert len(Sm.data) == 2


class TestChargeConservation:
    """Test charge conservation in operators."""
    
    def test_sz_charge_conservation(self):
        """Test Sz preserves charge."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.5})
        Sz = Op["Sz"]
        
        # Charge conservation: -q_out + q_in = 0
        for (q_out, q_in), block in Sz.data.items():
            assert -q_out + q_in == 0
            assert q_out == q_in
    
    def test_sp_charge_conservation(self):
        """Test Sp charge conservation."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.5})
        Sp = Op["Sp"]
        
        # Charge conservation: -q_out + q_in + q_aux = 0
        for (q_out, q_in, q_aux), block in Sp.data.items():
            assert -q_out + q_in + q_aux == 0
            # Sp should raise: q_out = q_in + 2
            assert q_out == q_in + 2
            assert q_aux == 2
    
    def test_sm_charge_conservation(self):
        """Test Sm charge conservation."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.5})
        Sm = Op["Sm"]
        
        # Charge conservation: -q_out + q_in + q_aux = 0
        for (q_out, q_in, q_aux), block in Sm.data.items():
            assert -q_out + q_in + q_aux == 0
            # Sm should lower: q_out = q_in - 2
            assert q_out == q_in - 2
            assert q_aux == -2


class TestMatrixElements:
    """Test operator matrix elements."""
    
    def test_sz_eigenvalues(self):
        """Test Sz has correct eigenvalues (m_z values)."""
        for J in [0, 0.5, 1, 1.5, 2]:
            Spc, Op = load_space("Spin", "U1", {"J": J})
            Sz = Op["Sz"]
            
            # Collect all m_z values from Sz blocks
            m_z_values = []
            for (q_out, q_in), block in Sz.data.items():
                m_z = block[0, 0]
                m_z_values.append(m_z)
            
            # Should range from -J to +J in steps of 1.0
            expected = torch.arange(-J, J + 0.5, 1.0).tolist()
            # Convert both to sorted lists for comparison
            assert sorted([float(x) for x in m_z_values]) == sorted(expected)
    
    def test_sp_matrix_elements_spin_half(self):
        """Test Sp matrix elements for spin-1/2."""
        Spc, Op = load_space("Spin", "U1", {"J": 0.5})
        Sp = Op["Sp"]
        J = 0.5
        
        # Should have single block: |-1/2⟩ → |+1/2⟩
        assert len(Sp.data) == 1
        key = (1, -1, 2)  # (q_out, q_in, q_aux)
        assert key in Sp.data
        
        m_z_in = -0.5
        # Expected: -sqrt(J(J+1) - m_z(m_z+1)) / sqrt(2)
        expected = -math.sqrt(J * (J + 1) - m_z_in * (m_z_in + 1)) / math.sqrt(2.0)
        assert torch.isclose(Sp.data[key][0, 0, 0], torch.tensor(expected, dtype=Sp.dtype), atol=1e-9)
        assert torch.isclose(Sp.data[key][0, 0, 0], torch.tensor(-1/math.sqrt(2), dtype=Sp.dtype), atol=1e-9)
    
    def test_sm_matrix_elements_spin_half(self):
        """Test Sm matrix elements for spin-1/2."""
        Spc, Op = load_space("Spin", "U1", {"J": 0.5})
        Sm = Op["Sm"]
        J = 0.5
        
        # Should have single block: |+1/2⟩ → |-1/2⟩
        assert len(Sm.data) == 1
        key = (-1, 1, -2)  # (q_out, q_in, q_aux)
        assert key in Sm.data
        
        m_z_in = 0.5
        # Expected: sqrt(J(J+1) - m_z(m_z-1)) / sqrt(2)
        expected = math.sqrt(J * (J + 1) - m_z_in * (m_z_in - 1)) / math.sqrt(2.0)
        assert torch.isclose(Sm.data[key][0, 0, 0], torch.tensor(expected, dtype=Sm.dtype), atol=1e-9)
        assert torch.isclose(Sm.data[key][0, 0, 0], torch.tensor(1/math.sqrt(2), dtype=Sm.dtype), atol=1e-9)
    
    def test_sp_matrix_elements_spin_one(self):
        """Test Sp matrix elements for spin-1."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.0})
        Sp = Op["Sp"]
        J = 1.0
        
        # Should have 2 blocks
        assert len(Sp.data) == 2
        
        # |-1⟩ → |0⟩
        key1 = (0, -2, 2)
        m_z_in = -1.0
        expected1 = -math.sqrt(J * (J + 1) - m_z_in * (m_z_in + 1)) / math.sqrt(2.0)
        assert torch.isclose(Sp.data[key1][0, 0, 0], torch.tensor(expected1, dtype=Sp.dtype), atol=1e-9)
        assert torch.isclose(Sp.data[key1][0, 0, 0], torch.tensor(-1.0, dtype=Sp.dtype), atol=1e-9)
        
        # |0⟩ → |+1⟩
        key2 = (2, 0, 2)
        m_z_in = 0.0
        expected2 = -math.sqrt(J * (J + 1) - m_z_in * (m_z_in + 1)) / math.sqrt(2.0)
        assert torch.isclose(Sp.data[key2][0, 0, 0], torch.tensor(expected2, dtype=Sp.dtype), atol=1e-9)
        assert torch.isclose(Sp.data[key2][0, 0, 0], torch.tensor(-1.0, dtype=Sp.dtype), atol=1e-9)
    
    def test_sm_matrix_elements_spin_one(self):
        """Test Sm matrix elements for spin-1."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.0})
        Sm = Op["Sm"]
        J = 1.0
        
        # Should have 2 blocks
        assert len(Sm.data) == 2
        
        # |0⟩ → |-1⟩
        key1 = (-2, 0, -2)
        m_z_in = 0.0
        expected1 = math.sqrt(J * (J + 1) - m_z_in * (m_z_in - 1)) / math.sqrt(2.0)
        assert torch.isclose(Sm.data[key1][0, 0, 0], torch.tensor(expected1, dtype=Sm.dtype), atol=1e-9)
        assert torch.isclose(Sm.data[key1][0, 0, 0], torch.tensor(1.0, dtype=Sm.dtype), atol=1e-9)
        
        # |+1⟩ → |0⟩
        key2 = (0, 2, -2)
        m_z_in = 1.0
        expected2 = math.sqrt(J * (J + 1) - m_z_in * (m_z_in - 1)) / math.sqrt(2.0)
        assert torch.isclose(Sm.data[key2][0, 0, 0], torch.tensor(expected2, dtype=Sm.dtype), atol=1e-9)
        assert torch.isclose(Sm.data[key2][0, 0, 0], torch.tensor(1.0, dtype=Sm.dtype), atol=1e-9)
    
    def test_sp_has_minus_sign(self):
        """Test Sp has minus sign (spherical tensor convention)."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.0})
        
        # All Sp matrix elements should be negative
        for block in Op["Sp"].data.values():
            assert block[0, 0, 0] < 0
        
        # All Sm matrix elements should be positive
        for block in Op["Sm"].data.values():
            assert block[0, 0, 0] > 0


class TestErrorHandling:
    """Test error handling."""
    
    def test_unsupported_preset(self):
        """Test error for unsupported preset."""
        with pytest.raises(ValueError, match="Unsupported system preset"):
            load_space("Boson", "U1", {})
    
    def test_unsupported_symmetry(self):
        """Test error for unsupported symmetry."""
        with pytest.raises(ValueError, match="Unsupported symmetry"):
            load_space("Spin", "Z2", {"J": 0.5})
    
    def test_missing_j_option(self):
        """Test error when J option is missing."""
        with pytest.raises(ValueError, match="Option 'J' .* is required"):
            load_space("Spin", "U1", {})
        
        with pytest.raises(ValueError, match="Option 'J' .* is required"):
            load_space("Spin", "U1", None)
    
    def test_invalid_j_type(self):
        """Test error for invalid J type."""
        with pytest.raises(ValueError, match="J must be a number"):
            load_space("Spin", "U1", {"J": "invalid"})
    
    def test_negative_j(self):
        """Test error for negative J."""
        with pytest.raises(ValueError, match="J must be non-negative"):
            load_space("Spin", "U1", {"J": -1})
    
    def test_non_half_integer_j(self):
        """Test error for non-half-integer J."""
        with pytest.raises(ValueError, match="J must be a half-integer"):
            load_space("Spin", "U1", {"J": 0.3})
        
        with pytest.raises(ValueError, match="J must be a half-integer"):
            load_space("Spin", "U1", {"J": 1.7})


class TestOperatorRelations:
    """Test physical relations between operators."""
    
    def test_sp_boundary(self):
        """Test Sp acting on highest m_z gives zero."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.0})
        Sp = Op["Sp"]
        
        # Highest m_z is +1 (charge +2)
        # Sp should not have blocks with q_in = 2
        q_in_values = [key[1] for key in Sp.data.keys()]
        assert 2 not in q_in_values or len([k for k in Sp.data.keys() if k[1] == 2]) == 0
        
        # Actually, for J=1, highest m_z should not appear as input
        highest_charge = 2
        for (q_out, q_in, q_aux) in Sp.data.keys():
            assert q_in < highest_charge
    
    def test_sm_boundary(self):
        """Test Sm acting on lowest m_z gives zero."""
        Spc, Op = load_space("Spin", "U1", {"J": 1.0})
        Sm = Op["Sm"]
        
        # Lowest m_z is -1 (charge -2)
        # Sm should not have blocks with q_in = -2
        lowest_charge = -2
        for (q_out, q_in, q_aux) in Sm.data.keys():
            assert q_in > lowest_charge
    
    def test_sz_traceless_for_j_gt_half(self):
        """Test Sz is traceless for J > 1/2."""
        for J in [1, 1.5, 2]:
            Spc, Op = load_space("Spin", "U1", {"J": J})
            Sz = Op["Sz"]
            
            # Sum of diagonal elements should be 0
            trace = sum(block[0, 0] for block in Sz.data.values())
            assert math.isclose(trace.item(), 0.0)
    
    def test_dimensionality_formula(self):
        """Test space dimension equals 2J+1."""
        for J in [0, 0.5, 1, 1.5, 2, 2.5, 3]:
            Spc, Op = load_space("Spin", "U1", {"J": J})
            expected_dim = int(2 * J + 1)
            assert Spc.dim == expected_dim
            assert len(Spc.sectors) == expected_dim


class TestFermionBasic:
    """Test basic functionality of spinless fermion space."""
    
    def test_ferm_u1_space(self):
        """Test spinless fermion space creation with U(1) symmetry."""
        Spc, Op = load_space("Ferm", "U1")
        
        # Check space properties
        assert Spc.direction == Direction.IN
        assert isinstance(Spc.group, U1Group)
        assert Spc.dim == 2
        assert len(Spc.sectors) == 2
        
        # Check sectors: |0⟩ (charge -1) and |1⟩ (charge 1) - half-filling at 0
        charges = [s.charge for s in Spc.sectors]
        assert charges == [-1, 1]
        for sector in Spc.sectors:
            assert sector.dim == 1
        
        # Check operators exist
        assert set(Op.keys()) == {"F", "Z", "vac"}
    
    def test_ferm_z2_space(self):
        """Test spinless fermion space creation with Z2 symmetry."""
        Spc, Op = load_space("Ferm", "Z2")
        
        # Check space properties
        assert Spc.direction == Direction.IN
        assert isinstance(Spc.group, Z2Group)
        assert Spc.dim == 2
        assert len(Spc.sectors) == 2
        
        # Check sectors: |0⟩ (parity 0) and |1⟩ (parity 1)
        parities = [s.charge for s in Spc.sectors]
        assert parities == [0, 1]
        for sector in Spc.sectors:
            assert sector.dim == 1
        
        # Check operators exist
        assert set(Op.keys()) == {"F", "Z", "vac"}
    
    def test_ferm_no_options_required(self):
        """Test that no options are required for spinless fermions."""
        Spc1, Op1 = load_space("Ferm", "U1")
        Spc2, Op2 = load_space("Ferm", "U1", None)
        Spc3, Op3 = load_space("Ferm", "U1", {})
        
        # All should produce identical results
        assert Spc1.dim == Spc2.dim == Spc3.dim
        assert len(Op1["F"].data) == len(Op2["F"].data) == len(Op3["F"].data)


class TestFermionOperatorStructure:
    """Test fermionic operator structure."""
    
    def test_f_structure_u1(self):
        """Test F (annihilation) operator structure with U(1)."""
        Spc, Op = load_space("Ferm", "U1")
        F = Op["F"]
        
        # Should be 3-index tensor
        assert len(F.indices) == 3
        assert F.itags == ("_init_", "_init_", "_aux_")
        
        # Directions: (IN, OUT, OUT)
        assert F.indices[0].direction == Direction.IN
        assert F.indices[1].direction == Direction.OUT
        assert F.indices[2].direction == Direction.OUT
        
        # Auxiliary index should have single sector with charge -2
        aux_idx = F.indices[2]
        assert len(aux_idx.sectors) == 1
        assert aux_idx.sectors[0].charge == -2
        assert aux_idx.sectors[0].dim == 1
        
        # Should have single block: |1⟩ → |0⟩
        assert len(F.data) == 1
        assert (-1, 1, -2) in F.data
    
    def test_f_structure_z2(self):
        """Test F (annihilation) operator structure with Z2."""
        Spc, Op = load_space("Ferm", "Z2")
        F = Op["F"]
        
        # Should be 3-index tensor
        assert len(F.indices) == 3
        assert F.itags == ("_init_", "_init_", "_aux_")
        
        # Auxiliary index should have single sector with parity 1
        aux_idx = F.indices[2]
        assert len(aux_idx.sectors) == 1
        assert aux_idx.sectors[0].charge == 1
        assert aux_idx.sectors[0].dim == 1
        
        # Should have single block: |1⟩ → |0⟩
        assert len(F.data) == 1
        assert (0, 1, 1) in F.data
    
    def test_z_structure_u1(self):
        """Test Z operator structure with U(1)."""
        Spc, Op = load_space("Ferm", "U1")
        Z = Op["Z"]
        
        # Should be 2-index tensor
        assert len(Z.indices) == 2
        assert Z.itags == ("_init_", "_init_")
        
        # Directions: (IN, OUT)
        assert Z.indices[0].direction == Direction.IN
        assert Z.indices[1].direction == Direction.OUT
        
        # Should have diagonal blocks only
        assert len(Z.data) == 2
        for (q_out, q_in) in Z.data.keys():
            assert q_out == q_in  # Diagonal
    
    def test_z_structure_z2(self):
        """Test Z operator structure with Z2."""
        Spc, Op = load_space("Ferm", "Z2")
        Z = Op["Z"]
        
        # Should be 2-index tensor
        assert len(Z.indices) == 2
        
        # Should have diagonal blocks only
        assert len(Z.data) == 2
        for (p_out, p_in) in Z.data.keys():
            assert p_out == p_in  # Diagonal


class TestFermionChargeConservation:
    """Test charge conservation in fermionic operators."""
    
    def test_f_charge_conservation_u1(self):
        """Test F charge conservation with U(1)."""
        Spc, Op = load_space("Ferm", "U1")
        F = Op["F"]
        
        # Charge conservation with directions (IN, OUT, OUT): (+1)*q₀ + (-1)*q₁ + (-1)*q₂ = 0
        for (q_out, q_in, q_aux), block in F.data.items():
            assert (+1)*q_out + (-1)*q_in + (-1)*q_aux == 0
            # F should annihilate: q_out = q_in - 2 (from charge 1 to -1)
            assert q_out == q_in - 2
            assert q_aux == -2
    
    def test_f_charge_conservation_z2(self):
        """Test F charge conservation with Z2."""
        Spc, Op = load_space("Ferm", "Z2")
        F = Op["F"]
        
        # Charge conservation (mod 2): -q_out + q_in + q_aux = 0 (mod 2)
        for (q_out, q_in, q_aux), block in F.data.items():
            assert (-q_out + q_in + q_aux) % 2 == 0
            # F should annihilate: even → odd (0 → 1)
            assert q_in == 1 and q_out == 0
            assert q_aux == 1
    
    def test_z_charge_conservation_u1(self):
        """Test Z preserves charge with U(1)."""
        Spc, Op = load_space("Ferm", "U1")
        Z = Op["Z"]
        
        # Charge conservation with directions (IN, OUT): (+1)*q₀ + (-1)*q₁ = 0
        for (q_out, q_in), block in Z.data.items():
            assert (+1)*q_out + (-1)*q_in == 0
            assert q_out == q_in
    
    def test_z_charge_conservation_z2(self):
        """Test Z preserves parity with Z2."""
        Spc, Op = load_space("Ferm", "Z2")
        Z = Op["Z"]
        
        # Parity conservation: -p_out + p_in = 0 (mod 2)
        for (p_out, p_in), block in Z.data.items():
            assert (-p_out + p_in) % 2 == 0
            assert p_out == p_in


class TestFermionMatrixElements:
    """Test fermionic operator matrix elements."""
    
    def test_f_matrix_elements_u1(self):
        """Test F matrix elements with U(1)."""
        Spc, Op = load_space("Ferm", "U1")
        F = Op["F"]
        
        # F|1⟩ = |0⟩ → ⟨0|F|1⟩ = 1
        key = (-1, 1, -2)
        assert key in F.data
        assert math.isclose(F.data[key][0, 0, 0].item(), 1.0)
    
    def test_f_matrix_elements_z2(self):
        """Test F matrix elements with Z2."""
        Spc, Op = load_space("Ferm", "Z2")
        F = Op["F"]
        
        # F|1⟩ = |0⟩ → ⟨0|F|1⟩ = 1
        key = (0, 1, 1)
        assert key in F.data
        assert math.isclose(F.data[key][0, 0, 0].item(), 1.0)
    
    def test_z_matrix_elements_u1(self):
        """Test Z matrix elements with U(1)."""
        Spc, Op = load_space("Ferm", "U1")
        Z = Op["Z"]
        
        # Z|0⟩ = |0⟩ → ⟨0|Z|0⟩ = 1 (charge -1)
        assert (-1, -1) in Z.data
        assert math.isclose(Z.data[(-1, -1)][0, 0].item(), 1.0)
        
        # Z|1⟩ = -|1⟩ → ⟨1|Z|1⟩ = -1 (charge 1)
        assert (1, 1) in Z.data
        assert math.isclose(Z.data[(1, 1)][0, 0].item(), -1.0)
    
    def test_z_matrix_elements_z2(self):
        """Test Z matrix elements with Z2."""
        Spc, Op = load_space("Ferm", "Z2")
        Z = Op["Z"]
        
        # Z|0⟩ = |0⟩ → ⟨0|Z|0⟩ = 1
        assert (0, 0) in Z.data
        assert math.isclose(Z.data[(0, 0)][0, 0].item(), 1.0)
        
        # Z|1⟩ = -|1⟩ → ⟨1|Z|1⟩ = -1
        assert (1, 1) in Z.data
        assert math.isclose(Z.data[(1, 1)][0, 0].item(), -1.0)
    
    def test_z_eigenvalues(self):
        """Test Z has eigenvalues +1 and -1."""
        for preserv in ["U1", "Z2"]:
            Spc, Op = load_space("Ferm", preserv)
            Z = Op["Z"]
            
            eigenvalues = [float(Z.data[key][0, 0]) for key in sorted(Z.data.keys())]
            expected = [1.0, -1.0]
            assert all(abs(a - b) < 1e-7 for a, b in zip(eigenvalues, expected))


class TestFermionErrorHandling:
    """Test error handling for fermionic systems."""
    
    def test_unsupported_symmetry(self):
        """Test error for unsupported symmetry."""
        with pytest.raises(ValueError, match="Unsupported symmetry"):
            load_space("Ferm", "SU2")
    
    def test_ferm_z2_and_u1_are_different(self):
        """Test that Z2 and U1 produce different auxiliary charges."""
        Spc_u1, Op_u1 = load_space("Ferm", "U1")
        Spc_z2, Op_z2 = load_space("Ferm", "Z2")
        
        # Both have same dimension
        assert Spc_u1.dim == Spc_z2.dim == 2
        
        # But auxiliary indices have different charges
        aux_u1 = Op_u1["F"].indices[2]
        aux_z2 = Op_z2["F"].indices[2]
        
        assert aux_u1.sectors[0].charge == -2  # U1
        assert aux_z2.sectors[0].charge == 1   # Z2


class TestFermionVacuumIndex:
    """Test vacuum index for fermionic systems."""
    
    def test_vacuum_index_u1(self):
        """Test vacuum index structure with U(1)."""
        Spc, Op = load_space("Ferm", "U1")
        vac = Op["vac"]
        
        # Should be an Index, not a Tensor
        assert isinstance(vac, Index)
        assert vac.direction == Direction.IN
        assert isinstance(vac.group, U1Group)
        
        # Should have single sector with charge 0
        assert len(vac.sectors) == 1
        assert vac.sectors[0].charge == 0
        assert vac.sectors[0].dim == 1
    
    def test_vacuum_index_z2(self):
        """Test vacuum index structure with Z2."""
        Spc, Op = load_space("Ferm", "Z2")
        vac = Op["vac"]
        
        # Should be an Index
        assert isinstance(vac, Index)
        assert vac.direction == Direction.IN
        assert isinstance(vac.group, Z2Group)
        
        # Should have single sector with parity 0
        assert len(vac.sectors) == 1
        assert vac.sectors[0].charge == 0
        assert vac.sectors[0].dim == 1


class TestBandBasic:
    """Test basic functionality of spinful fermion (Band) space."""
    
    def test_band_u1u1_space(self):
        """Test Band space creation with U(1)xU(1) symmetry."""
        Spc, Op = load_space("Band", "U1, U1")
        
        # Check space properties
        assert Spc.direction == Direction.IN
        assert isinstance(Spc.group, ProductGroup)
        assert Spc.dim == 4
        assert len(Spc.sectors) == 4
        
        # Check sectors: |0⟩, |↓⟩, |↑⟩, |↑↓⟩ - half-filling at charge 0
        charges = [s.charge for s in Spc.sectors]
        assert charges == [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for sector in Spc.sectors:
            assert sector.dim == 1
        
        # Check operators exist
        assert set(Op.keys()) == {"F_up", "F_dn", "Z", "Sz", "Sp", "Sm", "vac"}
    
    def test_band_z2u1_space(self):
        """Test Band space creation with Z2xU(1) symmetry."""
        Spc, Op = load_space("Band", "Z2, U1")
        
        # Check space properties
        assert Spc.direction == Direction.IN
        assert isinstance(Spc.group, ProductGroup)
        assert Spc.dim == 4
        assert len(Spc.sectors) == 3  # (0,0) contains 2 states
        
        # Check sectors
        charges = [s.charge for s in Spc.sectors]
        dims = [s.dim for s in Spc.sectors]
        assert charges == [(0, 0), (1, -1), (1, 1)]
        assert dims == [2, 1, 1]
        
        # Check operators exist
        assert set(Op.keys()) == {"F_up", "F_dn", "Z", "Sz", "Sp", "Sm", "vac"}
    
    def test_band_no_options_required(self):
        """Test that no options are required for Band systems."""
        Spc1, Op1 = load_space("Band", "U1, U1")
        Spc2, Op2 = load_space("Band", "U1, U1", None)
        Spc3, Op3 = load_space("Band", "U1, U1", {})
        
        # All should produce identical results
        assert Spc1.dim == Spc2.dim == Spc3.dim
        assert len(Op1["F_up"].data) == len(Op2["F_up"].data) == len(Op3["F_up"].data)
    
    def test_band_preserv_with_and_without_spaces(self):
        """Test that preserv works with and without spaces."""
        # U1,U1 variants
        Spc1, Op1 = load_space("Band", "U1,U1")
        Spc2, Op2 = load_space("Band", "U1, U1")
        assert Spc1.dim == Spc2.dim
        assert len(Op1["F_up"].data) == len(Op2["F_up"].data)
        
        # Z2,U1 variants
        Spc3, Op3 = load_space("Band", "Z2,U1")
        Spc4, Op4 = load_space("Band", "Z2, U1")
        assert Spc3.dim == Spc4.dim
        assert len(Op3["F_up"].data) == len(Op4["F_up"].data)


class TestBandOperatorStructure:
    """Test Band operator structure."""
    
    def test_f_up_structure_u1u1(self):
        """Test F_up operator structure with U(1)xU(1)."""
        Spc, Op = load_space("Band", "U1, U1")
        F_up = Op["F_up"]
        
        # Should be 3-index tensor
        assert len(F_up.indices) == 3
        assert F_up.itags == ("_init_", "_init_", "_aux_")
        
        # Auxiliary index should have charge (-1, -1)
        aux_idx = F_up.indices[2]
        assert len(aux_idx.sectors) == 1
        assert aux_idx.sectors[0].charge == (-1, -1)
        
        # Should have 2 blocks: |↑⟩ → |0⟩ and |↑↓⟩ → |↓⟩
        assert len(F_up.data) == 2
    
    def test_f_dn_structure_u1u1(self):
        """Test F_dn operator structure with U(1)xU(1)."""
        Spc, Op = load_space("Band", "U1, U1")
        F_dn = Op["F_dn"]
        
        # Auxiliary index should have charge (-1, 1)
        aux_idx = F_dn.indices[2]
        assert len(aux_idx.sectors) == 1
        assert aux_idx.sectors[0].charge == (-1, 1)
        
        # Should have 2 blocks: |↓⟩ → |0⟩ and |↑↓⟩ → |↑⟩
        assert len(F_dn.data) == 2
    
    def test_spin_operators_structure(self):
        """Test spin operator structure."""
        Spc, Op = load_space("Band", "U1, U1")
        
        # Sz should be diagonal with zero sectors trimmed
        Sz = Op["Sz"]
        assert len(Sz.indices) == 2
        assert len(Sz.data) == 2  # Only non-zero eigenvalues (|↑⟩ and |↓⟩)
        
        # Sp should have 1 block (|↓⟩ → |↑⟩)
        Sp = Op["Sp"]
        assert len(Sp.indices) == 3
        assert len(Sp.data) == 1
        
        # Sm should have 1 block (|↑⟩ → |↓⟩)
        Sm = Op["Sm"]
        assert len(Sm.indices) == 3
        assert len(Sm.data) == 1


class TestBandChargeConservation:
    """Test charge conservation in Band operators."""
    
    def test_f_up_charge_conservation_u1u1(self):
        """Test F_up charge conservation with U(1)xU(1)."""
        Spc, Op = load_space("Band", "U1, U1")
        F_up = Op["F_up"]
        
        for (q_out, q_in, q_aux), block in F_up.data.items():
            # Charge conservation with directions (IN, OUT, OUT): (+1)*q₀ + (-1)*q₁ + (-1)*q₂ = 0
            assert (+1)*q_out[0] + (-1)*q_in[0] + (-1)*q_aux[0] == 0  # Particle number
            assert (+1)*q_out[1] + (-1)*q_in[1] + (-1)*q_aux[1] == 0  # Spin
            # Charge changes: (0,1)→(-1,0), (1,0)→(0,-1) - both remove one particle and spin-up
            assert q_out[0] == q_in[0] - 1
            assert q_out[1] == q_in[1] - 1
    
    def test_f_dn_charge_conservation_u1u1(self):
        """Test F_dn charge conservation with U(1)xU(1)."""
        Spc, Op = load_space("Band", "U1, U1")
        F_dn = Op["F_dn"]
        
        for (q_out, q_in, q_aux), block in F_dn.data.items():
            assert (+1)*q_out[0] + (-1)*q_in[0] + (-1)*q_aux[0] == 0
            assert (+1)*q_out[1] + (-1)*q_in[1] + (-1)*q_aux[1] == 0
            # Charge changes: (0,-1)→(-1,0), (1,0)→(0,1) - both remove one particle and spin-down
            assert q_out[0] == q_in[0] - 1
            assert q_out[1] == q_in[1] + 1
    
    def test_spin_charge_conservation(self):
        """Test spin operators conserve particle number."""
        Spc, Op = load_space("Band", "U1, U1")
        
        # Sp should conserve particle number, increase spin by 2
        Sp = Op["Sp"]
        for (q_out, q_in, q_aux), block in Sp.data.items():
            assert q_out[0] == q_in[0]  # Same particle number
            assert q_out[1] == q_in[1] + 2  # Increase spin
        
        # Sm should conserve particle number, decrease spin by 2
        Sm = Op["Sm"]
        for (q_out, q_in, q_aux), block in Sm.data.items():
            assert q_out[0] == q_in[0]  # Same particle number
            assert q_out[1] == q_in[1] - 2  # Decrease spin


class TestBandMatrixElements:
    """Test Band operator matrix elements."""
    
    def test_f_up_matrix_elements(self):
        """Test F_up matrix elements."""
        Spc, Op = load_space("Band", "U1, U1")
        F_up = Op["F_up"]
        
        # F_up|↑⟩ = |0⟩: (0,1) → (-1,0)
        key1 = ((-1, 0), (0, 1), (-1, -1))
        assert key1 in F_up.data
        assert math.isclose(F_up.data[key1][0, 0, 0].item(), 1.0)
        
        # F_up|↑↓⟩ = |↓⟩: (1,0) → (0,-1)
        key2 = ((0, -1), (1, 0), (-1, -1))
        assert key2 in F_up.data
        assert math.isclose(F_up.data[key2][0, 0, 0].item(), 1.0)
    
    def test_f_dn_matrix_elements(self):
        """Test F_dn matrix elements."""
        Spc, Op = load_space("Band", "U1, U1")
        F_dn = Op["F_dn"]
        
        # F_dn|↓⟩ = |0⟩: (0,-1) → (-1,0)
        key1 = ((-1, 0), (0, -1), (-1, 1))
        assert key1 in F_dn.data
        assert math.isclose(F_dn.data[key1][0, 0, 0].item(), 1.0)
        
        # F_dn|↑↓⟩ = -|↑⟩ (minus sign from anticommutation): (1,0) → (0,1)
        key2 = ((0, 1), (1, 0), (-1, 1))
        assert key2 in F_dn.data
        assert math.isclose(F_dn.data[key2][0, 0, 0].item(), -1.0)
    
    def test_sz_eigenvalues(self):
        """Test Sz eigenvalues (zero eigenvalues trimmed)."""
        Spc, Op = load_space("Band", "U1, U1")
        Sz = Op["Sz"]
        
        # Collect diagonal elements
        eigenvalues = {}
        for (q_out, q_in), block in Sz.data.items():
            if q_out == q_in:
                eigenvalues[q_in] = block[0, 0]
        
        # Check expected non-zero values (zero eigenvalues have been trimmed)
        assert math.isclose(eigenvalues[(0, -1)], -0.5)  # |↓⟩
        assert math.isclose(eigenvalues[(0, 1)], 0.5)    # |↑⟩
        # Zero eigenvalues for |0⟩ and |↑↓⟩ are trimmed
        assert (-1, 0) not in eigenvalues  # |0⟩ with Sz=0 trimmed
        assert (1, 0) not in eigenvalues   # |↑↓⟩ with Sz=0 trimmed
    
    def test_spin_ladder_operators(self):
        """Test spin ladder operator matrix elements."""
        Spc, Op = load_space("Band", "U1, U1")
        Sp = Op["Sp"]
        Sm = Op["Sm"]
        
        # Sp|↓⟩ = |↑⟩: (0,-1) → (0,1) with spherical convention
        # For spin-1/2: coefficient = -1/sqrt(2)
        key_p = ((0, 1), (0, -1), (0, 2))
        assert key_p in Sp.data
        assert torch.isclose(Sp.data[key_p][0, 0, 0], torch.tensor(-1.0 / math.sqrt(2.0), dtype=Sp.dtype), atol=1e-9)
        
        # Sm|↑⟩ = |↓⟩: (0,1) → (0,-1) with spherical convention
        # For spin-1/2: coefficient = +1/sqrt(2)
        key_m = ((0, -1), (0, 1), (0, -2))
        assert key_m in Sm.data
        assert torch.isclose(Sm.data[key_m][0, 0, 0], torch.tensor(+1.0 / math.sqrt(2.0), dtype=Sm.dtype), atol=1e-9)


class TestBandErrorHandling:
    """Test error handling for Band systems."""
    
    def test_unsupported_symmetry(self):
        """Test error for unsupported symmetry."""
        with pytest.raises(ValueError, match="Unsupported symmetry"):
            load_space("Band", "U1")
        
        with pytest.raises(ValueError, match="Unsupported symmetry"):
            load_space("Band", "Z2")
    
    def test_band_different_symmetries(self):
        """Test that different symmetries produce different structures."""
        Spc_u1u1, Op_u1u1 = load_space("Band", "U1, U1")
        Spc_z2u1, Op_z2u1 = load_space("Band", "Z2, U1")
        
        # Different number of sectors
        assert len(Spc_u1u1.sectors) == 4
        assert len(Spc_z2u1.sectors) == 3
        
        # But same total dimension
        assert Spc_u1u1.dim == Spc_z2u1.dim == 4


class TestLoadSpinSU2:
    """Test SU(2)-symmetric spin space and operator construction."""

    # ── Space structure ────────────────────────────────────────────────────────

    def test_spc_group(self):
        """Test that Spc uses SU2Group."""
        Spc, _ = load_space("Spin", "SU2", {"J": 0.5})
        assert isinstance(Spc.group, SU2Group)

    def test_spc_direction(self):
        """Test that Spc has IN direction."""
        Spc, _ = load_space("Spin", "SU2", {"J": 0.5})
        assert Spc.direction == Direction.IN

    def test_spc_dim_is_one_multiplet(self):
        """Spc.dim is always 1 (one irrep, no degeneracy) for all J."""
        for J in [0, 0.5, 1, 1.5, 2, 2.5]:
            Spc, _ = load_space("Spin", "SU2", {"J": J})
            assert Spc.dim == 1, f"Expected dim=1 for J={J}, got {Spc.dim}"

    def test_spc_num_states(self):
        """Spc.num_states equals the physical dimension 2J+1."""
        for J in [0, 0.5, 1, 1.5, 2, 2.5, 3]:
            Spc, _ = load_space("Spin", "SU2", {"J": J})
            expected = int(2 * J + 1)
            assert Spc.num_states == expected, (
                f"J={J}: expected num_states={expected}, got {Spc.num_states}"
            )

    def test_spc_single_sector_charge(self):
        """Spc has exactly one sector with charge = 2J."""
        for J in [0, 0.5, 1, 1.5, 2]:
            Spc, _ = load_space("Spin", "SU2", {"J": J})
            assert len(Spc.sectors) == 1
            assert Spc.sectors[0].charge == int(round(2 * J))
            assert Spc.sectors[0].dim == 1

    # ── Operator keys ──────────────────────────────────────────────────────────

    def test_operator_keys(self):
        """Op contains exactly 'S' and 'vac'."""
        for J in [0, 0.5, 1, 1.5, 2]:
            _, Op = load_space("Spin", "SU2", {"J": J})
            assert set(Op.keys()) == {"S", "vac"}

    # ── S tensor structure ─────────────────────────────────────────────────────

    def test_s_has_three_indices(self):
        """S operator is a 3-index tensor."""
        _, Op = load_space("Spin", "SU2", {"J": 1.0})
        assert len(Op["S"].indices) == 3

    def test_s_itags(self):
        """S operator has correct itags."""
        _, Op = load_space("Spin", "SU2", {"J": 1.0})
        assert Op["S"].itags == ("_init_", "_init_", "_aux_")

    def test_s_index_directions(self):
        """S tensor has directions (IN, OUT, OUT)."""
        _, Op = load_space("Spin", "SU2", {"J": 1.0})
        S = Op["S"]
        assert S.indices[0].direction == Direction.IN
        assert S.indices[1].direction == Direction.OUT
        assert S.indices[2].direction == Direction.OUT

    def test_s_aux_index(self):
        """Auxiliary index of S carries rank-1 (spin-1, charge=2) character."""
        for J in [0.5, 1, 1.5, 2]:
            _, Op = load_space("Spin", "SU2", {"J": J})
            aux = Op["S"].indices[2]
            assert isinstance(aux.group, SU2Group)
            assert len(aux.sectors) == 1
            assert aux.sectors[0].charge == 2   # spin-1 in 2j convention
            assert aux.sectors[0].dim == 1

    def test_s_is_tensor(self):
        """S is a Tensor instance."""
        _, Op = load_space("Spin", "SU2", {"J": 0.5})
        assert isinstance(Op["S"], Tensor)

    # ── Non-Abelian block structure ────────────────────────────────────────────

    def test_s_intw_populated(self):
        """S.intw is populated (mandatory for non-Abelian tensors)."""
        for J in [0.5, 1, 1.5, 2]:
            _, Op = load_space("Spin", "SU2", {"J": J})
            assert Op["S"].intw is not None

    def test_s_intw_keys_match_data_keys(self):
        """S.intw and S.data have identical key sets."""
        for J in [0, 0.5, 1, 1.5, 2]:
            _, Op = load_space("Spin", "SU2", {"J": J})
            S = Op["S"]
            assert set(S.intw.keys()) == set(S.data.keys())

    def test_s_bridge_num_external(self):
        """Each Bridge in S.intw has 3 external edges (one per tensor index)."""
        for J in [0.5, 1, 1.5, 2]:
            _, Op = load_space("Spin", "SU2", {"J": J})
            S = Op["S"]
            for bridge in S.intw.values():
                assert bridge.num_external == 3

    def test_s_bridge_om_dimension(self):
        """Bridge OM dimension is 1 (unique fusion channel for each block)."""
        for J in [0.5, 1, 1.5, 2]:
            _, Op = load_space("Spin", "SU2", {"J": J})
            S = Op["S"]
            for bridge in S.intw.values():
                assert bridge.om_dimension == 1

    def test_s_block_shape(self):
        """S data block has shape (1, 1, 1, 1) = (dim_out, dim_in, dim_aux, num_components)."""
        for J in [0.5, 1, 1.5, 2]:
            two_J = int(round(2 * J))
            _, Op = load_space("Spin", "SU2", {"J": J})
            S = Op["S"]
            assert S.data[(two_J, two_J, 2)].shape == (1, 1, 1, 1)

    # ── Reduced matrix elements ────────────────────────────────────────────────

    def test_s_data_value(self):
        """data block stores sqrt(J(J+1)), Bridge weight stores sqrt(2J+1)."""
        for J in [0.5, 1, 1.5, 2, 2.5]:
            two_J = int(round(2 * J))
            _, Op = load_space("Spin", "SU2", {"J": J})
            S = Op["S"]
            key = (two_J, two_J, 2)

            data_val = S.data[key][0, 0, 0, 0].item()
            weight_val = S.intw[key].weights[0, 0].item()

            assert math.isclose(data_val, math.sqrt(J * (J + 1)), rel_tol=1e-12), (
                f"J={J}: data value {data_val} != sqrt(J(J+1))={math.sqrt(J*(J+1))}"
            )
            assert math.isclose(weight_val, math.sqrt(2 * J + 1), rel_tol=1e-12), (
                f"J={J}: weight {weight_val} != sqrt(2J+1)={math.sqrt(2*J+1)}"
            )

    def test_s_full_rme(self):
        """data * weight = sqrt(J(J+1)(2J+1)), the full reduced matrix element."""
        for J in [0.5, 1, 1.5, 2, 2.5]:
            two_J = int(round(2 * J))
            _, Op = load_space("Spin", "SU2", {"J": J})
            S = Op["S"]
            key = (two_J, two_J, 2)

            full_rme = S.data[key][0, 0, 0, 0].item() * S.intw[key].weights[0, 0].item()
            expected = math.sqrt(J * (J + 1) * (2 * J + 1))
            assert math.isclose(full_rme, expected, rel_tol=1e-12), (
                f"J={J}: full RME {full_rme} != sqrt(J(J+1)(2J+1))={expected}"
            )

    def test_s_spin_zero_is_empty(self):
        """For J=0, S has no blocks (charge conservation forbids the block (0,0,2))."""
        _, Op = load_space("Spin", "SU2", {"J": 0})
        S = Op["S"]
        assert len(S.data) == 0
        assert len(S.intw) == 0

    def test_s_nonzero_j_has_one_block(self):
        """For J > 0, S has exactly one block."""
        for J in [0.5, 1, 1.5, 2, 2.5]:
            _, Op = load_space("Spin", "SU2", {"J": J})
            assert len(Op["S"].data) == 1

    def test_s_block_key(self):
        """Block key is (2J, 2J, 2) for all J > 0."""
        for J in [0.5, 1, 1.5, 2]:
            two_J = int(round(2 * J))
            _, Op = load_space("Spin", "SU2", {"J": J})
            assert (two_J, two_J, 2) in Op["S"].data

    # ── Vacuum index ───────────────────────────────────────────────────────────

    def test_vac_is_index(self):
        """vac is an Index, not a Tensor."""
        _, Op = load_space("Spin", "SU2", {"J": 0.5})
        assert isinstance(Op["vac"], Index)

    def test_vac_structure(self):
        """vac has SU2Group, IN direction, single sector with charge=0 and dim=1."""
        for J in [0, 0.5, 1]:
            _, Op = load_space("Spin", "SU2", {"J": J})
            vac = Op["vac"]
            assert isinstance(vac.group, SU2Group)
            assert vac.direction == Direction.IN
            assert len(vac.sectors) == 1
            assert vac.sectors[0].charge == 0
            assert vac.sectors[0].dim == 1

    # ── Error handling ─────────────────────────────────────────────────────────

    def test_unsupported_symmetry_still_raises(self):
        """Non-SU2/U1 symmetries for Spin still raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported symmetry"):
            load_space("Spin", "Z2", {"J": 0.5})

    def test_missing_j_raises(self):
        """Missing J option raises ValueError."""
        with pytest.raises(ValueError, match="Option 'J'"):
            load_space("Spin", "SU2", {})

    def test_negative_j_raises(self):
        """Negative J raises ValueError."""
        with pytest.raises(ValueError, match="J must be non-negative"):
            load_space("Spin", "SU2", {"J": -1})

    def test_non_half_integer_j_raises(self):
        """Non-half-integer J raises ValueError."""
        with pytest.raises(ValueError, match="J must be a half-integer"):
            load_space("Spin", "SU2", {"J": 0.3})
