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


"""Tests for display module functions."""

import torch

from nicole import Direction, Index, Tensor, U1Group, Sector
from nicole.symmetry.unitary import SU2Group
from nicole.display import (
    _charge_components,
    _format_bytes,
    _format_count_list,
    _format_single_value,
    _group_signature
)


# Helper function tests

def test_format_bytes_small():
    """Test _format_bytes with small values."""
    assert _format_bytes(0) == "0 B"
    assert _format_bytes(100) == "100 B"
    assert _format_bytes(1023) == "1023 B"


def test_format_bytes_kilobytes():
    """Test _format_bytes with kilobyte values."""
    assert _format_bytes(1024) == "1 kB"
    assert _format_bytes(2048) == "2 kB"
    assert _format_bytes(1536) == "1.5 kB"


def test_format_bytes_megabytes():
    """Test _format_bytes with megabyte values."""
    assert _format_bytes(1024 * 1024) == "1 MB"
    assert _format_bytes(5 * 1024 * 1024) == "5 MB"


def test_format_bytes_gigabytes():
    """Test _format_bytes with gigabyte values."""
    assert _format_bytes(1024 * 1024 * 1024) == "1 GB"


def test_format_single_value_real():
    """Test _format_single_value with real numbers."""
    arr = torch.tensor([[3.14159]])
    result = _format_single_value(arr)
    assert "3.142" in result


def test_format_single_value_complex():
    """Test _format_single_value with complex numbers."""
    arr = torch.tensor([[2.0 + 3.0j]])
    result = _format_single_value(arr)
    assert "2" in result
    assert "3" in result
    assert "i" in result


def test_format_single_value_negative_imaginary():
    """Test _format_single_value with negative imaginary part."""
    arr = torch.tensor([[1.0 - 2.0j]])
    result = _format_single_value(arr)
    assert "1" in result
    assert "2" in result
    assert "-" in result


def test_format_count_list_single():
    """Test _format_count_list with single element."""
    assert _format_count_list([5]) == "5"


def test_format_count_list_multiple():
    """Test _format_count_list with multiple elements."""
    result = _format_count_list([2, 3, 5])
    assert "2" in result
    assert "3" in result
    assert "5" in result
    assert "x" in result


def test_format_count_list_padding():
    """Test _format_count_list with different widths."""
    result = _format_count_list([1, 10, 100])
    # Three elements should be joined by two "x" separators
    assert result.count("x") == 2


def test_format_count_list_empty():
    """Test _format_count_list with empty list."""
    assert _format_count_list([]) == "0"


def test_charge_components_scalar():
    """Test _charge_components with scalar charge."""
    assert _charge_components(5) == (5,)
    assert _charge_components(0) == (0,)


def test_charge_components_tuple():
    """Test _charge_components with tuple charge."""
    assert _charge_components((1, 2, 3)) == (1, 2, 3)


def test_charge_components_list():
    """Test _charge_components with list charge."""
    assert _charge_components([1, 2]) == (1, 2)


def test_group_signature_abelian():
    """Test _group_signature with Abelian group."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    sig = _group_signature([idx], 1)
    assert "A" in sig


def test_group_signature_multiple_components():
    """Test _group_signature with multiple charge components."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    sig = _group_signature([idx], 3)
    assert sig.count("A") == 3
    assert sig.count(",") == 2


def test_group_signature_empty():
    """Test _group_signature with empty indices."""
    sig = _group_signature([], 0)
    assert sig == ""


# tensor_summary tests

def test_tensor_summary_basic():
    """Test tensor_summary with basic tensor."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.zeros([idx, idx.flip()], itags=["a", "b"])
    
    summary = str(tensor)
    
    assert "Tensor" in summary
    assert "a" in summary


def test_tensor_summary_includes_norm():
    """Test that tensor_summary includes norm."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], seed=1, itags=["a", "b"])
    
    summary = str(tensor)
    
    assert "norm" in summary


def test_tensor_summary_includes_dtype():
    """Test that tensor_summary includes dtype."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.zeros([idx, idx.flip()], dtype=torch.float32, itags=["a", "b"])
    
    summary = str(tensor)
    
    assert "float32" in summary


def test_tensor_summary_includes_blocks():
    """Test that tensor_summary includes block information."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 3)))
    
    tensor = Tensor.random([idx1, idx2], seed=1, itags=["a", "b"])
    
    summary = str(tensor)
    
    # Should have block listings
    assert "1." in summary or "2." in summary


def test_tensor_summary_direction_markers():
    """Test that tensor_summary marks OUT directions with asterisk."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.zeros([idx_out, idx_in], itags=["out", "in"])
    
    summary = str(tensor)
    
    # OUT direction should have asterisk
    assert "out*" in summary
    # IN direction should not
    assert "in*" not in summary or "in," in summary  # "in," would indicate it's not marked


def test_tensor_summary_multiple_blocks():
    """Test tensor_summary with multiple blocks."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(-1, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 2), Sector(1, 2)))
    
    tensor = Tensor.random([idx1, idx2], seed=1, itags=["a", "b"])
    
    summary = str(tensor)
    
    # Should list blocks (at most 9 shown)
    lines = summary.split("\n")
    block_lines = [l for l in lines if l.strip().startswith(("1.", "2.", "3."))]
    assert len(block_lines) > 0


def test_tensor_summary_truncates_many_blocks():
    """Test that tensor_summary truncates when many blocks."""
    group = U1Group()
    # Create indices with many sectors
    charges = [(i, 1) for i in range(10)]
    idx1 = Index(Direction.OUT, group, sectors=tuple(Sector(c, d) for c, d in charges))
    idx2 = Index(Direction.IN, group, sectors=tuple(Sector(c, d) for c, d in charges))
    
    tensor = Tensor.random([idx1, idx2], seed=1, itags=["a", "b"])
    
    summary = str(tensor)
    
    # Should show "more" indicator if > 9 blocks
    if len(tensor.data) > 9:
        assert "more" in summary


def test_tensor_summary_scalar_block():
    """Test tensor_summary with scalar (1x1) blocks."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    
    tensor = Tensor.random([idx, idx.flip()], seed=1, itags=["a", "b"])
    
    summary = str(tensor)
    
    # Scalar blocks should show the value
    assert "." in summary  # Should show actual value with decimal point


def test_tensor_summary_empty_tensor():
    """Test tensor_summary with tensor having no blocks."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=())
    
    tensor = Tensor(indices=(idx, idx.flip()), itags=("a", "b"), data={}, dtype=torch.float64)
    
    summary = str(tensor)
    
    assert "no sectors" in summary or "0 x" in summary


def test_tensor_summary_complex_dtype():
    """Test tensor_summary with complex dtype."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], dtype=torch.complex128, seed=1, itags=["a", "b"])
    
    summary = str(tensor)
    
    assert "complex" in summary


def test_tensor_repr_equals_str():
    """Test that __repr__ equals __str__."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.zeros([idx, idx.flip()], itags=["a", "b"])
    
    assert repr(tensor) == str(tensor)


def test_tensor_summary_3rd_order():
    """Test tensor_summary with three indices."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    
    tensor = Tensor.random([idx1, idx2, idx3], seed=1, itags=["a", "b", "c"])
    
    summary = str(tensor)
    
    assert "3x" in summary or "3-D" in summary
    assert "a" in summary
    assert "b" in summary
    assert "c" in summary


def test_tensor_summary_custom_label():
    """Test tensor_summary with custom label."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    data = {(0, 0): torch.zeros((2, 2))}
    tensor = Tensor(indices=(idx, idx.flip()), itags=("a", "b"), data=data, dtype=torch.float64, label="MyTensor")
    
    summary = str(tensor)
    
    assert "MyTensor" in summary


def test_tensor_summary_formatting_consistent():
    """Test that tensor_summary produces consistent formatting."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    tensor = Tensor.random([idx, idx.flip()], seed=1, itags=["a", "b"])
    
    summary1 = str(tensor)
    summary2 = str(tensor)
    
    # Should be identical
    assert summary1 == summary2


# SU2 tensor_summary tests

def test_tensor_summary_su2_identity():
    """Test tensor_summary with SU2 identity tensor."""
    from nicole.identity import identity
    
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(2, 3)))
    
    tensor = identity(idx)
    summary = str(tensor)
    
    # Check basic structure
    assert "Tensor" in summary
    
    # Check CGC dims show irrep dimensions (2j+1)
    # For spin-0 (two_j=0): irrep_dim = 1
    # For spin-1 (two_j=2): irrep_dim = 3
    assert "1x1" in summary  # spin-0 block
    assert "3x3" in summary  # spin-1 block
    
    # Check multiplet and state counts
    # idx.dim = 2 + 3 = 5 (multiplet count)
    # idx.num_states = 2*1 + 3*3 = 11 (state count)
    # Identity has two such indices
    assert "5 x 5 => 11 x 11" in summary


def test_tensor_summary_su2_isometry():
    """Test tensor_summary with SU2 isometry tensor."""
    from nicole.identity import isometry
    
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))  # spin-1/2
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 3),))  # spin-1/2
    
    tensor = isometry(idx1, idx2)
    summary = str(tensor)
    
    # Check basic structure
    assert "Tensor" in summary
    
    # Check CGC dims show irrep dimensions
    # idx1: spin-1/2 (two_j=1) -> irrep_dim = 2
    # idx2: spin-1/2 (two_j=1) -> irrep_dim = 2
    # Blocks should show pattern like "2x2xN" where N is fused irrep_dim
    assert "2x2x" in summary  # First two indices have irrep_dim=2
    
    # Check multiplet and state counts
    # idx1.dim = 2, idx1.num_states = 2 * 2 = 4
    # idx2.dim = 3, idx2.num_states = 3 * 2 = 6
    # For fused index: will depend on which charges appear
    # 1/2 ⊗ 1/2 = 0 ⊕ 1, so fused could be 0 (dim 1, irrep 1) or 2 (dim varies, irrep 3)
    assert "2 x 3" in summary  # multiplet counts for first two indices


def test_tensor_summary_su2_random():
    """Test tensor_summary with random SU2 tensor."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(2, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(2, 2)))
    
    tensor = Tensor.random([idx1, idx2], seed=1, itags=["left", "right"])
    summary = str(tensor)
    
    # Check basic structure
    assert "Tensor" in summary
    assert "left" in summary
    assert "right" in summary
    
    # Check CGC dims
    # spin-0 (two_j=0): irrep_dim = 1
    # spin-1 (two_j=2): irrep_dim = 3
    assert "1x1" in summary  # (0,0) block
    assert "3x3" in summary  # (2,2) block
    
    # Check multiplet and state counts
    # idx1.dim = 2 + 1 = 3, idx1.num_states = 2*1 + 1*3 = 5
    # idx2.dim = 3 + 2 = 5, idx2.num_states = 3*1 + 2*3 = 9
    assert "3 x 5 => 5 x 9" in summary


def test_tensor_summary_su2_trims_reduced_multiplicity():
    """Test that tensor_summary trims trailing reduced multiplicity dimension."""
    from nicole.identity import isometry
    
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    tensor = isometry(idx1, idx2)
    summary = str(tensor)
    
    # The actual block shapes in data have trailing dimension for reduced multiplicity
    # But display should trim it
    # Should NOT see something like "2x2x4x1" (4-D with trailing 1)
    # Should see "2x2x4" (3-D, trimmed)
    
    # Check that we see 3-D shapes (not 4-D)
    assert "2x2x" in summary
    # Verify it's not showing 4-D
    lines = summary.split("\n")
    for line in lines:
        if "2x2x" in line:
            # Should not have a 4th dimension shown
            import re
            # Look for pattern like "2x2xNxM" which would indicate 4 dimensions
            assert not re.search(r'2x2x\d+x\d+', line), f"Found 4-D shape in: {line}"


# Sign suffix tests

def test_tensor_summary_su2_sign_suffix_positive_scalar_block():
    """Test that {+} is appended for a scalar SU(2) block with positive single weight.
    
    Sector(0, 1) gives a 1x1 state block so the value_repr path is exercised.
    The default weight initialised by Bridge.from_block is [[1.0]], numel == 1.
    """
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    
    tensor = Tensor.zeros([idx, idx.flip()], itags=["a", "b"])
    summary = str(tensor)
    
    assert "{+}" in summary


def test_tensor_summary_su2_sign_suffix_negative_scalar_block():
    """Test that {-} is appended for a scalar SU(2) block with negative single weight.
    
    The weight tensor is mutated in-place to -1.0 (frozen dataclass allows
    in-place tensor operations).
    """
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    
    tensor = Tensor.zeros([idx, idx.flip()], itags=["a", "b"])
    # Flip the sign in-place; frozen dataclass allows tensor mutation
    tensor.intw[(0, 0)].weights.fill_(-1.0)
    summary = str(tensor)
    
    assert "{-}" in summary


def test_tensor_summary_su2_sign_suffix_positive_large_block():
    """Test that {+} is appended on the byte_repr path for a non-scalar block.
    
    Sector(2, 2) (spin-1, 2 multiplets) yields a 2x2 state block so the
    byte_repr branch is taken rather than the value_repr branch.
    """
    group = SU2Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(2, 2),))
    
    tensor = Tensor.zeros([idx, idx.flip()], itags=["a", "b"])
    summary = str(tensor)
    
    assert "{+}" in summary


def test_tensor_summary_su2_sign_suffix_absent_for_abelian():
    """Test that no sign suffix appears for Abelian (U1) tensors."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    
    tensor = Tensor.zeros([idx, idx.flip()], itags=["a", "b"])
    summary = str(tensor)
    
    assert "{+}" not in summary
    assert "{-}" not in summary

