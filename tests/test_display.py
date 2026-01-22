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


"""Tests for display module functions."""

import numpy as np

from nicole import Direction, Index, Tensor, U1Group, Sector
from nicole.display import (
    _charge_components,
    _format_bytes,
    _format_count_list,
    _format_single_value,
    _group_signature,
    tensor_summary,
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
    arr = np.array([[3.14159]])
    result = _format_single_value(arr)
    assert "3.14159" in result


def test_format_single_value_complex():
    """Test _format_single_value with complex numbers."""
    arr = np.array([[2.0 + 3.0j]])
    result = _format_single_value(arr)
    assert "2" in result
    assert "3" in result
    assert "i" in result


def test_format_single_value_negative_imaginary():
    """Test _format_single_value with negative imaginary part."""
    arr = np.array([[1.0 - 2.0j]])
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
    # Should have right-aligned padding
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
    
    tensor = Tensor.zeros([idx, idx.flip()], dtype=np.float32, itags=["a", "b"])
    
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
    
    tensor = Tensor(indices=(idx, idx.flip()), itags=("a", "b"), data={}, dtype=np.float64)
    
    summary = str(tensor)
    
    assert "no sectors" in summary or "0 x" in summary


def test_tensor_summary_complex_dtype():
    """Test tensor_summary with complex dtype."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.random([idx, idx.flip()], dtype=np.complex128, seed=1, itags=["a", "b"])
    
    summary = str(tensor)
    
    assert "complex" in summary


def test_tensor_repr_equals_str():
    """Test that __repr__ equals __str__."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    tensor = Tensor.zeros([idx, idx.flip()], itags=["a", "b"])
    
    assert repr(tensor) == str(tensor)


def test_tensor_summary_three_indices():
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
    
    data = {(0, 0): np.zeros((2, 2))}
    tensor = Tensor(indices=(idx, idx.flip()), itags=("a", "b"), data=data, dtype=np.float64, label="MyTensor")
    
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

