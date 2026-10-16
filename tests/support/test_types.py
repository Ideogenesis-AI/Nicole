# Copyright (C) 2025 Changkai Zhang.
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


"""Tests for core types: Direction, Sector, Charge, and the dtype helpers."""

import pytest
import torch

from nicole import Direction, Index, Sector, Tensor, U1Group
from nicole.typing import (
    complex_dtype_for,
    normalize_dtype_for_device,
    promote_tensors,
    real_dtype_for,
    scalar_result_dtype,
)


def test_direction_values():
    """Test Direction enum values."""
    assert Direction.IN == 1
    assert Direction.OUT == -1


def test_direction_reverse():
    """Test Direction.reverse() method."""
    assert Direction.IN.reverse() == Direction.OUT
    assert Direction.OUT.reverse() == Direction.IN


def test_direction_double_reverse():
    """Test that reversing twice returns original direction."""
    assert Direction.IN.reverse().reverse() == Direction.IN
    assert Direction.OUT.reverse().reverse() == Direction.OUT


def test_sector_construction():
    """Test basic Sector construction."""
    s = Sector(charge=0, dim=5)
    assert s.charge == 0
    assert s.dim == 5


def test_sector_with_tuple_charge():
    """Test Sector with tuple-valued charge."""
    s = Sector(charge=(1, 2), dim=3)
    assert s.charge == (1, 2)
    assert s.dim == 3


def test_sector_with_string_charge():
    """Test Sector with string charge (hashable)."""
    s = Sector(charge="A", dim=2)
    assert s.charge == "A"
    assert s.dim == 2


def test_sector_invalid_dimension_zero():
    """Test that Sector rejects zero dimension."""
    with pytest.raises(ValueError, match="dimension must be positive"):
        Sector(charge=0, dim=0)


def test_sector_invalid_dimension_negative():
    """Test that Sector rejects negative dimension."""
    with pytest.raises(ValueError, match="dimension must be positive"):
        Sector(charge=1, dim=-5)


def test_sector_frozen():
    """Test that Sector is immutable (frozen dataclass)."""
    s = Sector(charge=0, dim=5)
    with pytest.raises(AttributeError):
        s.charge = 1  # type: ignore
    with pytest.raises(AttributeError):
        s.dim = 10  # type: ignore


def test_sector_equality():
    """Test Sector equality."""
    s1 = Sector(charge=0, dim=5)
    s2 = Sector(charge=0, dim=5)
    s3 = Sector(charge=1, dim=5)
    s4 = Sector(charge=0, dim=3)
    
    assert s1 == s2
    assert s1 != s3
    assert s1 != s4


def test_sector_hashable():
    """Test that Sector can be used in sets and as dict keys."""
    s1 = Sector(charge=0, dim=5)
    s2 = Sector(charge=1, dim=3)
    s3 = Sector(charge=0, dim=5)
    
    sector_set = {s1, s2, s3}
    assert len(sector_set) == 2  # s1 and s3 are equal
    
    sector_dict = {s1: "first", s2: "second"}
    assert sector_dict[s3] == "first"


def test_charge_type_hashable():
    """Test that various hashable types can be used as charges."""
    # Integer charge
    s1 = Sector(charge=42, dim=1)
    assert s1.charge == 42
    
    # Tuple charge
    s2 = Sector(charge=(1, 2, 3), dim=1)
    assert s2.charge == (1, 2, 3)
    
    # String charge
    s3 = Sector(charge="quantum", dim=1)
    assert s3.charge == "quantum"
    
    # Float charge (hashable but unusual)
    s4 = Sector(charge=3.14, dim=1)
    assert s4.charge == 3.14


# Dtype helper tests

def test_complex_dtype_for():
    """Test that each real dtype maps to the complex dtype of its precision."""
    assert complex_dtype_for(torch.float16) is torch.complex32
    assert complex_dtype_for(torch.float32) is torch.complex64
    assert complex_dtype_for(torch.float64) is torch.complex128
    
    # bfloat16 has no complex counterpart of its own width
    assert complex_dtype_for(torch.bfloat16) is torch.complex64


def test_complex_dtype_for_complex_input():
    """Test that an already complex dtype is returned unchanged."""
    assert complex_dtype_for(torch.complex32) is torch.complex32
    assert complex_dtype_for(torch.complex64) is torch.complex64
    assert complex_dtype_for(torch.complex128) is torch.complex128


def test_complex_dtype_for_non_floating():
    """Test that integer and boolean dtypes fall back to complex128."""
    assert complex_dtype_for(torch.int32) is torch.complex128
    assert complex_dtype_for(torch.int64) is torch.complex128
    assert complex_dtype_for(torch.bool) is torch.complex128


def test_real_dtype_for():
    """Test that each complex dtype maps to the real dtype of its precision."""
    assert real_dtype_for(torch.complex32) is torch.float16
    assert real_dtype_for(torch.complex64) is torch.float32
    assert real_dtype_for(torch.complex128) is torch.float64


def test_real_dtype_for_real_input():
    """Test that a real dtype is returned unchanged, so the helper is total."""
    assert real_dtype_for(torch.float32) is torch.float32
    assert real_dtype_for(torch.float64) is torch.float64
    assert real_dtype_for(torch.int64) is torch.int64


def test_dtype_for_round_trip():
    """Test that promoting to complex and back recovers the original dtype."""
    for dtype in (torch.float16, torch.float32, torch.float64):
        assert real_dtype_for(complex_dtype_for(dtype)) is dtype


def test_scalar_result_dtype_complex_scalar_keeps_precision():
    """Test that a complex scalar widens to its own precision, not complex128."""
    assert scalar_result_dtype(torch.float32, 1 + 0j) is torch.complex64
    assert scalar_result_dtype(torch.float64, 1 + 0j) is torch.complex128
    
    # An already complex tensor is unaffected
    assert scalar_result_dtype(torch.complex64, 1 + 0j) is torch.complex64


def test_scalar_result_dtype_real_scalar():
    """Test that a real scalar leaves a floating-point dtype alone."""
    assert scalar_result_dtype(torch.float32, 2.0) is torch.float32
    assert scalar_result_dtype(torch.float64, 2) is torch.float64
    assert scalar_result_dtype(torch.complex128, 2.0) is torch.complex128


def test_scalar_result_dtype_integer_tensor():
    """Test that an integer dtype widens under a float scalar but not an int."""
    assert scalar_result_dtype(torch.int64, 2.0) is torch.get_default_dtype()
    assert scalar_result_dtype(torch.int64, 2) is torch.int64


def test_promote_tensors_matching_returns_same_objects():
    """Test that matching dtypes short-circuit and return the inputs themselves."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    a = Tensor.random([idx, idx.flip()], seed=1, itags=["a", "b"])
    b = Tensor.random([idx, idx.flip()], seed=2, itags=["a", "b"])
    
    promoted = promote_tensors(a, b)
    
    # Identity, not equality: the real path must neither cast nor allocate
    assert promoted[0] is a
    assert promoted[1] is b


def test_promote_tensors_mixed_dtypes():
    """Test that a mixed pair is brought to the common dtype."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    a = Tensor.random([idx, idx.flip()], seed=1, itags=["a", "b"])
    b = Tensor.random([idx, idx.flip()], seed=2, itags=["a", "b"],
                      dtype=torch.complex128)
    
    promoted_a, promoted_b = promote_tensors(a, b)
    
    assert promoted_a.dtype is torch.complex128
    assert promoted_b.dtype is torch.complex128
    
    # Only the operand that needed casting is a new object
    assert promoted_a is not a
    assert promoted_b is b


def test_promote_tensors_single_operand():
    """Test that a lone tensor is returned untouched."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    a = Tensor.random([idx, idx.flip()], seed=1, itags=["a", "b"])
    
    assert promote_tensors(a)[0] is a


def test_normalize_dtype_for_device_mps():
    """Test that the dtypes MPS lacks are demoted to the precision it has.

    This needs no MPS hardware: torch.device('mps') is only a descriptor, so
    the mapping is exercised wherever the suite runs.
    """
    assert normalize_dtype_for_device(torch.float64, 'mps') is torch.float32
    assert normalize_dtype_for_device(torch.complex128, 'mps') is torch.complex64
    
    # Dtypes MPS supports pass through
    assert normalize_dtype_for_device(torch.float32, 'mps') is torch.float32
    assert normalize_dtype_for_device(torch.complex64, 'mps') is torch.complex64


def test_normalize_dtype_for_device_cpu():
    """Test that CPU imposes no demotion."""
    for dtype in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        assert normalize_dtype_for_device(dtype, 'cpu') is dtype


def test_normalize_dtype_for_device_accepts_device_object():
    """Test that a torch.device is accepted alongside a string."""
    assert normalize_dtype_for_device(torch.float64, torch.device('mps')) is torch.float32
    assert normalize_dtype_for_device(torch.float64, torch.device('cpu')) is torch.float64

