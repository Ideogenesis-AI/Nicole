# Copyright (C) 2026 Changkai Zhang.
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


"""Tests for oplus (direct sum) operation."""

import numpy as np
import pytest

from nicole import Direction, Index, Sector, Tensor, U1Group, Z2Group, contract, oplus
from nicole.symmetry.product import ProductGroup
from .utils import assert_charge_neutral


# ============================================================================
# Basic functionality tests
# ============================================================================

def test_oplus_all_axes_default():
    """Test that default behavior merges all axes."""
    group = U1Group()
    
    # Tensor A with sectors [(0, 2), (1, 3)]
    idx_A0 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_A1 = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 2)))
    A = Tensor.random([idx_A0, idx_A1], seed=1, itags=['i', 'j'])
    
    # Tensor B with sectors [(0, 1), (2, 2)]
    idx_B0 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
    idx_B1 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(2, 1)))
    B = Tensor.random([idx_B0, idx_B1], seed=2, itags=['i', 'j'])
    
    # Default: merge all axes
    C = oplus(A, B)
    
    # Check index 0: should have sectors [(0, 3), (1, 3), (2, 2)]
    assert len(C.indices[0].sectors) == 3
    dim_map_0 = C.indices[0].sector_dim_map()
    assert dim_map_0[0] == 3  # 2 + 1
    assert dim_map_0[1] == 3  # 3 + 0
    assert dim_map_0[2] == 2  # 0 + 2
    
    # Check index 1: should have sectors [(0, 5), (1, 2), (2, 1)]
    assert len(C.indices[1].sectors) == 3
    dim_map_1 = C.indices[1].sector_dim_map()
    assert dim_map_1[0] == 5  # 3 + 2
    assert dim_map_1[1] == 2  # 2 + 0
    assert dim_map_1[2] == 1  # 0 + 1
    
    # Check charge conservation
    assert_charge_neutral(C)


def test_oplus_simple_u1():
    """Test simple direct sum with non-overlapping sectors."""
    group = U1Group()
    
    # A has sector 0, B has sector 1 (non-overlapping)
    idx_A = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_B = Index(Direction.OUT, group, sectors=(Sector(1, 3),))
    
    A = Tensor.random([idx_A, idx_A.flip()], seed=10, itags=['a', 'b'])
    B = Tensor.random([idx_B, idx_B.flip()], seed=20, itags=['a', 'b'])
    
    C = oplus(A, B)
    
    # Result should have both sectors
    dim_map = C.indices[0].sector_dim_map()
    assert 0 in dim_map
    assert 1 in dim_map
    assert dim_map[0] == 2
    assert dim_map[1] == 3


def test_oplus_overlapping_sectors():
    """Test direct sum with some overlapping charge sectors."""
    group = U1Group()
    
    # Both have sector 0, but different dimensions
    idx_A = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_B = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
    
    A = Tensor.random([idx_A, idx_A.flip()], seed=30, itags=['a', 'b'])
    B = Tensor.random([idx_B, idx_B.flip()], seed=40, itags=['a', 'b'])
    
    C = oplus(A, B)
    
    # Check merged dimensions
    dim_map = C.indices[0].sector_dim_map()
    assert dim_map[0] == 3  # 2 + 1
    assert dim_map[1] == 3  # 3 + 0
    assert dim_map[2] == 2  # 0 + 2


def test_oplus_identical_sectors():
    """Test direct sum with identical sector structure produces block-diagonal."""
    group = U1Group()
    
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    A = Tensor.random([idx, idx.flip()], seed=50, itags=['a', 'b'])
    B = Tensor.random([idx, idx.flip()], seed=60, itags=['a', 'b'])
    
    C = oplus(A, B)
    
    # Dimensions should double
    dim_map = C.indices[0].sector_dim_map()
    assert dim_map[0] == 4  # 2 + 2
    assert dim_map[1] == 6  # 3 + 3
    
    # Check block-diagonal structure for charge (0, 0)
    if (0, 0) in C.data:
        block = C.data[(0, 0)]
        assert block.shape == (4, 4)
        
        # Top-left should be from A, bottom-right from B
        # Off-diagonal blocks should be zero
        np.testing.assert_allclose(block[0:2, 2:4], 0, atol=1e-10)
        np.testing.assert_allclose(block[2:4, 0:2], 0, atol=1e-10)


# ============================================================================
# Selective axes tests
# ============================================================================

def test_oplus_single_axis_by_int():
    """Test merging only axis 0 using integer specification."""
    group = U1Group()
    
    # Create tensors where axis 1 matches exactly
    # For charge conservation: OUT - IN = 0, so we need matching charges
    idx_A0 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_B0 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_match = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 5)))
    
    A = Tensor.random([idx_A0, idx_match], seed=100, itags=['i', 'j'])
    B = Tensor.random([idx_B0, idx_match], seed=200, itags=['i', 'j'])
    
    C = oplus(A, B, axes=[0])
    
    # Index 0 should be merged
    dim_map_0 = C.indices[0].sector_dim_map()
    assert dim_map_0[0] == 3  # 2 + 1
    assert dim_map_0[1] == 5  # 3 + 2
    
    # Index 1 should be unchanged
    dim_map_1 = C.indices[1].sector_dim_map()
    assert dim_map_1[0] == 3
    assert dim_map_1[1] == 5
    assert len(dim_map_1) == 2


def test_oplus_single_axis_by_itag():
    """Test merging only axis using itag specification."""
    group = U1Group()
    
    idx_A0 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_B0 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_match = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 5)))
    
    A = Tensor.random([idx_A0, idx_match], seed=110, itags=['i', 'j'])
    B = Tensor.random([idx_B0, idx_match], seed=210, itags=['i', 'j'])
    
    C = oplus(A, B, axes=['i'])
    
    # Same result as test_oplus_single_axis_by_int
    dim_map_0 = C.indices[0].sector_dim_map()
    assert dim_map_0[0] == 3
    assert dim_map_0[1] == 5


def test_oplus_multiple_axes():
    """Test merging multiple non-contiguous axes [0, 2] with axis 1 matching."""
    group = U1Group()
    
    # For charge conservation with 3 indices: OUT + OUT - IN = 0
    # So charge_0 + charge_1 - charge_2 = 0, which means charge_2 = charge_0 + charge_1
    idx_match = Index(Direction.OUT, group, sectors=(Sector(0, 4),))
    
    # A: OUT(0) + OUT(0) - IN(0) = 0 ✓
    idx_A0 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_A2 = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    
    # B: OUT(0) + OUT(0) - IN(0) = 0 ✓
    idx_B0 = Index(Direction.OUT, group, sectors=(Sector(0, 1),))
    idx_B2 = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx_A0, idx_match, idx_A2], seed=120, itags=['i', 'j', 'k'])
    B = Tensor.random([idx_B0, idx_match, idx_B2], seed=220, itags=['i', 'j', 'k'])
    
    C = oplus(A, B, axes=[0, 2])
    
    # Axes 0 and 2 should be merged
    dim_map_0 = C.indices[0].sector_dim_map()
    assert 0 in dim_map_0
    assert dim_map_0[0] == 3  # 2 + 1
    
    dim_map_2 = C.indices[2].sector_dim_map()
    assert 0 in dim_map_2
    assert dim_map_2[0] == 5  # 3 + 2
    
    # Axis 1 should be unchanged
    dim_map_1 = C.indices[1].sector_dim_map()
    assert dim_map_1[0] == 4
    assert len(dim_map_1) == 1


def test_oplus_last_axis_only():
    """Test merging only the last axis."""
    group = U1Group()
    
    # Non-merged axis must match exactly between A and B
    idx_match = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))
    
    # A: OUT(0) - IN(0) = 0 ✓ and OUT(1) - IN(1) = 0 ✓
    idx_A1 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    # B: OUT(0) - IN(0) = 0 ✓ and OUT(1) - IN(1) = 0 ✓
    idx_B1 = Index(Direction.IN, group, sectors=(Sector(0, 4), Sector(1, 5)))
    
    A = Tensor.random([idx_match, idx_A1], seed=130, itags=['i', 'j'])
    B = Tensor.random([idx_match, idx_B1], seed=230, itags=['i', 'j'])
    
    C = oplus(A, B, axes=[1])
    
    # Axis 0 unchanged
    dim_map_0 = C.indices[0].sector_dim_map()
    assert dim_map_0[0] == 3
    assert dim_map_0[1] == 2
    
    # Axis 1 merged
    dim_map_1 = C.indices[1].sector_dim_map()
    assert dim_map_1[0] == 6  # 2 + 4
    assert dim_map_1[1] == 8  # 3 + 5


# ============================================================================
# Validation tests
# ============================================================================

def test_oplus_incompatible_num_indices():
    """Test that error is raised for different number of indices."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=['a', 'b'])
    B = Tensor.random([idx, idx.flip(), idx], seed=2, itags=['a', 'b', 'c'])
    
    with pytest.raises(ValueError, match="same number of indices"):
        oplus(A, B)


def test_oplus_incompatible_groups():
    """Test error for mismatched symmetry groups on merged axes."""
    u1_group = U1Group()
    z2_group = Z2Group()
    
    idx_u1 = Index(Direction.OUT, u1_group, sectors=(Sector(0, 2),))
    idx_z2 = Index(Direction.OUT, z2_group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx_u1, idx_u1.flip()], seed=1, itags=['a', 'b'])
    B = Tensor.random([idx_z2, idx_z2.flip()], seed=2, itags=['a', 'b'])
    
    with pytest.raises(ValueError, match="same symmetry group"):
        oplus(A, B)


def test_oplus_incompatible_directions():
    """Test error for mismatched directions on merged axes."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_in = Index(Direction.IN, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx_out, idx_out.flip()], seed=1, itags=['a', 'b'])
    B = Tensor.random([idx_in, idx_in.flip()], seed=2, itags=['a', 'b'])
    
    with pytest.raises(ValueError, match="same direction"):
        oplus(A, B)


def test_oplus_non_merged_axes_mismatch():
    """Test error if non-merged axes don't match exactly."""
    group = U1Group()
    
    idx_A0 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_B0 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Different sectors on axis 1 (non-merged)
    idx_A1 = Index(Direction.IN, group, sectors=(Sector(0, 5),))
    idx_B1 = Index(Direction.IN, group, sectors=(Sector(1, 5),))  # Different charge!
    
    A = Tensor.random([idx_A0, idx_A1], seed=1, itags=['i', 'j'])
    B = Tensor.random([idx_B0, idx_B1], seed=2, itags=['i', 'j'])
    
    with pytest.raises(ValueError, match="identical charge sectors"):
        oplus(A, B, axes=[0])


def test_oplus_non_merged_axes_dimension_mismatch():
    """Test error if non-merged axes have same charge but different dimensions."""
    group = U1Group()
    
    # Both tensors need charge 0 on axis 0 and axis 1 for charge conservation
    idx_A0 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_B0 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    # Same charges but different dimensions on non-merged axis
    idx_A1 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(1, 4)))
    idx_B1 = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(1, 4)))  # Different dimension for charge 0!
    
    A = Tensor.random([idx_A0, idx_A1], seed=1, itags=['i', 'j'])
    B = Tensor.random([idx_B0, idx_B1], seed=2, itags=['i', 'j'])
    
    with pytest.raises(ValueError, match="identical dimensions"):
        oplus(A, B, axes=[0])


def test_oplus_invalid_axes_int():
    """Test error for out-of-bounds integer axes."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=['a', 'b'])
    B = Tensor.random([idx, idx.flip()], seed=2, itags=['a', 'b'])
    
    with pytest.raises(ValueError, match="Invalid axis"):
        oplus(A, B, axes=[0, 5])


def test_oplus_invalid_axes_itag():
    """Test error for non-existent itag."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1, itags=['a', 'b'])
    B = Tensor.random([idx, idx.flip()], seed=2, itags=['a', 'b'])
    
    with pytest.raises(ValueError, match="not found"):
        oplus(A, B, axes=['c'])


def test_oplus_scalar_error():
    """Test that scalars raise an error."""
    s1 = Tensor.from_scalar(3.14)
    s2 = Tensor.from_scalar(2.71)
    
    with pytest.raises(ValueError, match="scalar"):
        oplus(s1, s2)


# ============================================================================
# Multi-index tests
# ============================================================================

def test_oplus_two_indices():
    """Test basic 2-index tensor case."""
    group = U1Group()
    
    idx_A = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_B = Index(Direction.OUT, group, sectors=(Sector(1, 3),))
    
    A = Tensor.random([idx_A, idx_A.flip()], seed=300, itags=['a', 'b'])
    B = Tensor.random([idx_B, idx_B.flip()], seed=400, itags=['a', 'b'])
    
    C = oplus(A, B)
    
    assert len(C.indices) == 2
    assert C.indices[0].dim == 5  # 2 + 3
    assert C.indices[1].dim == 5  # 2 + 3


def test_oplus_three_indices():
    """Test that oplus works for 3-index tensors."""
    group = U1Group()
    
    # For charge conservation with 3 indices (OUT, IN, OUT): charge_0 - charge_1 + charge_2 = 0
    # Using charge 0 for all: 0 - 0 + 0 = 0 ✓
    idx_A = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_B = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx_A, idx_A.flip(), idx_A], seed=310, itags=['a', 'b', 'c'])
    B = Tensor.random([idx_B, idx_B.flip(), idx_B], seed=410, itags=['a', 'b', 'c'])
    
    C = oplus(A, B)
    
    assert len(C.indices) == 3
    for idx in C.indices:
        assert idx.dim == 4  # 2 + 2


def test_oplus_four_indices_partial():
    """Test merging 2 of 4 axes."""
    group = U1Group()
    
    # For 4 indices (OUT, IN, OUT, OUT): charge_0 - charge_1 + charge_2 + charge_3 = 0
    # Using all 0: 0 - 0 + 0 + 0 = 0 ✓
    idx_A0 = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_B0 = Index(Direction.OUT, group, sectors=(Sector(0, 3),))
    
    idx_match1 = Index(Direction.IN, group, sectors=(Sector(0, 3),))
    idx_match3 = Index(Direction.OUT, group, sectors=(Sector(0, 4),))
    
    A = Tensor.random([idx_A0, idx_match1, idx_A0, idx_match3], 
                      seed=320, itags=['a', 'b', 'c', 'd'])
    B = Tensor.random([idx_B0, idx_match1, idx_B0, idx_match3], 
                      seed=420, itags=['a', 'b', 'c', 'd'])
    
    C = oplus(A, B, axes=[0, 2])
    
    # Merged axes should have charge 0
    assert 0 in C.indices[0].sector_dim_map()
    assert C.indices[0].sector_dim_map()[0] == 5  # 2 + 3
    assert 0 in C.indices[2].sector_dim_map()
    assert C.indices[2].sector_dim_map()[0] == 5  # 2 + 3
    
    # Non-merged axes unchanged
    assert C.indices[1].sector_dim_map()[0] == 3
    assert C.indices[3].sector_dim_map()[0] == 4


# ============================================================================
# Symmetry tests
# ============================================================================

def test_oplus_z2_symmetry():
    """Test oplus with Z2 symmetry."""
    group = Z2Group()
    
    idx_A = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_B = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))
    
    A = Tensor.random([idx_A, idx_A.flip()], seed=500, itags=['a', 'b'])
    B = Tensor.random([idx_B, idx_B.flip()], seed=600, itags=['a', 'b'])
    
    C = oplus(A, B)
    
    dim_map = C.indices[0].sector_dim_map()
    assert dim_map[0] == 3  # 2 + 1
    assert dim_map[1] == 5  # 3 + 2


def test_oplus_product_group():
    """Test oplus with ProductGroup (U1×U1)."""
    group = ProductGroup((U1Group(), U1Group()))
    
    # Sectors with tuple charges
    idx_A = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 2),
        Sector((1, 0), 3)
    ))
    idx_B = Index(Direction.OUT, group, sectors=(
        Sector((0, 0), 1),
        Sector((0, 1), 2)
    ))
    
    A = Tensor.random([idx_A, idx_A.flip()], seed=700, itags=['a', 'b'])
    B = Tensor.random([idx_B, idx_B.flip()], seed=800, itags=['a', 'b'])
    
    C = oplus(A, B)
    
    dim_map = C.indices[0].sector_dim_map()
    assert dim_map[(0, 0)] == 3  # 2 + 1
    assert dim_map[(1, 0)] == 3  # 3 + 0
    assert dim_map[(0, 1)] == 2  # 0 + 2


# ============================================================================
# Property tests
# ============================================================================

def test_oplus_dimension_sum_merged_axes():
    """Verify dimensions sum correctly on merged axes."""
    group = U1Group()
    
    idx_A = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_B = Index(Direction.OUT, group, sectors=(Sector(0, 5), Sector(2, 1)))
    
    A = Tensor.random([idx_A, idx_A.flip()], seed=900, itags=['a', 'b'])
    B = Tensor.random([idx_B, idx_B.flip()], seed=1000, itags=['a', 'b'])
    
    C = oplus(A, B)
    
    # Total dimension should be sum of individual dimensions
    assert C.indices[0].dim == A.indices[0].dim + B.indices[0].dim
    assert C.indices[1].dim == A.indices[1].dim + B.indices[1].dim


def test_oplus_dimension_unchanged_non_merged():
    """Verify dimensions unchanged on non-merged axes."""
    group = U1Group()
    
    # Both tensors need the same charges on non-merged axis
    idx_A0 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_B0 = Index(Direction.OUT, group, sectors=(Sector(0, 4), Sector(1, 5)))
    idx_match = Index(Direction.IN, group, sectors=(Sector(0, 7), Sector(1, 6)))
    
    A = Tensor.random([idx_A0, idx_match], seed=910, itags=['a', 'b'])
    B = Tensor.random([idx_B0, idx_match], seed=1010, itags=['a', 'b'])
    
    C = oplus(A, B, axes=[0])
    
    # Non-merged axis should be unchanged
    assert C.indices[1].dim == 13  # 7 + 6
    assert C.indices[1].dim == A.indices[1].dim
    assert C.indices[1].dim == B.indices[1].dim


def test_oplus_charge_conservation():
    """Verify all output blocks satisfy charge conservation."""
    group = U1Group()
    
    idx_A = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_B = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
    
    A = Tensor.random([idx_A, idx_A.flip()], seed=920, itags=['a', 'b'])
    B = Tensor.random([idx_B, idx_B.flip()], seed=1020, itags=['a', 'b'])
    
    C = oplus(A, B)
    
    # All blocks should satisfy charge conservation
    assert_charge_neutral(C)


def test_oplus_orthogonality():
    """Verify blocks from A and B don't interfere (block-diagonal on merged axes)."""
    group = U1Group()
    
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    # Create tensors with known values
    A = Tensor.zeros([idx, idx.flip()], itags=['a', 'b'])
    A.data[(0, 0)] = np.ones((2, 2))
    
    B = Tensor.zeros([idx, idx.flip()], itags=['a', 'b'])
    B.data[(0, 0)] = 2 * np.ones((2, 2))
    
    C = oplus(A, B)
    
    # Check block-diagonal structure
    block = C.data[(0, 0)]
    assert block.shape == (4, 4)
    
    # Top-left should be all ones (from A)
    np.testing.assert_array_equal(block[0:2, 0:2], 1)
    
    # Bottom-right should be all twos (from B)
    np.testing.assert_array_equal(block[2:4, 2:4], 2)
    
    # Off-diagonal should be zero
    np.testing.assert_array_equal(block[0:2, 2:4], 0)
    np.testing.assert_array_equal(block[2:4, 0:2], 0)


# ============================================================================
# Integration tests
# ============================================================================

def test_oplus_dtype_promotion():
    """Verify dtype handling (int + float → float)."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1100, itags=['a', 'b'], dtype=np.float32)
    B = Tensor.random([idx, idx.flip()], seed=1200, itags=['a', 'b'], dtype=np.float64)
    
    C = oplus(A, B)
    
    # Should promote to float64
    assert C.dtype == np.float64


def test_oplus_preserves_itags():
    """Verify itags are preserved correctly."""
    group = U1Group()
    idx_A = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_B = Index(Direction.OUT, group, sectors=(Sector(1, 3),))
    
    A = Tensor.random([idx_A, idx_A.flip()], seed=1110, itags=['x', 'y'])
    B = Tensor.random([idx_B, idx_B.flip()], seed=1210, itags=['x', 'y'])
    
    C = oplus(A, B)
    
    assert C.itags == ('x', 'y')


def test_oplus_then_contract():
    """Test combining oplus with contraction."""
    group = U1Group()
    
    idx_A = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    idx_B = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    A = Tensor.random([idx_A, idx_A.flip()], seed=1120, itags=['a', 'b'])
    B = Tensor.random([idx_B, idx_B.flip()], seed=1220, itags=['a', 'b'])
    
    C = oplus(A, B)
    
    # Now contract with another tensor
    # Create another tensor to contract with - matching itag and opposite direction
    # C.indices[1] is IN, so D.indices[0] should be OUT (flip it)
    D = Tensor.random([C.indices[1].flip(), C.indices[0]], seed=1300, itags=['b', 'c'])
    
    result = contract(C, D, axes=(1, 0))
    
    # Should have 2 indices remaining
    assert len(result.indices) == 2


def test_oplus_preserves_label():
    """Test that label from first tensor is preserved."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    A = Tensor.random([idx, idx.flip()], seed=1130, itags=['a', 'b'])
    A.label = "TensorA"
    B = Tensor.random([idx, idx.flip()], seed=1230, itags=['a', 'b'])
    B.label = "TensorB"
    
    C = oplus(A, B)
    
    assert C.label == "TensorA"


def test_oplus_empty_blocks():
    """Test oplus when one tensor has no blocks for certain charges."""
    group = U1Group()
    
    idx_A = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx_B = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 2)))
    
    # Create A with only charge 0 block
    A = Tensor.zeros([idx_A, idx_A.flip()], itags=['a', 'b'])
    A.data[(0, 0)] = np.ones((2, 2))
    
    # Create B with only charge 0 block
    B = Tensor.zeros([idx_B, idx_B.flip()], itags=['a', 'b'])
    B.data[(0, 0)] = 2 * np.ones((1, 1))
    
    C = oplus(A, B)
    
    # Result should have charge (0, 0) block
    assert (0, 0) in C.data
    assert C.data[(0, 0)].shape == (3, 3)  # 2+1 x 2+1
