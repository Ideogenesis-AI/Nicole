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


"""Tests for diag function."""

import numpy as np
import pytest

from nicole import Direction, Index, Sector, Tensor, U1Group, Z2Group, diag
from nicole.symmetry.product import ProductGroup
from nicole.decomp import svd, eig


def test_diag_basic_u1():
    """Test basic diag functionality with U1Group."""
    group = U1Group()
    # Create bond index with IN direction (as returned by SVD)
    bond_index = Index(Direction.IN, group, (Sector(0, 3), Sector(1, 2)))
    
    # Create singular value blocks
    S_blocks = {
        (0, 0): np.array([3.0, 2.0, 1.0]),
        (1, 1): np.array([0.5, 0.3])
    }
    
    S_diag = diag(S_blocks, bond_index)
    
    # Check basic properties
    assert len(S_diag.indices) == 2
    assert S_diag.label == "Diagonal"
    assert S_diag.itags == ("_bond_L", "_bond_R")
    
    # Check index structure (diag creates (bond_index.flip(), bond_index))
    assert S_diag.indices[0].direction == Direction.OUT
    assert S_diag.indices[1].direction == Direction.IN
    assert S_diag.indices[0].group == group
    assert S_diag.indices[1].group == group
    
    # Check data blocks
    assert set(S_diag.data.keys()) == {(0, 0), (1, 1)}
    
    # Verify block (0, 0) is diagonal
    block_00 = S_diag.data[(0, 0)]
    assert block_00.shape == (3, 3)
    expected_00 = np.diag([3.0, 2.0, 1.0])
    np.testing.assert_allclose(block_00, expected_00)
    assert np.allclose(block_00, np.diag(np.diag(block_00)))
    
    # Verify block (1, 1) is diagonal
    block_11 = S_diag.data[(1, 1)]
    assert block_11.shape == (2, 2)
    expected_11 = np.diag([0.5, 0.3])
    np.testing.assert_allclose(block_11, expected_11)
    assert np.allclose(block_11, np.diag(np.diag(block_11)))


def test_diag_custom_itags():
    """Test diag with custom itags."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2),))
    
    S_blocks = {(0, 0): np.array([1.0, 0.5])}
    
    S_diag = diag(S_blocks, bond_index, itags=("left", "right"))
    
    assert S_diag.itags == ("left", "right")
    assert S_diag.label == "Diagonal"


def test_diag_custom_dtype():
    """Test diag with custom dtype."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2),))
    
    S_blocks = {(0, 0): np.array([1.0, 0.5])}
    
    S_diag = diag(S_blocks, bond_index, dtype=np.float32)
    
    assert S_diag.dtype == np.float32


def test_diag_z2_group():
    """Test diag with Z2Group."""
    group = Z2Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2), Sector(1, 3)))
    
    S_blocks = {
        (0, 0): np.array([2.0, 1.5]),
        (1, 1): np.array([1.0, 0.8, 0.3])
    }
    
    S_diag = diag(S_blocks, bond_index)
    
    assert len(S_diag.indices) == 2
    assert S_diag.indices[0].group == group
    assert S_diag.indices[1].group == group
    
    # Check diagonal structure
    assert np.allclose(S_diag.data[(0, 0)], np.diag([2.0, 1.5]))
    assert np.allclose(S_diag.data[(1, 1)], np.diag([1.0, 0.8, 0.3]))


def test_diag_product_group():
    """Test diag with ProductGroup."""
    group = ProductGroup([U1Group(), Z2Group()])
    bond_index = Index(
        Direction.IN,
        group,
        (Sector((0, 0), 2), Sector((1, 1), 3), Sector((-1, 0), 1))
    )
    
    # Keys are tuples of tuples for ProductGroup
    S_blocks = {
        ((0, 0), (0, 0)): np.array([2.5, 1.2]),
        ((1, 1), (1, 1)): np.array([1.5, 0.8, 0.4]),
        ((-1, 0), (-1, 0)): np.array([0.9])
    }
    
    S_diag = diag(S_blocks, bond_index)
    
    assert len(S_diag.indices) == 2
    assert S_diag.indices[0].group == group
    assert S_diag.indices[1].group == group
    
    # Check all blocks are present
    assert set(S_diag.data.keys()) == {((0, 0), (0, 0)), ((1, 1), (1, 1)), ((-1, 0), (-1, 0))}
    
    # Verify diagonal structure for each block
    np.testing.assert_allclose(S_diag.data[((0, 0), (0, 0))], np.diag([2.5, 1.2]))
    np.testing.assert_allclose(S_diag.data[((1, 1), (1, 1))], np.diag([1.5, 0.8, 0.4]))
    np.testing.assert_allclose(S_diag.data[((-1, 0), (-1, 0))], np.diag([0.9]))


def test_diag_with_svd_u1():
    """Test diag integration with SVD for U1Group."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, (Sector(0, 3), Sector(1, 2)))
    idx_j = Index(Direction.IN, group, (Sector(0, 3), Sector(1, 2)))
    
    T = Tensor.random([idx_i, idx_j], itags=["i", "j"], seed=42)
    
    # Perform SVD
    U, S_blocks, Vh = svd(T, axis=0)
    
    # Convert to diagonal matrix
    bond_index = U.indices[1]
    S_diag = diag(S_blocks, bond_index, itags=("bond_L", "bond_R"))
    
    # Check properties
    assert S_diag.label == "Diagonal"
    assert S_diag.itags == ("bond_L", "bond_R")
    assert len(S_diag.indices) == 2
    
    # Verify all blocks are diagonal
    for key, block in S_diag.data.items():
        assert block.ndim == 2
        assert block.shape[0] == block.shape[1]
        # Check it's diagonal (off-diagonal elements are zero)
        off_diag = block - np.diag(np.diag(block))
        assert np.max(np.abs(off_diag)) < 1e-14


def test_diag_with_svd_product_group():
    """Test diag integration with SVD for ProductGroup."""
    group = ProductGroup([U1Group(), Z2Group()])
    idx_i = Index(
        Direction.OUT,
        group,
        (Sector((0, 0), 2), Sector((1, 1), 2), Sector((-1, 0), 2))
    )
    idx_j = Index(
        Direction.IN,
        group,
        (Sector((0, 0), 2), Sector((1, 1), 2), Sector((-1, 0), 2))
    )
    
    T = Tensor.random([idx_i, idx_j], itags=["i", "j"], seed=123)
    
    # Perform SVD
    U, S_blocks, Vh = svd(T, axis=0)
    
    # Convert to diagonal matrix
    bond_index = U.indices[1]
    S_diag = diag(S_blocks, bond_index)
    
    # Check properties
    assert S_diag.label == "Diagonal"
    assert len(S_diag.indices) == 2
    assert S_diag.indices[0].group == group
    
    # Verify all blocks are diagonal and keys are correct format
    for key, block in S_diag.data.items():
        # Key should be tuple of tuples for ProductGroup
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert isinstance(key[0], tuple)
        assert isinstance(key[1], tuple)
        assert key[0] == key[1]  # Diagonal block
        
        # Check diagonal structure
        assert block.ndim == 2
        assert block.shape[0] == block.shape[1]
        off_diag = block - np.diag(np.diag(block))
        assert np.max(np.abs(off_diag)) < 1e-14


def test_diag_with_eig():
    """Test diag integration with eig."""
    group = U1Group()
    idx = Index(Direction.OUT, group, (Sector(0, 3), Sector(1, 2)))
    
    # Create a test tensor (eig works on any square tensor)
    T = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=42)
    
    # Perform eigendecomposition
    U, D_blocks = eig(T)
    
    # Convert eigenvalues to diagonal matrix
    bond_index = U.indices[1]
    D_diag = diag(D_blocks, bond_index, itags=("eig_L", "eig_R"))
    
    # Check properties
    assert D_diag.label == "Diagonal"
    assert D_diag.itags == ("eig_L", "eig_R")
    assert len(D_diag.indices) == 2
    
    # Verify all blocks are diagonal
    for key, block in D_diag.data.items():
        assert block.ndim == 2
        off_diag = block - np.diag(np.diag(block))
        assert np.max(np.abs(off_diag)) < 1e-14


def test_diag_empty_blocks():
    """Test diag with empty blocks dictionary."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2),))
    
    S_blocks = {}
    
    S_diag = diag(S_blocks, bond_index)
    
    assert len(S_diag.data) == 0
    assert S_diag.label == "Diagonal"


def test_diag_single_element_blocks():
    """Test diag with single-element (scalar) blocks."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 1), Sector(1, 1)))
    
    S_blocks = {
        (0, 0): np.array([5.0]),
        (1, 1): np.array([3.0])
    }
    
    S_diag = diag(S_blocks, bond_index)
    
    # Single element should become 1x1 matrix
    np.testing.assert_allclose(S_diag.data[(0, 0)], np.array([[5.0]]))
    np.testing.assert_allclose(S_diag.data[(1, 1)], np.array([[3.0]]))


def test_diag_negative_charges():
    """Test diag with negative charges."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(-2, 2), Sector(-1, 1), Sector(0, 3)))
    
    S_blocks = {
        (-2, -2): np.array([1.5, 0.8]),
        (-1, -1): np.array([2.0]),
        (0, 0): np.array([3.0, 2.5, 1.0])
    }
    
    S_diag = diag(S_blocks, bond_index)
    
    assert set(S_diag.data.keys()) == {(-2, -2), (-1, -1), (0, 0)}
    np.testing.assert_allclose(S_diag.data[(-2, -2)], np.diag([1.5, 0.8]))
    np.testing.assert_allclose(S_diag.data[(-1, -1)], np.diag([2.0]))
    np.testing.assert_allclose(S_diag.data[(0, 0)], np.diag([3.0, 2.5, 1.0]))


def test_diag_error_non_1d_blocks():
    """Test diag raises error for non-1D blocks."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2),))
    
    # Create 2D block (invalid)
    S_blocks = {
        (0, 0): np.array([[1.0, 0.5], [0.5, 1.0]])
    }
    
    with pytest.raises(ValueError, match="1-dimensional.*shape \\(2, 2\\)"):
        diag(S_blocks, bond_index)


def test_diag_error_invalid_itags():
    """Test diag raises error for invalid itags."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2),))
    S_blocks = {(0, 0): np.array([1.0, 0.5])}
    
    # Wrong number of itags
    with pytest.raises(ValueError, match="tuple of two strings"):
        diag(S_blocks, bond_index, itags=("only_one",))
    
    # Wrong type
    with pytest.raises(ValueError, match="tuple of two strings"):
        diag(S_blocks, bond_index, itags=["left", "right"])


def test_diag_complex_dtype():
    """Test diag with complex singular values."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2),))
    
    S_blocks = {(0, 0): np.array([1.0 + 0.5j, 0.5 + 0.2j])}
    
    S_diag = diag(S_blocks, bond_index)
    
    expected = np.diag([1.0 + 0.5j, 0.5 + 0.2j])
    np.testing.assert_allclose(S_diag.data[(0, 0)], expected)
    assert np.iscomplexobj(S_diag.data[(0, 0)])


def test_diag_preserves_charge_conservation():
    """Test that diag output satisfies charge conservation."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2), Sector(1, 3)))
    
    S_blocks = {
        (0, 0): np.array([2.0, 1.0]),
        (1, 1): np.array([3.0, 2.0, 1.0])
    }
    
    S_diag = diag(S_blocks, bond_index)
    
    # For a charge-neutral tensor with (OUT, IN) directions,
    # blocks must have (q, q) structure
    for key in S_diag.data.keys():
        assert len(key) == 2
        assert key[0] == key[1], f"Block {key} violates charge conservation"
    
    # Check index directions are opposite
    assert S_diag.indices[0].direction != S_diag.indices[1].direction
