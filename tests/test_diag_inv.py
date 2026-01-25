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


"""Tests for diag and inv functions."""

import numpy as np
import pytest

from nicole import Direction, Index, Sector, Tensor, U1Group, Z2Group, diag, inv
from nicole.symmetry.product import ProductGroup
from nicole.decomp import svd, eig


# =============================================================================
#  Tests for diag function
# =============================================================================

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


# =============================================================================
#  Tests for inv function
# =============================================================================

def test_inv_basic_u1():
    """Test basic inv functionality with U1Group."""
    from nicole import contract
    
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(-1, 2), Sector(0, 2), Sector(1, 2), Sector(2, 1)))
    
    # Create a diagonal tensor
    D = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("i", "j"),
        data={
            (-1, -1): np.diag([1.0, 2.0]),
            (0, 0): np.diag([2.0, 4.0]),
            (1, 1): np.diag([5.0, 10.0]),
            (2, 2): np.array([[3.0]])
        },
        label="Diagonal"
    )
    
    # Invert
    D_inv = inv(D)
    
    # Check basic properties
    assert len(D_inv.indices) == 2
    assert D_inv.label == "Diagonal"
    assert D_inv.itags == ("j", "i")  # Swapped (transposed)
    assert D_inv.indices[0].direction == Direction.OUT  # Flipped from IN
    assert D_inv.indices[1].direction == Direction.IN  # Flipped from OUT
    
    # Check inverted values (keys remain same for diagonal blocks)
    np.testing.assert_allclose(D_inv.data[(-1, -1)], np.diag([1.0, 0.5]))
    np.testing.assert_allclose(D_inv.data[(0, 0)], np.diag([0.5, 0.25]))
    np.testing.assert_allclose(D_inv.data[(1, 1)], np.diag([0.2, 0.1]))
    np.testing.assert_allclose(D_inv.data[(2, 2)], np.array([[1.0/3.0]]))
    
    # Verify D * D_inv = I (element-wise check)
    for key in D.data.keys():
        product = np.diag(D.data[key]) * np.diag(D_inv.data[key])
        np.testing.assert_allclose(product, 1.0)
    
    # Verify D @ D_inv = I (tensor contraction check)
    result = contract(D, D_inv, axes=(1, 0))
    for key, block in result.data.items():
        assert block.ndim == 2
        assert block.shape[0] == block.shape[1]
        np.testing.assert_allclose(block, np.eye(block.shape[0]), atol=1e-14)


def test_inv_with_diag_output():
    """Test inv works with output from diag function."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 3), Sector(1, 2)))
    
    S_blocks = {
        (0, 0): np.array([3.0, 2.0, 1.0]),
        (1, 1): np.array([0.5, 0.3])
    }
    
    S_diag = diag(S_blocks, bond_index)
    S_inv = inv(S_diag)
    
    # Check it's labeled Diagonal
    assert S_diag.label == "Diagonal"
    assert S_inv.label == "Diagonal"
    
    # Verify inversion
    expected_00 = np.diag([1/3.0, 1/2.0, 1/1.0])
    expected_11 = np.diag([1/0.5, 1/0.3])
    
    np.testing.assert_allclose(S_inv.data[(0, 0)], expected_00)
    np.testing.assert_allclose(S_inv.data[(1, 1)], expected_11)


def test_inv_with_svd():
    """Test inv integration with SVD output."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, (Sector(0, 3), Sector(1, 2)))
    idx_j = Index(Direction.IN, group, (Sector(0, 3), Sector(1, 2)))
    
    T = Tensor.random([idx_i, idx_j], itags=["i", "j"], seed=42)
    
    # Perform SVD
    U, S_blocks, Vh = svd(T, axis=0)
    
    # Create diagonal matrix and invert
    bond_index = U.indices[1]
    S_diag = diag(S_blocks, bond_index)
    S_inv = inv(S_diag)
    
    # Verify all blocks give identity when multiplied
    for key in S_diag.data.keys():
        product = np.diag(S_diag.data[key]) * np.diag(S_inv.data[key])
        np.testing.assert_allclose(product, 1.0, rtol=1e-14)


def test_inv_z2_group():
    """Test inv with Z2Group."""
    group = Z2Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2), Sector(1, 3)))
    
    D = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("i", "j"),
        data={
            (0, 0): np.diag([2.0, 4.0]),
            (1, 1): np.diag([1.0, 0.5, 0.25])
        },
        label="Diagonal"
    )
    
    D_inv = inv(D)
    
    np.testing.assert_allclose(D_inv.data[(0, 0)], np.diag([0.5, 0.25]))
    np.testing.assert_allclose(D_inv.data[(1, 1)], np.diag([1.0, 2.0, 4.0]))


def test_inv_product_group():
    """Test inv with ProductGroup."""
    group = ProductGroup([U1Group(), Z2Group()])
    bond_index = Index(
        Direction.IN,
        group,
        (Sector((0, 0), 2), Sector((1, 1), 3))
    )
    
    D = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("i", "j"),
        data={
            ((0, 0), (0, 0)): np.diag([2.0, 5.0]),
            ((1, 1), (1, 1)): np.diag([1.0, 0.5, 0.2])
        },
        label="Diagonal"
    )
    
    D_inv = inv(D)
    
    # Check charges are preserved
    assert set(D_inv.data.keys()) == {((0, 0), (0, 0)), ((1, 1), (1, 1))}
    
    # Check inversions
    np.testing.assert_allclose(D_inv.data[((0, 0), (0, 0))], np.diag([0.5, 0.2]))
    np.testing.assert_allclose(D_inv.data[((1, 1), (1, 1))], np.diag([1.0, 2.0, 5.0]))


def test_inv_single_element():
    """Test inv with single-element diagonal blocks."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 1), Sector(1, 1)))
    
    D = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("i", "j"),
        data={
            (0, 0): np.array([[5.0]]),
            (1, 1): np.array([[2.0]])
        },
        label="Diagonal"
    )
    
    D_inv = inv(D)
    
    np.testing.assert_allclose(D_inv.data[(0, 0)], np.array([[0.2]]))
    np.testing.assert_allclose(D_inv.data[(1, 1)], np.array([[0.5]]))


def test_inv_negative_charges():
    """Test inv with negative charges."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(-1, 2), Sector(0, 2), Sector(1, 1)))
    
    D = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("i", "j"),
        data={
            (-1, -1): np.diag([2.0, 4.0]),
            (0, 0): np.diag([1.0, 0.5]),
            (1, 1): np.diag([10.0])
        },
        label="Diagonal"
    )
    
    D_inv = inv(D)
    
    np.testing.assert_allclose(D_inv.data[(-1, -1)], np.diag([0.5, 0.25]))
    np.testing.assert_allclose(D_inv.data[(0, 0)], np.diag([1.0, 2.0]))
    np.testing.assert_allclose(D_inv.data[(1, 1)], np.diag([0.1]))


def test_inv_complex_dtype():
    """Test inv with complex diagonal values."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2),))
    
    # Complex diagonal (like phase factors)
    D = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("i", "j"),
        data={(0, 0): np.diag([1.0 + 1.0j, 2.0 + 0.0j])},
        label="Diagonal"
    )
    
    D_inv = inv(D)
    
    # Verify inversion
    expected = np.diag([1/(1.0 + 1.0j), 1/(2.0 + 0.0j)])
    np.testing.assert_allclose(D_inv.data[(0, 0)], expected)
    
    # Verify D * D_inv = I
    product_diag = np.diag(D.data[(0, 0)]) * np.diag(D_inv.data[(0, 0)])
    np.testing.assert_allclose(product_diag, 1.0)


def test_inv_non_diagonal_without_label():
    """Test inv raises error for non-diagonal tensor without 'Diagonal' label."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2),))
    
    # Non-diagonal matrix
    T = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("i", "j"),
        data={(0, 0): np.array([[1.0, 0.5], [0.0, 2.0]])},  # Upper triangular
        label="Tensor"  # Not labeled "Diagonal"
    )
    
    with pytest.raises(ValueError, match="non-zero off-diagonal elements"):
        inv(T)


def test_inv_non_diagonal_skipped_with_label():
    """Test inv skips diagonal check when labeled 'Diagonal'."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2),))
    
    # Non-diagonal but labeled "Diagonal" (user responsibility)
    T = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("i", "j"),
        data={(0, 0): np.array([[1.0, 0.0], [0.0, 2.0]])},  # Actually diagonal
        label="Diagonal"
    )
    
    # Should not raise error
    T_inv = inv(T)
    assert T_inv.label == "Diagonal"


def test_inv_error_not_two_indices():
    """Test inv raises error if tensor doesn't have exactly 2 indices."""
    group = U1Group()
    idx = Index(Direction.OUT, group, (Sector(0, 2),))
    
    # 3-index tensor
    T = Tensor.random([idx, idx.flip(), idx], itags=["i", "j", "k"], seed=42)
    
    with pytest.raises(ValueError, match="exactly 2 indices"):
        inv(T)


def test_inv_same_direction_flips():
    """Test inv automatically flips indices when they have same direction."""
    group = U1Group()
    idx = Index(Direction.OUT, group, (Sector(0, 2),))
    
    # Both indices have same direction (both OUT)
    T = Tensor(
        indices=(idx, idx),  # Both OUT
        itags=("i", "j"),
        data={(0, 0): np.diag([2.0, 4.0])},
        label="Diagonal"
    )
    
    # Should automatically flip both indices
    T_inv = inv(T)
    
    # Check that directions were flipped to opposite
    assert T_inv.indices[0].direction == Direction.IN
    assert T_inv.indices[1].direction == Direction.IN
    
    # Verify inversion is correct
    np.testing.assert_allclose(T_inv.data[(0, 0)], np.diag([0.5, 0.25]))


def test_inv_same_direction_in_gives_identity():
    """Test inv(D) @ D = I when both indices are IN."""
    from nicole import contract
    
    group = U1Group()
    idx = Index(Direction.IN, group, (Sector(-1, 2), Sector(0, 2), Sector(1, 1)))
    
    # Create diagonal tensor with both IN (requires flip to construct)
    D = Tensor(
        indices=(idx.flip(), idx),
        itags=("i", "j"),
        data={
            (-1, -1): np.diag([1.5, 2.5]),
            (0, 0): np.diag([2.0, 4.0]),
            (1, 1): np.array([[3.0]])
        },
        label="Diagonal"
    )
    # Flip to make both IN
    D.flip(0)
    
    # Verify both indices are IN
    assert D.indices[0].direction == Direction.IN
    assert D.indices[1].direction == Direction.IN
    
    # Invert - should flip to opposite directions
    D_inv = inv(D)
    
    # Verify inverted has opposite directions
    assert D_inv.indices[0].direction == Direction.OUT
    assert D_inv.indices[1].direction == Direction.OUT
    
    # Contract D_inv with D should give identity
    # D_inv has itags ('j', 'i') with directions (OUT, OUT) - transposed from same-dir input
    # D has itags ('i', 'j') with directions (IN, IN)
    # Contract axis 1 of D_inv ('i', OUT) with axis 0 of D ('i', IN)
    result = contract(D_inv, D, axes=(1, 0))
    
    # Result should be identity matrix for each sector
    # Verify all blocks are identity matrices
    for key, block in result.data.items():
        assert block.ndim == 2
        assert block.shape[0] == block.shape[1]
        np.testing.assert_allclose(block, np.eye(block.shape[0]), atol=1e-14)


def test_inv_same_direction_out_gives_identity():
    """Test D @ inv(D) = I when both indices are OUT."""
    from nicole import contract
    
    group = U1Group()
    idx = Index(Direction.IN, group, (Sector(-2, 1), Sector(-1, 2), Sector(0, 2), Sector(1, 1)))
    
    # Create diagonal tensor with both OUT (requires valid construction)
    # Start with valid opposite directions, then flip one
    D = Tensor(
        indices=(idx.flip(), idx),
        itags=("i", "j"),
        data={
            (-2, -2): np.array([[5.0]]),
            (-1, -1): np.diag([1.5, 2.5]),
            (0, 0): np.diag([2.0, 4.0]),
            (1, 1): np.array([[3.0]])
        },
        label="Diagonal"
    )
    # Flip second index to make both OUT
    D.flip(1)
    
    # Verify both indices are OUT
    assert D.indices[0].direction == Direction.OUT
    assert D.indices[1].direction == Direction.OUT
    
    # Invert - should flip to opposite directions
    D_inv = inv(D)
    
    # Verify inverted has opposite directions
    assert D_inv.indices[0].direction == Direction.IN
    assert D_inv.indices[1].direction == Direction.IN
    
    # Contract D with D_inv should give identity
    # D has itags ('i', 'j') with directions (OUT, OUT)
    # D_inv has itags ('j', 'i') with directions (IN, IN) - transposed from same-dir input
    # Contract axis 1 of D ('j', OUT) with axis 0 of D_inv ('j', IN)
    result = contract(D, D_inv, axes=(1, 0))
    
    # Result should be identity matrix for each sector
    # Verify all blocks are identity matrices
    for key, block in result.data.items():
        assert block.ndim == 2
        assert block.shape[0] == block.shape[1]
        np.testing.assert_allclose(block, np.eye(block.shape[0]), atol=1e-14)


def test_inv_error_zero_element():
    """Test inv raises error when diagonal contains zero."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 3),))
    
    D = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("i", "j"),
        data={(0, 0): np.diag([1.0, 0.0, 3.0])},  # Zero in the middle
        label="Diagonal"
    )
    
    with pytest.raises(ZeroDivisionError, match="zero elements.*diagonal positions \\[1\\]"):
        inv(D)


def test_inv_error_non_square_block():
    """Test inv raises error for non-square blocks when not labeled Diagonal."""
    group = U1Group()
    idx_i = Index(Direction.OUT, group, (Sector(0, 2),))
    idx_j = Index(Direction.IN, group, (Sector(0, 3),))
    
    T = Tensor(
        indices=(idx_i, idx_j),
        itags=("i", "j"),
        data={(0, 0): np.zeros((2, 3))},
        label="Tensor"
    )
    
    with pytest.raises(ValueError, match="square matrix blocks"):
        inv(T)


def test_inv_preserves_structure():
    """Test inv preserves tensor structure (itags, dtype, label)."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2), Sector(1, 3)))
    
    D = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("left", "right"),
        data={
            (0, 0): np.diag([2.0, 4.0]),
            (1, 1): np.diag([1.0, 0.5, 0.25])
        },
        dtype=np.float64,
        label="Diagonal"
    )
    
    D_inv = inv(D)
    
    # Check structure preservation (always transposed)
    assert D_inv.itags == (D.itags[1], D.itags[0])  # Swapped
    assert D_inv.dtype == D.dtype
    assert D_inv.label == D.label
    assert len(D_inv.indices) == len(D.indices)
    # Indices are swapped and flipped
    assert D_inv.indices[0].direction == D.indices[1].direction.reverse()
    assert D_inv.indices[1].direction == D.indices[0].direction.reverse()


def test_inv_double_inversion():
    """Test that inverting twice gives back the original (up to numerical error)."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2), Sector(1, 2)))
    
    D = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("i", "j"),
        data={
            (0, 0): np.diag([2.0, 5.0]),
            (1, 1): np.diag([0.5, 10.0])
        },
        label="Diagonal"
    )
    
    D_inv = inv(D)
    D_inv_inv = inv(D_inv)
    
    # Should recover original
    for key in D.data.keys():
        np.testing.assert_allclose(D_inv_inv.data[key], D.data[key], rtol=1e-14)


def test_inv_very_small_values():
    """Test inv raises error for values below machine epsilon."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2),))
    
    # Value below machine epsilon (treated as zero)
    eps = np.finfo(np.float64).eps
    tiny_val = eps / 10  # Below threshold
    D = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("i", "j"),
        data={(0, 0): np.diag([tiny_val, 1.0])},
        label="Diagonal"
    )
    
    # Should raise error because tiny_val < eps
    with pytest.raises(ZeroDivisionError, match="zero elements.*diagonal positions \\[0\\]"):
        inv(D)


def test_inv_small_but_invertible():
    """Test inv works with small but invertible values (above machine epsilon)."""
    group = U1Group()
    bond_index = Index(Direction.IN, group, (Sector(0, 2),))
    
    # Small but above machine epsilon
    eps = np.finfo(np.float64).eps
    small_val = eps * 100  # Well above threshold
    D = Tensor(
        indices=(bond_index.flip(), bond_index),
        itags=("i", "j"),
        data={(0, 0): np.diag([small_val, 1.0])},
        label="Diagonal"
    )
    
    D_inv = inv(D)
    
    # Inversion should work
    expected = np.diag([1/small_val, 1.0])
    np.testing.assert_allclose(D_inv.data[(0, 0)], expected)
