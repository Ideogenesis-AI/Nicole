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


"""Tests for BlockSchema utility class."""

import math
import torch
import pytest

from nicole import Direction, Index, Sector, U1Group, Z2Group, SU2Group
from nicole.blocks import BlockSchema
import nicole.symmetry.delegate as dg


def test_iter_admissible_keys_simple():
    """Test BlockSchema.iter_admissible_keys with simple indices."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 3), Sector(-1, 2)))
    
    keys = list(BlockSchema.iter_admissible_keys([idx1, idx2]))
    
    # Cartesian product: (0,0), (0,-1), (1,0), (1,-1)
    assert set(keys) == {(0, 0), (0, -1), (1, 0), (1, -1)}


def test_iter_admissible_keys_single_index():
    """Test BlockSchema.iter_admissible_keys with single index."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    keys = list(BlockSchema.iter_admissible_keys([idx]))
    
    assert set(keys) == {(0,), (1,)}


def test_iter_admissible_keys_three_indices():
    """Test BlockSchema.iter_admissible_keys with three indices."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))
    
    keys = list(BlockSchema.iter_admissible_keys([idx1, idx2, idx3]))
    
    # All combinations: 2^3 = 8
    assert len(keys) == 8


def test_iter_admissible_keys_empty():
    """Test BlockSchema.iter_admissible_keys with empty indices."""
    keys = list(BlockSchema.iter_admissible_keys([]))
    
    # Empty cartesian product yields one empty tuple
    assert keys == [()]


def test_shape_for_key():
    """Test BlockSchema.shape_for_key."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(-1, 4)))
    
    shape1 = BlockSchema.shape_for_key([idx1, idx2], (0, 0))
    assert shape1 == (2, 5)
    
    shape2 = BlockSchema.shape_for_key([idx1, idx2], (1, -1))
    assert shape2 == (3, 4)


def test_shape_for_key_single():
    """Test BlockSchema.shape_for_key with single index."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    
    shape = BlockSchema.shape_for_key([idx], (1,))
    assert shape == (3,)


def test_shape_for_key_invalid_length():
    """Test BlockSchema.shape_for_key with mismatched key length."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    with pytest.raises(ValueError, match="length does not match"):
        BlockSchema.shape_for_key([idx], (0, 1))


def test_shape_for_key_missing_charge():
    """Test BlockSchema.shape_for_key with charge not in index."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    with pytest.raises(KeyError, match="not present"):
        BlockSchema.shape_for_key([idx], (1,))


def test_shape_for_key_with_num_components():
    """Test BlockSchema.shape_for_key with num_components parameter."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(-1, 4)))
    
    # Without num_components
    shape1 = BlockSchema.shape_for_key([idx1, idx2], (0, 0))
    assert shape1 == (2, 5)
    
    # With num_components
    shape2 = BlockSchema.shape_for_key([idx1, idx2], (0, 0), num_components=3)
    assert shape2 == (2, 5, 3)
    
    shape3 = BlockSchema.shape_for_key([idx1, idx2], (1, -1), num_components=1)
    assert shape3 == (3, 4, 1)


def test_shape_for_key_num_components_validation():
    """Test BlockSchema.shape_for_key validates num_components > 0."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    with pytest.raises(ValueError, match="num_components must be at least 1"):
        BlockSchema.shape_for_key([idx], (0,), num_components=0)
    
    with pytest.raises(ValueError, match="num_components must be at least 1"):
        BlockSchema.shape_for_key([idx], (0,), num_components=-1)


def test_validate_blocks_valid():
    """Test BlockSchema.validate_blocks with valid blocks."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(-1, 4)))
    
    blocks = {
        (0, 0): torch.zeros((2, 5)),
        (1, -1): torch.zeros((3, 4))
    }
    
    # Should not raise
    BlockSchema.validate_blocks([idx1, idx2], blocks)


def test_validate_blocks_wrong_shape():
    """Test BlockSchema.validate_blocks with wrong shape."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5),))
    
    blocks = {
        (0, 0): torch.zeros((2, 3))  # Wrong shape, should be (2, 5)
    }
    
    with pytest.raises(ValueError, match="expected"):
        BlockSchema.validate_blocks([idx1, idx2], blocks)


def test_validate_blocks_not_torch():
    """Test BlockSchema.validate_blocks with non-torch tensors."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    blocks = {
        (0,): [[1, 2], [3, 4]]  # List, not torch tensor
    }
    
    with pytest.raises(TypeError, match="torch tensors"):
        BlockSchema.validate_blocks([idx], blocks)


def test_validate_blocks_with_intw_valid():
    """Test BlockSchema.validate_blocks with valid intw for non-Abelian groups."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Create Bridge with 2 components
    bridge = dg.Bridge.from_block(group, (1, 1), [Direction.OUT, Direction.OUT], dtype=torch.float64)
    # Block shape should be (2, 2, num_components)
    blocks = {
        (1, 1): torch.zeros((2, 2, bridge.num_components))
    }
    intw = {
        (1, 1): bridge
    }
    
    # Should not raise
    BlockSchema.validate_blocks([idx1, idx2], blocks, intw)


def test_validate_blocks_with_intw_wrong_shape():
    """Test BlockSchema.validate_blocks raises when block shape doesn't match intw."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    bridge = dg.Bridge.from_block(group, (1, 1), [Direction.OUT, Direction.OUT], dtype=torch.float64)
    # Wrong shape: missing trailing dimension
    blocks = {
        (1, 1): torch.zeros((2, 2))
    }
    intw = {
        (1, 1): bridge
    }
    
    with pytest.raises(ValueError, match="expected.*trailing reduced multiplicity"):
        BlockSchema.validate_blocks([idx1, idx2], blocks, intw)


def test_validate_blocks_with_intw_key_mismatch():
    """Test BlockSchema.validate_blocks raises when intw keys don't match block keys."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    bridge = dg.Bridge.from_block(group, (1, 1), [Direction.OUT, Direction.OUT], dtype=torch.float64)
    blocks = {
        (1, 1): torch.zeros((2, 2, bridge.num_components))
    }
    # Wrong key in intw
    intw = {
        (0, 0): bridge
    }
    
    with pytest.raises(ValueError, match="keys must match block keys"):
        BlockSchema.validate_blocks([idx1, idx2], blocks, intw)


def test_validate_blocks_with_intw_multiple_blocks():
    """Test BlockSchema.validate_blocks with multiple blocks and intertwiners."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Create bridges for different keys
    bridge1 = dg.Bridge.from_block(group, (1, 1), [Direction.OUT, Direction.OUT], dtype=torch.float64)
    bridge2 = dg.Bridge.from_block(group, (2, 1), [Direction.OUT, Direction.OUT], dtype=torch.float64)
    
    blocks = {
        (1, 1): torch.zeros((2, 2, bridge1.num_components)),
        (2, 1): torch.zeros((3, 2, bridge2.num_components))
    }
    intw = {
        (1, 1): bridge1,
        (2, 1): bridge2
    }
    
    # Should not raise
    BlockSchema.validate_blocks([idx1, idx2], blocks, intw)


def test_validate_blocks_abelian_with_none_intw():
    """Test BlockSchema.validate_blocks works for Abelian groups with intw=None."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(-1, 4)))
    
    blocks = {
        (0, 0): torch.zeros((2, 5)),
        (1, -1): torch.zeros((3, 4))
    }
    
    # Abelian: intw=None, no trailing dimension
    BlockSchema.validate_blocks([idx1, idx2], blocks, intw=None)


def test_charge_totals_neutral():
    """Test BlockSchema.charge_totals for neutral block."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(1, 4)))
    
    # Block (1, 1): OUT(1) + IN(1) = 1 + inverse(1) = 1 + (-1) = 0
    total = BlockSchema.charge_totals([idx1, idx2], (1, 1))
    
    assert total == 0


def test_charge_totals_non_neutral():
    """Test BlockSchema.charge_totals for non-neutral block."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(-1, 4)))
    
    # Block (1, 0): OUT(1) + IN(0) = 1 + 0 = 1 (not neutral)
    total = BlockSchema.charge_totals([idx1, idx2], (1, 0))
    
    assert total == 1


def test_charge_totals_multiple_out():
    """Test BlockSchema.charge_totals with multiple OUT indices."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(2, 3),))
    idx3 = Index(Direction.IN, group, sectors=(Sector(3, 4),))
    
    # OUT(1) + OUT(2) + IN(3) = 1 + 2 - 3 = 0
    total = BlockSchema.charge_totals([idx1, idx2, idx3], (1, 2, 3))
    
    assert total == 0


def test_charge_totals_z2():
    """Test BlockSchema.charge_totals with Z2 group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))
    
    # OUT(1) + IN(1) = 1 XOR 1 = 0
    total = BlockSchema.charge_totals([idx1, idx2], (1, 1))
    
    assert total == 0


def test_charges_conserved_true():
    """Test BlockSchema.charges_conserved returns True for neutral."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(1, 4)))
    
    assert BlockSchema.charges_conserved([idx1, idx2], (0, 0))
    assert BlockSchema.charges_conserved([idx1, idx2], (1, 1))


def test_charges_conserved_false():
    """Test BlockSchema.charges_conserved returns False for non-neutral."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(-1, 4)))
    
    assert not BlockSchema.charges_conserved([idx1, idx2], (1, 0))
    assert not BlockSchema.charges_conserved([idx1, idx2], (0, -1))


def test_charges_conserved_z2():
    """Test BlockSchema.charges_conserved with Z2 group."""
    group = Z2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 1)))
    
    # Conservation: q1 XOR q2 XOR q3 = 0
    assert BlockSchema.charges_conserved([idx1, idx2, idx3], (0, 0, 0))
    assert BlockSchema.charges_conserved([idx1, idx2, idx3], (1, 1, 0))
    assert BlockSchema.charges_conserved([idx1, idx2, idx3], (1, 0, 1))
    assert BlockSchema.charges_conserved([idx1, idx2, idx3], (0, 1, 1))
    
    assert not BlockSchema.charges_conserved([idx1, idx2, idx3], (1, 0, 0))
    assert not BlockSchema.charges_conserved([idx1, idx2, idx3], (0, 1, 0))


def test_block_schema_empty_indices():
    """Test BlockSchema methods with empty indices."""
    keys = list(BlockSchema.iter_admissible_keys([]))
    assert keys == [()]
    
    shape = BlockSchema.shape_for_key([], ())
    assert shape == ()
    
    # Empty blocks dict should be valid
    BlockSchema.validate_blocks([], {})


def test_validate_blocks_empty_dict():
    """Test BlockSchema.validate_blocks with empty blocks dict."""
    group = U1Group()
    idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
    
    # Should not raise
    BlockSchema.validate_blocks([idx], {})


# charge_avail tests (new method for non-Abelian groups)

def test_charge_avail_non_abelian_su2():
    """Test BlockSchema.charge_avail with non-Abelian SU2 group."""
    group = SU2Group()
    # Two spin-1/2 indices (charge 1 in 2j notation)
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # spin-1/2 ⊗ spin-1/2 → spin-0 or spin-1 (2j: 0 or 2)
    avail = BlockSchema.charge_avail([idx1, idx2], (1, 1))
    assert avail == (0, 2)  # Multiple channels


def test_charge_avail_non_abelian_three_spins():
    """Test BlockSchema.charge_avail with three spin-1/2 particles."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Three spin-1/2: 1⊗1⊗1 → {1, 3} (only half-integer spins)
    avail = BlockSchema.charge_avail([idx1, idx2, idx3], (1, 1, 1))
    assert avail == (1, 3)


def test_charge_avail_with_directions():
    """Test BlockSchema.charge_avail respects IN/OUT directions."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(2, 3),))  # spin-1
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 2),))   # spin-1/2 (dual)
    
    # OUT: 2, IN: dual(1) = 1 (SU2 is self-dual)
    # So: 2 ⊗ 1 → {1, 3}
    avail = BlockSchema.charge_avail([idx1, idx2], (2, 1))
    assert avail == (1, 3)


# charges_conserved tests (uses is_abelian to switch logic)

def test_charges_conserved_abelian_still_works():
    """Test that charges_conserved still works for Abelian groups."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 1), Sector(2, 1)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(1, 1), Sector(2, 1)))
    
    # 2 + dual(2) = 2 + (-2) = 0 (conserved)
    assert BlockSchema.charges_conserved([idx1, idx2], (2, 2)) is True
    
    # 1 + dual(2) = 1 + (-2) = -1 (not conserved)
    assert BlockSchema.charges_conserved([idx1, idx2], (1, 2)) is False


def test_charges_conserved_non_abelian_su2():
    """Test charges_conserved with non-Abelian SU2 group."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # spin-1/2 ⊗ spin-1/2 → {0, 2}
    # Neutral is 0, which is in the set → conserved
    assert BlockSchema.charges_conserved([idx1, idx2], (1, 1)) is True


def test_charges_conserved_non_abelian_not_conserved():
    """Test charges_conserved returns False when neutral not in channels."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(2, 3),))  # spin-1
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))  # spin-1/2
    
    # spin-1 ⊗ spin-1/2 (2 ⊗ 1) → {1, 3}
    # Neutral is 0, which is NOT in the set → not conserved
    assert BlockSchema.charges_conserved([idx1, idx2], (2, 1)) is False


def test_charges_conserved_non_abelian_three_spins():
    """Test charges_conserved with three spin-1/2 particles."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx2 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
    
    # Three spin-1/2: 1⊗1⊗1 → {1, 3}
    # Neutral is 0, NOT in the set → not conserved
    assert BlockSchema.charges_conserved([idx1, idx2, idx3], (1, 1, 1)) is False
    
    # But if we add one more spin-1/2 in IN direction:
    idx4 = Index(Direction.IN, group, sectors=(Sector(1, 2),))
    # 1⊗1⊗1⊗dual(1) = 1⊗1⊗1⊗1 → {0, 2, 4}
    # Neutral is 0, which IS in the set → conserved
    assert BlockSchema.charges_conserved([idx1, idx2, idx3, idx4], (1, 1, 1, 1)) is True


# bridge_collinear tests

def test_bridge_collinear_identical():
    """Test bridge_collinear with identical weights."""
    group = SU2Group()
    key = (1, 1)
    directions = [Direction.OUT, Direction.OUT]
    
    bridge_a = dg.Bridge.from_block(group, key, directions, dtype=torch.float64)
    bridge_b = dg.Bridge.from_block(group, key, directions, dtype=torch.float64)
    
    compatible, scale = BlockSchema.bridge_collinear(bridge_a, bridge_b)
    
    assert compatible is True
    assert math.isclose(scale, 1.0)


def test_bridge_collinear_parallel_vectors():
    """Test bridge_collinear with parallel weight vectors (collinear)."""
    group = SU2Group()
    key = (1, 1)
    directions = [Direction.OUT, Direction.OUT]
    
    bridge_a = dg.Bridge.from_block(group, key, directions, dtype=torch.float64)
    
    # Create bridge_b with scaled weights
    alpha = 2.5
    weights_b = bridge_a.weights * alpha
    bridge_b = dg.Bridge(cgspec=bridge_a.cgspec, weights=weights_b)
    
    compatible, scale = BlockSchema.bridge_collinear(bridge_a, bridge_b)
    
    assert compatible is True
    assert math.isclose(scale, alpha, rel_tol=1e-9)


def test_bridge_collinear_antiparallel_vectors():
    """Test bridge_collinear with antiparallel weight vectors (negative scale)."""
    group = SU2Group()
    key = (1, 1)
    directions = [Direction.OUT, Direction.OUT]
    
    bridge_a = dg.Bridge.from_block(group, key, directions, dtype=torch.float64)
    
    # Create bridge_b with negatively scaled weights
    alpha = -1.5
    weights_b = bridge_a.weights * alpha
    bridge_b = dg.Bridge(cgspec=bridge_a.cgspec, weights=weights_b)
    
    compatible, scale = BlockSchema.bridge_collinear(bridge_a, bridge_b)
    
    assert compatible is True
    assert math.isclose(scale, alpha, rel_tol=1e-9)


def test_bridge_collinear_different_directions():
    """Test bridge_collinear with non-parallel weight vectors."""
    group = SU2Group()
    key = (1, 1, 1, 1)  # 4 indices for non-trivial OM
    directions = [Direction.IN, Direction.IN, Direction.IN, Direction.OUT]
    
    bridge_a = dg.Bridge.from_block(group, key, directions, dtype=torch.float64)
    
    # Default weights from from_block are [1, 0, 0, ...]
    # Create bridge_b with orthogonal weights [0, 1, 0, ...]
    om_dim = bridge_a.om_dimension
    weights_b = torch.zeros(1, om_dim, dtype=torch.float64)
    if om_dim > 1:
        weights_b[0, 1] = 1.0  # Orthogonal to default
    else:
        # om_dim == 1 edge case: use different scalar
        weights_b[0, 0] = 2.0
    bridge_b = dg.Bridge(cgspec=bridge_a.cgspec, weights=weights_b)
    
    compatible, scale = BlockSchema.bridge_collinear(bridge_a, bridge_b)
    
    # Orthogonal or different vectors → incompatible
    assert compatible is False


def test_bridge_collinear_different_num_components():
    """Test bridge_collinear with different num_components."""
    group = SU2Group()
    key = (1, 1)
    directions = [Direction.OUT, Direction.OUT]
    
    bridge_a = dg.Bridge.from_block(group, key, directions, dtype=torch.float64)
    
    # Create bridge_b with multiple components
    om_dim = bridge_a.om_dimension
    weights_b = torch.randn(3, om_dim, dtype=torch.float64)
    bridge_b = dg.Bridge(cgspec=bridge_a.cgspec, weights=weights_b)
    
    compatible, scale = BlockSchema.bridge_collinear(bridge_a, bridge_b)
    
    # Different shapes, not single component → incompatible
    assert compatible is False


def test_bridge_collinear_zero_weights():
    """Test bridge_collinear with zero weight vectors."""
    group = SU2Group()
    key = (1, 1)
    directions = [Direction.OUT, Direction.OUT]
    
    bridge_a = dg.Bridge.from_block(group, key, directions, dtype=torch.float64)
    
    # Create bridge with zero weights
    om_dim = bridge_a.om_dimension
    weights_zero = torch.zeros(1, om_dim, dtype=torch.float64)
    bridge_zero = dg.Bridge(cgspec=bridge_a.cgspec, weights=weights_zero)
    
    # Both zero
    compatible, scale = BlockSchema.bridge_collinear(bridge_zero, bridge_zero)
    assert compatible is True
    
    # One zero, one non-zero
    compatible, scale = BlockSchema.bridge_collinear(bridge_a, bridge_zero)
    assert compatible is False

