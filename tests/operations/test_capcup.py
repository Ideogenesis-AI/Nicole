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


"""Tests for the capcup maneuver.

capcup(A, axis_a, B, axis_b) inverts both bond directions of a contraction pair
and applies the Frobenius-Schur (FS) phase (-1)^{2j} block-wise to B for SU(2)
tensors.  After the operation any contraction involving the bond yields the same
numerical result as before.

Test groups
-----------
1. Error handling  – sanity checks raise the correct exceptions.
2. Direction updates – both bond directions are inverted in-place.
3. SU(2) phase – B's data is multiplied by (-1)^{2j}; A's data is unchanged.
4. Abelian no-phase – no phase is applied for Abelian groups.
5. Correctness – contraction before and after capcup yields identical results,
   for U1 (2nd/3rd/5th order), SU(2) (2nd/3rd/5th order, all 4 bond-position
   combinations of first-(n-1) vs last axis), Abelian ProductGroup (U1×U1),
   and non-Abelian ProductGroup (Z2×SU2).
"""

import pytest
import torch

from nicole import Direction, Index, Sector, Tensor
from nicole import SU2Group, U1Group, Z2Group, ProductGroup
from nicole import capcup, contract
from ..utils import (
    assert_blocks_equal,
    assert_physical_tensors_equal,
    populate_random_weights,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _u1_pair(seed_a: int = 0, seed_b: int = 1):
    """Return (A, B) U1 tensors with a shared bond at axis 0 (tag 'bond')."""
    group = U1Group()
    idx_bond_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(-1, 2)))
    idx_free_a = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_free_b = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    A = Tensor.random([idx_bond_out, idx_free_a], seed=seed_a, itags=["bond", "free_a"])
    B = Tensor.random([idx_bond_out.flip(), idx_free_b], seed=seed_b, itags=["bond", "free_b"])
    return A, B


def _su2_pair(seed_a: int = 0, seed_b: int = 1):
    """Return (A, B) SU(2) tensors with a shared bond at axis 0 (tag 'bond').

    The bond index carries spin-1/2 (2j=1, FS phase -1) and spin-1 (2j=2,
    FS phase +1) sectors, giving a non-trivial phase pattern.
    """
    group = SU2Group()
    idx_bond_out = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    idx_free_a = Index(Direction.IN, group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_free_b = Index(Direction.IN, group, sectors=(Sector(1, 2), Sector(2, 3)))
    A = Tensor.random([idx_bond_out, idx_free_a], seed=seed_a, itags=["bond", "free_a"])
    B = Tensor.random([idx_bond_out.flip(), idx_free_b], seed=seed_b, itags=["bond", "free_b"])
    return A, B


# ---------------------------------------------------------------------------
# 1. Error handling
# ---------------------------------------------------------------------------

def test_capcup_raises_on_mismatched_itag():
    """ValueError if the two bond indices carry different itags."""
    A, B = _u1_pair()
    B.retag({"bond": "other"})
    with pytest.raises(ValueError, match="tags do not match"):
        capcup(A, 0, B, 0)


def test_capcup_raises_on_same_direction_both_out():
    """ValueError if both bonds point in the same direction (OUT/OUT)."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 1)))
    A = Tensor.random([idx_out, idx_in], seed=0, itags=["x", "y"])
    B = Tensor.random([idx_out, idx_in], seed=1, itags=["x", "y"])
    with pytest.raises(ValueError, match="opposite directions"):
        capcup(A, 0, B, 0)


def test_capcup_raises_on_same_direction_both_in():
    """ValueError if both bonds point in the same direction (IN/IN)."""
    group = U1Group()
    idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
    idx_in  = Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 1)))
    A = Tensor.random([idx_out, idx_in], seed=0, itags=["x", "y"])
    B = Tensor.random([idx_out, idx_in], seed=1, itags=["x", "y"])
    with pytest.raises(ValueError, match="opposite directions"):
        capcup(A, 1, B, 1)


def test_capcup_raises_on_out_of_range_axis_handled_by_invert():
    """Out-of-range integer axes are caught by the downstream invert call."""
    A, B = _u1_pair()
    with pytest.raises(Exception):
        capcup(A, 99, B, 0)




# ---------------------------------------------------------------------------
# 2. Direction updates
# ---------------------------------------------------------------------------

def test_capcup_inverts_A_direction():
    """A's bond direction is inverted from OUT to IN."""
    A, B = _u1_pair()
    original_dir = A.indices[0].direction
    capcup(A, 0, B, 0)
    assert A.indices[0].direction != original_dir


def test_capcup_inverts_B_direction():
    """B's bond direction is inverted from IN to OUT."""
    A, B = _u1_pair()
    original_dir = B.indices[0].direction
    capcup(A, 0, B, 0)
    assert B.indices[0].direction != original_dir


def test_capcup_leaves_free_directions_unchanged():
    """Non-bond axes are not affected in either tensor."""
    A, B = _u1_pair()
    dir_free_a = A.indices[1].direction
    dir_free_b = B.indices[1].direction
    capcup(A, 0, B, 0)
    assert A.indices[1].direction == dir_free_a
    assert B.indices[1].direction == dir_free_b


def test_capcup_su2_inverts_bridge_direction_in_A():
    """For SU(2), A's Bridge edge direction at the bond axis is inverted."""
    A, B = _su2_pair()
    original_dirs = {key: bridge.cgspec.get_directions() for key, bridge in A.intw.items()}
    capcup(A, 0, B, 0)
    for key, bridge in A.intw.items():
        new_dirs = bridge.cgspec.get_directions()
        assert new_dirs[0] == -original_dirs[key][0], "Bond edge direction must flip"
        assert new_dirs[1] == original_dirs[key][1], "Free edge direction must be unchanged"


def test_capcup_su2_inverts_bridge_direction_in_B():
    """For SU(2), B's Bridge edge direction at the bond axis is inverted."""
    A, B = _su2_pair()
    original_dirs = {key: bridge.cgspec.get_directions() for key, bridge in B.intw.items()}
    capcup(A, 0, B, 0)
    for key, bridge in B.intw.items():
        new_dirs = bridge.cgspec.get_directions()
        assert new_dirs[0] == -original_dirs[key][0], "Bond edge direction must flip"
        assert new_dirs[1] == original_dirs[key][1], "Free edge direction must be unchanged"


# ---------------------------------------------------------------------------
# 3. SU(2) phase on B
# ---------------------------------------------------------------------------

def test_capcup_su2_applies_fs_phase_to_B():
    """B's intertwiner weights are multiplied by (-1)^{2j} at the bond axis.

    Spin-1/2 blocks (2j=1, phase=-1) are negated; spin-1 blocks (2j=2,
    phase=+1) are unchanged.  The data blocks are left intact.
    """
    A, B = _su2_pair(seed_a=10, seed_b=11)
    original_weights = {key: bridge.weights.clone() for key, bridge in B.intw.items()}
    original_data    = {key: block.clone() for key, block in B.data.items()}

    capcup(A, 0, B, 0)

    for orig_key, orig_w in original_weights.items():
        two_j = orig_key[0]          # bond is at axis 0
        expected_phase = (-1) ** two_j
        # For SU(2) dual(charge) == charge, so the key is unchanged
        assert orig_key in B.intw, "Block key must survive inversion for SU(2)"
        assert torch.allclose(B.intw[orig_key].weights, expected_phase * orig_w), (
            f"Block {orig_key}: expected phase {expected_phase} in weights"
        )
        assert torch.allclose(B.data[orig_key], original_data[orig_key]), (
            f"Block {orig_key}: data should not be modified"
        )


def test_capcup_su2_no_phase_on_A():
    """A's data blocks are not multiplied by any phase factor."""
    A, B = _su2_pair(seed_a=12, seed_b=13)
    original_data = {key: block.clone() for key, block in A.data.items()}

    capcup(A, 0, B, 0)

    for orig_key, orig_block in original_data.items():
        assert orig_key in A.data, "Block key must survive inversion for SU(2)"
        assert torch.allclose(A.data[orig_key], orig_block), (
            f"A's block {orig_key} should not be modified by capcup"
        )


# ---------------------------------------------------------------------------
# 4. Abelian: no phase applied
# ---------------------------------------------------------------------------

def test_capcup_abelian_B_data_unchanged():
    """For U1Group no phase is applied; B's block values are unchanged."""
    A, B = _u1_pair(seed_a=20, seed_b=21)
    # For U1Group, dual(q) = -q, so after invert the block keys change.
    # Collect (sorted original value, charge) pairs to verify values are intact.
    original_values = {key: block.clone() for key, block in B.data.items()}

    capcup(A, 0, B, 0)

    # After invert the bond charge becomes dual(-q) = +q (U1: dual is negation).
    # So the key (q_bond, q_free) maps to (-q_bond, q_free).
    for orig_key, orig_block in original_values.items():
        q_bond, q_free = orig_key
        new_key = (-q_bond, q_free)
        assert new_key in B.data, f"Expected key {new_key} not found"
        assert torch.allclose(B.data[new_key], orig_block), (
            f"Abelian capcup must not change block values"
        )


def test_capcup_abelian_A_data_unchanged():
    """For U1Group no phase is applied; A's block values are unchanged."""
    A, B = _u1_pair(seed_a=22, seed_b=23)
    original_values = {key: block.clone() for key, block in A.data.items()}

    capcup(A, 0, B, 0)

    for orig_key, orig_block in original_values.items():
        q_bond, q_free = orig_key
        new_key = (-q_bond, q_free)
        assert new_key in A.data, f"Expected key {new_key} not found"
        assert torch.allclose(A.data[new_key], orig_block)


# ---------------------------------------------------------------------------
# 5. Correctness: contraction is preserved
# ---------------------------------------------------------------------------

# Helper: build a pair of tensors where the bond is at a chosen axis position.
def _su2_pair_bond_at(bond_idx_out, free_a, free_b, axis_a, axis_b, seed):
    """Return (A, B) with the bond index inserted at axis_a / axis_b respectively."""
    idxs_a = list(free_a)
    idxs_a.insert(axis_a, bond_idx_out)
    itags_a = [f"fa{i}" for i in range(len(free_a))]
    itags_a.insert(axis_a, "bond")

    idxs_b = list(free_b)
    idxs_b.insert(axis_b, bond_idx_out.flip())
    itags_b = [f"fb{i}" for i in range(len(free_b))]
    itags_b.insert(axis_b, "bond")

    A = Tensor.random(idxs_a, seed=seed,   itags=itags_a)
    B = Tensor.random(idxs_b, seed=seed+1, itags=itags_b)
    populate_random_weights(A, seed=seed+2)
    populate_random_weights(B, seed=seed+3)
    return A, B


# --- Abelian (U1): 2nd / 3rd / 5th order ---

def test_capcup_abelian_contraction_preserved_2nd_order():
    """U1, 2nd order: contraction is unchanged after capcup."""
    A, B = _u1_pair(seed_a=30, seed_b=31)
    result_before = contract(A, B, axes=(0, 0))
    capcup(A, 0, B, 0)
    result_after = contract(A, B, axes=(0, 0))
    assert_blocks_equal(result_before, result_after)


def test_capcup_abelian_contraction_preserved_3rd_order():
    """U1, 3rd order: contraction is unchanged after capcup."""
    group = U1Group()
    idx_bond = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(-1, 2)))
    idx_fa1  = Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))
    idx_fa2  = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 1)))
    idx_fb1  = Index(Direction.IN,  group, sectors=(Sector(0, 3), Sector(-1, 2)))
    idx_fb2  = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 1)))

    A = Tensor.random([idx_bond, idx_fa1, idx_fa2], seed=32, itags=["bond", "fa1", "fa2"])
    B = Tensor.random([idx_bond.flip(), idx_fb1, idx_fb2], seed=33, itags=["bond", "fb1", "fb2"])
    result_before = contract(A, B, axes=(0, 0))
    capcup(A, 0, B, 0)
    result_after = contract(A, B, axes=(0, 0))
    assert_blocks_equal(result_before, result_after)


def test_capcup_abelian_contraction_preserved_5th_order():
    """U1, 5th order: contraction is unchanged after capcup."""
    group = U1Group()
    idx_bond = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3)))
    free_a   = [Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2))),
                Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(-1, 1))),
                Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(-1, 2))),
                Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 1)))]
    free_b   = [Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 1))),
                Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(-1, 1))),
                Index(Direction.IN,  group, sectors=(Sector(0, 2), Sector(1, 2))),
                Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(-1, 3)))]

    idxs_a = [idx_bond] + free_a
    idxs_b = [idx_bond.flip()] + free_b
    A = Tensor.random(idxs_a, seed=34, itags=["bond", "fa1", "fa2", "fa3", "fa4"])
    B = Tensor.random(idxs_b, seed=35, itags=["bond", "fb1", "fb2", "fb3", "fb4"])
    result_before = contract(A, B, axes=(0, 0))
    capcup(A, 0, B, 0)
    result_after = contract(A, B, axes=(0, 0))
    assert_blocks_equal(result_before, result_after)


# --- SU(2): 2nd / 3rd / 5th order, all 4 bond-position combinations ---
# For an n-th order tensor, the bond can sit in the "first (n-1)" positions
# or at the "last" position.  The 4 combinations cover both choices for A
# and B independently.

@pytest.mark.parametrize("axis_a,axis_b", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_capcup_su2_contraction_preserved_2nd_order(axis_a, axis_b):
    """SU(2), 2nd order: contraction is unchanged after capcup (all bond positions)."""
    group = SU2Group()
    bond_idx_out = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    free_a = [Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2)))]
    free_b = [Index(Direction.IN,  group, sectors=(Sector(1, 2), Sector(2, 3)))]

    A, B = _su2_pair_bond_at(bond_idx_out, free_a, free_b, axis_a, axis_b, seed=60)
    result_before = contract(A, B, axes=(axis_a, axis_b))
    capcup(A, axis_a, B, axis_b)
    result_after = contract(A, B, axes=(axis_a, axis_b))
    assert_physical_tensors_equal(result_before, result_after)


@pytest.mark.parametrize("axis_a,axis_b", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_capcup_su2_contraction_preserved_3rd_order(axis_a, axis_b):
    """SU(2), 3rd order: contraction is unchanged after capcup (all bond positions)."""
    group = SU2Group()
    bond_idx_out = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    free_a = [Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2))),
              Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))]
    free_b = [Index(Direction.IN,  group, sectors=(Sector(1, 2), Sector(2, 3))),
              Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 3)))]

    A, B = _su2_pair_bond_at(bond_idx_out, free_a, free_b, axis_a, axis_b, seed=70)
    result_before = contract(A, B, axes=(axis_a, axis_b))
    capcup(A, axis_a, B, axis_b)
    result_after = contract(A, B, axes=(axis_a, axis_b))
    assert_physical_tensors_equal(result_before, result_after)


@pytest.mark.parametrize("axis_a,axis_b", [(0, 0), (0, 4), (4, 0), (4, 4)])
def test_capcup_su2_contraction_preserved_5th_order(axis_a, axis_b):
    """SU(2), 5th order: contraction is unchanged after capcup (all bond positions)."""
    group = SU2Group()
    bond_idx_out = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    free_a = [Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2))),
              Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3))),
              Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(2, 3))),
              Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))]
    free_b = [Index(Direction.IN,  group, sectors=(Sector(1, 2), Sector(2, 3))),
              Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(2, 3))),
              Index(Direction.IN,  group, sectors=(Sector(1, 2), Sector(2, 3))),
              Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2)))]

    A, B = _su2_pair_bond_at(bond_idx_out, free_a, free_b, axis_a, axis_b, seed=80)
    result_before = contract(A, B, axes=(axis_a, axis_b))
    capcup(A, axis_a, B, axis_b)
    result_after = contract(A, B, axes=(axis_a, axis_b))
    assert_physical_tensors_equal(result_before, result_after)


# --- ProductGroup Abelian (U1 × U1): 3rd order ---

def test_capcup_product_abelian_contraction_preserved_3rd_order():
    """U1×U1, 3rd order: contraction is unchanged after capcup (Abelian ProductGroup)."""
    group = ProductGroup([U1Group(), U1Group()])
    idx_bond = Index(Direction.OUT, group,
                     sectors=(Sector((0, 0), 2), Sector((1, -1), 1), Sector((-1, 1), 1)))
    idx_fa1  = Index(Direction.IN,  group, sectors=(Sector((0, 0), 1), Sector((1, 1), 2)))
    idx_fa2  = Index(Direction.OUT, group, sectors=(Sector((0, 0), 2), Sector((-1, 0), 1)))
    idx_fb1  = Index(Direction.IN,  group, sectors=(Sector((0, 0), 2), Sector((-1, -1), 1)))
    idx_fb2  = Index(Direction.OUT, group, sectors=(Sector((0, 0), 1), Sector((1, 0), 2)))

    A = Tensor.random([idx_bond, idx_fa1, idx_fa2], seed=90, itags=["bond", "fa1", "fa2"])
    B = Tensor.random([idx_bond.flip(), idx_fb1, idx_fb2], seed=91, itags=["bond", "fb1", "fb2"])
    result_before = contract(A, B, axes=(0, 0))
    capcup(A, 0, B, 0)
    result_after = contract(A, B, axes=(0, 0))
    assert_blocks_equal(result_before, result_after)


# --- ProductGroup with SU(2) (Z2 × SU2): 3rd order, all 4 bond positions ---

@pytest.mark.parametrize("axis_a,axis_b", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_capcup_product_su2_contraction_preserved_3rd_order(axis_a, axis_b):
    """Z2×SU2, 3rd order: contraction is unchanged after capcup (all bond positions)."""
    group = ProductGroup([Z2Group(), SU2Group()])
    bond_idx_out = Index(Direction.OUT, group,
                         sectors=(Sector((0, 1), 2), Sector((1, 2), 3)))
    free_a = [Index(Direction.IN,  group, sectors=(Sector((0, 0), 1), Sector((1, 1), 2))),
              Index(Direction.OUT, group, sectors=(Sector((0, 1), 2), Sector((1, 2), 3)))]
    free_b = [Index(Direction.IN,  group, sectors=(Sector((0, 1), 2), Sector((1, 0), 1))),
              Index(Direction.OUT, group, sectors=(Sector((0, 0), 1), Sector((1, 2), 3)))]

    A, B = _su2_pair_bond_at(bond_idx_out, free_a, free_b, axis_a, axis_b, seed=100)
    result_before = contract(A, B, axes=(axis_a, axis_b))
    capcup(A, axis_a, B, axis_b)
    result_after = contract(A, B, axes=(axis_a, axis_b))
    assert_physical_tensors_equal(result_before, result_after)


# --- Full contraction to scalar ---
# Contract ALL axes of two 3rd-order SU(2) tensors.  capcup is applied to one
# bond; the other two pairs are contracted as-is.  The scalar result must be
# identical before and after.  All 4 bond-position combinations
# (leading/terminal in each tensor) are tested.

@pytest.mark.parametrize("axis_a,axis_b", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_capcup_su2_scalar_contraction_preserved(axis_a, axis_b):
    """SU(2) full contraction to scalar is unchanged after capcup (all bond positions)."""
    group = SU2Group()
    bond_idx_out = Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))
    free = [Index(Direction.IN,  group, sectors=(Sector(0, 1), Sector(1, 2))),
            Index(Direction.OUT, group, sectors=(Sector(1, 2), Sector(2, 3)))]

    n = 3
    free_axes_a = [i for i in range(n) if i != axis_a]
    free_axes_b = [i for i in range(n) if i != axis_b]

    # Free pair i in A and B share itag "fi" so contract's itag check passes.
    idxs_a: list = [None] * n
    itags_a: list = [None] * n
    idxs_a[axis_a] = bond_idx_out
    itags_a[axis_a] = "bond"
    for i, pos in enumerate(free_axes_a):
        idxs_a[pos] = free[i]
        itags_a[pos] = f"f{i}"

    idxs_b: list = [None] * n
    itags_b: list = [None] * n
    idxs_b[axis_b] = bond_idx_out.flip()
    itags_b[axis_b] = "bond"
    for i, pos in enumerate(free_axes_b):
        idxs_b[pos] = free[i].flip()
        itags_b[pos] = f"f{i}"

    A = Tensor.random(idxs_a, seed=110, itags=itags_a)
    B = Tensor.random(idxs_b, seed=111, itags=itags_b)
    populate_random_weights(A, seed=112)
    populate_random_weights(B, seed=113)

    axes_a = [axis_a] + free_axes_a
    axes_b = [axis_b] + free_axes_b

    result_before = contract(A, B, axes=(axes_a, axes_b))
    capcup(A, axis_a, B, axis_b)
    result_after = contract(A, B, axes=(axes_a, axes_b))

    # Scalar tensor: intw is None, single block keyed by ()
    assert torch.allclose(result_before.data[()], result_after.data[()], atol=1e-10)


@pytest.mark.parametrize("axis_a,axis_b", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_capcup_product_z2su2_scalar_contraction_preserved(axis_a, axis_b):
    """Z2×SU2 full contraction to scalar is unchanged after capcup (all bond positions)."""
    group = ProductGroup([Z2Group(), SU2Group()])
    bond_idx_out = Index(Direction.OUT, group,
                         sectors=(Sector((0, 1), 2), Sector((1, 2), 3)))
    free = [Index(Direction.IN,  group, sectors=(Sector((0, 0), 1), Sector((1, 1), 2))),
            Index(Direction.OUT, group, sectors=(Sector((0, 1), 2), Sector((1, 2), 3)))]

    n = 3
    free_axes_a = [i for i in range(n) if i != axis_a]
    free_axes_b = [i for i in range(n) if i != axis_b]

    idxs_a: list = [None] * n
    itags_a: list = [None] * n
    idxs_a[axis_a] = bond_idx_out
    itags_a[axis_a] = "bond"
    for i, pos in enumerate(free_axes_a):
        idxs_a[pos] = free[i]
        itags_a[pos] = f"f{i}"

    idxs_b: list = [None] * n
    itags_b: list = [None] * n
    idxs_b[axis_b] = bond_idx_out.flip()
    itags_b[axis_b] = "bond"
    for i, pos in enumerate(free_axes_b):
        idxs_b[pos] = free[i].flip()
        itags_b[pos] = f"f{i}"

    A = Tensor.random(idxs_a, seed=120, itags=itags_a)
    B = Tensor.random(idxs_b, seed=121, itags=itags_b)
    populate_random_weights(A, seed=122)
    populate_random_weights(B, seed=123)

    axes_a = [axis_a] + free_axes_a
    axes_b = [axis_b] + free_axes_b

    result_before = contract(A, B, axes=(axes_a, axes_b))
    capcup(A, axis_a, B, axis_b)
    result_after = contract(A, B, axes=(axes_a, axes_b))

    assert torch.allclose(result_before.data[()], result_after.data[()], atol=1e-10)


@pytest.mark.parametrize("axis_a,axis_b", [(0, 0), (0, 2), (2, 0), (2, 2)])
def test_capcup_product_u1su2_scalar_contraction_preserved(axis_a, axis_b):
    """U1×SU2 full contraction to scalar is unchanged after capcup (all bond positions)."""
    group = ProductGroup([U1Group(), SU2Group()])
    bond_idx_out = Index(Direction.OUT, group,
                         sectors=(Sector((0, 1), 2), Sector((1, 2), 3)))
    free = [Index(Direction.IN,  group, sectors=(Sector((0, 0), 1), Sector((1, 1), 2))),
            Index(Direction.OUT, group, sectors=(Sector((0, 1), 2), Sector((1, 2), 3)))]

    n = 3
    free_axes_a = [i for i in range(n) if i != axis_a]
    free_axes_b = [i for i in range(n) if i != axis_b]

    idxs_a: list = [None] * n
    itags_a: list = [None] * n
    idxs_a[axis_a] = bond_idx_out
    itags_a[axis_a] = "bond"
    for i, pos in enumerate(free_axes_a):
        idxs_a[pos] = free[i]
        itags_a[pos] = f"f{i}"

    idxs_b: list = [None] * n
    itags_b: list = [None] * n
    idxs_b[axis_b] = bond_idx_out.flip()
    itags_b[axis_b] = "bond"
    for i, pos in enumerate(free_axes_b):
        idxs_b[pos] = free[i].flip()
        itags_b[pos] = f"f{i}"

    A = Tensor.random(idxs_a, seed=130, itags=itags_a)
    B = Tensor.random(idxs_b, seed=131, itags=itags_b)
    populate_random_weights(A, seed=132)
    populate_random_weights(B, seed=133)

    axes_a = [axis_a] + free_axes_a
    axes_b = [axis_b] + free_axes_b

    result_before = contract(A, B, axes=(axes_a, axes_b))
    capcup(A, axis_a, B, axis_b)
    result_after = contract(A, B, axes=(axes_a, axes_b))

    assert torch.allclose(result_before.data[()], result_after.data[()], atol=1e-10)
