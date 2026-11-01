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


"""Tests for the randomized SVD engine and the non-convergence fallback ladder."""

import logging
import math

import torch
import pytest

from nicole import Direction, Index, Sector, Tensor, contract, decomp
from nicole import U1Group, SU2Group
from nicole.decomp import svd, _rsvd_budget, _rsvd_matrix, _svd_robust
from ..utils import assert_charge_neutral, populate_random_weights


def _make_u1_tensor(seed: int = 7):
    """3rd-order U(1) tensor whose sectors have comfortably different sizes."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 8), Sector(1, 10), Sector(2, 6)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 7), Sector(1, 9)))
    idx3 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(1, 6)))
    return Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=seed)


def _make_su2_tensor(seed: int = 11):
    """3rd-order SU(2) tensor with populated intertwiner weights."""
    group = SU2Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 3), Sector(2, 2)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 2), Sector(2, 2)))
    idx3 = Index(Direction.OUT, group, sectors=(Sector(0, 1), Sector(1, 2), Sector(2, 2)))
    T = Tensor.random([idx1, idx2, idx3], itags=["a", "b", "c"], seed=seed)
    populate_random_weights(T, seed=seed + 1000)
    return T


def _sorted_values(s_blocks):
    """Flatten a block-keyed dict of singular values into one descending tensor."""
    return torch.cat([v for v in s_blocks.values()]).sort(descending=True).values


# Accuracy against the exact engine

def test_rsvd_matches_exact_singular_values():
    """The randomized engine recovers the retained spectrum of the exact engine."""
    T = _make_u1_tensor()

    _, s_exact, _ = svd(T, axis=0, trunc={"nkeep": 12})
    _, s_random, _ = svd(T, axis=0, trunc={"nkeep": 12, "seed": 5}, engine="random")

    exact = _sorted_values(s_exact)
    random = _sorted_values(s_random)

    assert len(exact) == len(random)
    assert torch.allclose(exact, random, rtol=1e-10, atol=1e-12)


def test_rsvd_bond_structure_matches_exact():
    """Both engines agree on the bond index sectors and their dimensions."""
    T = _make_u1_tensor()

    U_exact, _, Vh_exact = svd(T, axis=0, trunc={"nkeep": 12})
    U_random, _, Vh_random = svd(T, axis=0, trunc={"nkeep": 12, "seed": 5}, engine="random")

    assert U_exact.indices[1].sectors == U_random.indices[1].sectors
    assert Vh_exact.indices[0].sectors == Vh_random.indices[0].sectors
    assert_charge_neutral(U_random)
    assert_charge_neutral(Vh_random)


def test_rsvd_full_rank_reconstructs_tensor():
    """An nkeep at the total rank makes the randomized path lossless."""
    T = _make_u1_tensor()

    # Total rank across sectors bounds what any truncation could keep, so asking
    # for more leaves the decomposition complete.
    _, s_exact, _ = svd(T, axis=0)
    total_rank = sum(len(v) for v in s_exact.values())

    U, S, Vh = decomp(T, axes=0, mode="RSVD", trunc={"nkeep": total_rank, "seed": 3})
    reconstructed = contract(contract(U, S, axes=(1, 0)), Vh, axes=(1, 0))

    assert (T - reconstructed).norm() / T.norm() < 1e-10


# Per-sector rank budget

def test_rsvd_budget_covers_nkeep():
    """Ceiling allocation always leaves the global selection enough candidates."""
    ranks = {(0,): 20, (1,): 60, (2,): 100, (3,): 40}

    for nkeep in (1, 5, 40, 137, 220):
        budget = _rsvd_budget(ranks, nkeep)

        assert sum(budget.values()) >= min(nkeep, sum(ranks.values()))
        for q, full_rank in ranks.items():
            assert 1 <= budget[q] <= full_rank


def test_rsvd_budget_is_a_uniform_ratio():
    """Every sector is reduced by the same ratio of its own full rank."""
    ranks = {(0,): 20, (1,): 60, (2,): 100, (3,): 40}
    nkeep = 40
    rho = nkeep / sum(ranks.values())

    budget = _rsvd_budget(ranks, nkeep)

    for q, full_rank in ranks.items():
        assert budget[q] == min(full_rank, max(1, math.ceil(rho * full_rank)))


def test_rsvd_budget_floor_keeps_every_sector():
    """A tiny budget still leaves one singular value per sector."""
    ranks = {(0,): 100, (1,): 2, (2,): 50}

    budget = _rsvd_budget(ranks, 1)

    assert all(k >= 1 for k in budget.values())


def test_rsvd_keeps_every_sector_on_a_tight_budget():
    """The floor of one survives into the decomposition itself."""
    T = _make_u1_tensor()

    U, s_blocks, _ = svd(
        T, axis=0, engine="random",
        trunc={"nkeep": 2, "seed": 1, "oversample": 0},
    )

    # The global nkeep still caps the total, but no sector was starved before it.
    assert sum(len(v) for v in s_blocks.values()) == 2


# Isometry of the factors

@pytest.mark.parametrize("make_tensor", [_make_u1_tensor, _make_su2_tensor])
def test_rsvd_factors_are_isometric(make_tensor):
    """U and Vh from the randomized engine are isometries, Abelian or not."""
    T = make_tensor()

    U, _, Vh = svd(T, axis=0, trunc={"nkeep": 6, "seed": 2}, engine="random")

    # U† U = I over the bond index
    UdU = contract(U.conj(), U, axes=(0, 0))
    for key, block in UdU.data.items():
        mat = block if UdU.intw is None else block.squeeze(-1)
        assert torch.allclose(mat, torch.eye(mat.shape[0], dtype=mat.dtype), atol=1e-9)

    # Vh Vh† = I over the bond index, contracting away every right axis
    right_axes = tuple(range(1, len(Vh.indices)))
    VhVhd = contract(Vh, Vh.conj(), axes=(right_axes, right_axes))
    for key, block in VhVhd.data.items():
        mat = block if VhVhd.intw is None else block.squeeze(-1)
        assert torch.allclose(mat, torch.eye(mat.shape[0], dtype=mat.dtype), atol=1e-9)


def test_rsvd_su2_reconstructs_tensor():
    """A full-rank randomized decomposition round-trips an SU(2) tensor."""
    T = _make_su2_tensor()

    _, s_exact, _ = svd(T, axis=0)
    total_rank = sum(len(v) for v in s_exact.values())

    U, S, Vh = decomp(T, axes=0, mode="RSVD", trunc={"nkeep": total_rank, "seed": 4})
    reconstructed = contract(contract(U, S, axes=(1, 0)), Vh, axes=(1, 0))

    assert (T - reconstructed).norm() / T.norm() < 1e-9


# Seeding and reproducibility

def test_rsvd_seed_is_reproducible():
    """The same seed gives bit-identical singular values."""
    T = _make_u1_tensor()

    _, first, _ = svd(T, axis=0, trunc={"nkeep": 9, "seed": 17}, engine="random")
    _, second, _ = svd(T, axis=0, trunc={"nkeep": 9, "seed": 17}, engine="random")

    assert torch.equal(_sorted_values(first), _sorted_values(second))


def test_rsvd_different_seeds_agree_within_tolerance():
    """Changing the seed perturbs nothing observable on a resolvable spectrum."""
    T = _make_u1_tensor()

    _, first, _ = svd(T, axis=0, trunc={"nkeep": 9, "seed": 17}, engine="random")
    _, second, _ = svd(T, axis=0, trunc={"nkeep": 9, "seed": 99}, engine="random")

    assert torch.allclose(_sorted_values(first), _sorted_values(second), rtol=1e-8)


def test_rsvd_seed_does_not_disturb_global_rng():
    """Seeding is forked, so the caller's RNG stream continues unbroken."""
    T = _make_u1_tensor()

    torch.manual_seed(123)
    expected = torch.randn(4)

    torch.manual_seed(123)
    svd(T, axis=0, trunc={"nkeep": 5, "seed": 17}, engine="random")
    actual = torch.randn(4)

    assert torch.equal(expected, actual)


def test_rsvd_without_seed_leaves_rng_advanced():
    """Without a seed there is no fork, so the global generator is simply used."""
    T = _make_u1_tensor()

    torch.manual_seed(321)
    before = torch.randn(2)

    torch.manual_seed(321)
    svd(T, axis=0, trunc={"nkeep": 5}, engine="random")
    after = torch.randn(2)

    assert not torch.equal(before, after)


# Discarded weight

def test_rsvd_discarded_weight_matches_exact():
    """Both engines report the same relative squared weight loss."""
    T = _make_u1_tensor()

    *_, info_exact = svd(T, axis=0, trunc={"nkeep": 10}, requires_info=True)
    *_, info_random = svd(T, axis=0, trunc={"nkeep": 10, "seed": 8}, engine="random",
                          requires_info=True)

    assert info_random["discarded_weight"] == pytest.approx(
        info_exact["discarded_weight"], rel=1e-8
    )


def test_oversampling_rescues_a_misallocated_budget():
    """Oversampling repairs a dimension-based misallocation of the rank budget.

    The uniform ratio allocates rank by sector dimension, so a sector that is
    large but spectrally negligible draws budget away from a small dominant one.
    The oversampled triplets go into the global selection, which lets the
    under-served sector reclaim what the budget denied it.
    """
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 60), Sector(1, 12)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 60), Sector(1, 12)))
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=3)

    # Impose a decaying spectrum, scaled so the wide sector is the negligible one.
    for key, block in T.data.items():
        U, _, Vh = torch.linalg.svd(block, full_matrices=False)
        scale = 1e-3 if block.shape[0] == 60 else 1.0
        s = scale * torch.logspace(0, -3, min(block.shape), dtype=block.dtype)
        T.data[key] = (U * s) @ Vh

    *_, info_exact = svd(T, axis=0, trunc={"nkeep": 14}, requires_info=True)
    *_, info_bare = svd(
        T, axis=0, engine="random", requires_info=True,
        trunc={"nkeep": 14, "seed": 1, "oversample": 0},
    )
    *_, info_oversampled = svd(
        T, axis=0, engine="random", requires_info=True,
        trunc={"nkeep": 14, "seed": 1},
    )

    # Confined to the per-sector target, the budget loses real weight here...
    assert info_bare["discarded_weight"] > 100 * info_exact["discarded_weight"]
    # ...while the default oversampling recovers the exact answer.
    assert info_oversampled["discarded_weight"] == pytest.approx(
        info_exact["discarded_weight"], rel=1e-6
    )


def test_rsvd_discarded_weight_zero_without_truncation():
    """Keeping the whole spectrum discards nothing."""
    T = _make_u1_tensor()

    _, s_exact, _ = svd(T, axis=0)
    total_rank = sum(len(v) for v in s_exact.values())

    *_, info = svd(T, axis=0, trunc={"nkeep": total_rank, "seed": 6}, engine="random",
                   requires_info=True)

    assert info["discarded_weight"] == pytest.approx(0.0, abs=1e-12)


def test_exact_discarded_weight_unchanged():
    """Regression guard: the exact engine's bookkeeping is untouched by this work."""
    T = _make_u1_tensor()

    *_, info = svd(T, axis=0, trunc={"nkeep": 10}, requires_info=True)

    # Recompute independently from the full spectrum.
    _, s_full, _ = svd(T, axis=0)
    values = _sorted_values(s_full)
    expected = float((values[10:] ** 2).sum() / (values ** 2).sum())

    assert info["discarded_weight"] == pytest.approx(expected, rel=1e-10)
    assert info["engine"] == "exact"
    assert info["fallbacks"] == {}


# Dtypes and autograd

def test_rsvd_complex_dtype():
    """The randomized engine handles complex tensors."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 6), Sector(1, 8)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 5), Sector(1, 7)))
    T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=21, dtype=torch.complex128)

    _, s_exact, _ = svd(T, axis=0, trunc={"nkeep": 8})
    _, s_random, _ = svd(T, axis=0, trunc={"nkeep": 8, "seed": 13}, engine="random")

    assert torch.allclose(_sorted_values(s_exact), _sorted_values(s_random),
                          rtol=1e-10, atol=1e-12)


def test_rsvd_is_differentiable():
    """Gradients flow back through the randomized path to the data leaves."""
    group = U1Group()
    idx1 = Index(Direction.OUT, group, sectors=(Sector(0, 8), Sector(1, 10)))
    idx2 = Index(Direction.IN, group, sectors=(Sector(0, 7), Sector(1, 9)))

    # Autograd is disabled globally on import, so it has to be switched back on.
    with torch.enable_grad():
        T = Tensor.random([idx1, idx2], itags=["a", "b"], seed=31, requires_grad=True)

        U, _, _ = svd(T, axis=0, trunc={"nkeep": 6, "seed": 31}, engine="random")
        loss = sum((block.abs() ** 2).sum() for block in U.data.values())
        loss.backward()

    assert any(block.grad is not None and block.grad.abs().sum() > 0
               for block in T.data.values())


# Argument validation

def test_rsvd_requires_nkeep():
    """Without a target rank the randomized engine has nothing to aim at."""
    T = _make_u1_tensor()

    with pytest.raises(ValueError, match="requires trunc"):
        svd(T, axis=0, engine="random")

    with pytest.raises(ValueError, match="requires trunc"):
        svd(T, axis=0, trunc={"thresh": 0.1}, engine="random")


def test_rsvd_mode_requires_nkeep():
    """decomp(mode='RSVD') inherits the same requirement."""
    T = _make_u1_tensor()

    with pytest.raises(ValueError, match="requires trunc"):
        decomp(T, axes=0, mode="RSVD")


def test_invalid_engine_rejected():
    """Only the two documented engines are accepted."""
    T = _make_u1_tensor()

    with pytest.raises(ValueError, match="Invalid engine"):
        svd(T, axis=0, engine="approximate")


def test_invalid_trunc_key_rejected():
    """A misspelled tuning key is an error, never a silent no-op."""
    T = _make_u1_tensor()

    with pytest.raises(ValueError, match="Invalid truncation mode"):
        svd(T, axis=0, trunc={"nkeep": 5, "n_iter": 4}, engine="random")


def test_exact_engine_ignores_randomized_tuning():
    """The randomized keys share the trunc dict but mean nothing to the exact engine."""
    T = _make_u1_tensor()

    _, plain, _ = svd(T, axis=0, trunc={"nkeep": 9})
    _, with_tuning, _ = svd(T, axis=0, trunc={"nkeep": 9, "seed": 1, "niter": 7})

    assert torch.equal(_sorted_values(plain), _sorted_values(with_tuning))


# decomp() integration

def test_decomp_rsvd_matches_svd_mode():
    """RSVD mode produces the same factors as SVD mode under the same nkeep."""
    T = _make_u1_tensor()

    U_svd, S_svd, Vh_svd = decomp(T, axes=0, mode="SVD", trunc={"nkeep": 11})
    U_rsvd, S_rsvd, Vh_rsvd = decomp(T, axes=0, mode="RSVD", trunc={"nkeep": 11, "seed": 2})

    assert U_svd.indices[1].sectors == U_rsvd.indices[1].sectors
    for key, block in S_svd.data.items():
        assert torch.allclose(block, S_rsvd.data[key], rtol=1e-9, atol=1e-11)


def test_decomp_rsvd_multi_axis():
    """RSVD mode survives the merge and unmerge of a multi-axis decomposition."""
    T = _make_u1_tensor()

    U, S, Vh = decomp(T, axes=[0, 1], mode="RSVD", trunc={"nkeep": 10, "seed": 9})

    assert len(U.indices) == 3  # two original axes plus the bond
    assert_charge_neutral(U)
    assert_charge_neutral(Vh)


@pytest.mark.parametrize("axes", [0, [0, 1]])
def test_decomp_rsvd_case_insensitive(axes):
    """Mode strings are normalized before either path branches on them."""
    T = _make_u1_tensor()

    upper = decomp(T, axes=axes, mode="RSVD", trunc={"nkeep": 7, "seed": 1})
    lower = decomp(T, axes=axes, mode="rsvd", trunc={"nkeep": 7, "seed": 1})

    assert len(upper) == len(lower) == 3  # the SVD-shaped (U, S, Vh) triple
    for a, b in zip(upper, lower):
        assert a.indices == b.indices



# The randomized matrix helper in isolation

def test_rsvd_matrix_returns_the_oversampled_triplets():
    """Oversampled triplets are returned, since they cost nothing to keep."""
    mat = torch.randn(40, 25, dtype=torch.float64)

    _, s_bare, _ = _rsvd_matrix(mat, 5, oversample=0)
    _, s_oversampled, _ = _rsvd_matrix(mat, 5, oversample=10)

    assert len(s_bare) == 5
    assert len(s_oversampled) == 15


def test_rsvd_matrix_clamps_rank_to_full_rank():
    """A target beyond the matrix rank is capped, not an error."""
    mat = torch.randn(10, 4, dtype=torch.float64)

    _, s, _ = _rsvd_matrix(mat, 50, oversample=10)

    assert len(s) == 4


def test_rsvd_matrix_full_rank_is_exact():
    """At full rank the randomized route reproduces the exact spectrum."""
    mat = torch.randn(30, 18, dtype=torch.float64)

    _, s_random, _ = _rsvd_matrix(mat, 18, oversample=0)
    s_exact = torch.linalg.svdvals(mat)

    assert torch.allclose(s_random, s_exact, rtol=1e-9, atol=1e-11)
