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


"""Property-based tests for invariants that hold for every tensor.

The topical test modules pin down specific inputs. These tests instead state
invariants that must survive any admissible combination of symmetry group,
index directions, sector layout, order and dtype, and let Hypothesis search
for a counterexample.

Example counts are capped deliberately: every example builds dense blocks and
Clebsch-Gordan data, so the suite trades breadth per run for a bounded runtime
and accumulates coverage across runs through Hypothesis' example database.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pytest
import torch
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nicole import (
    Direction,
    Index,
    ProductGroup,
    SU2Group,
    Sector,
    Tensor,
    U1Group,
    Z2Group,
    allclose,
    conj,
    contract,
    decomp,
    deserialize,
    identity,
    oplus,
    permute,
    serialize,
    transpose,
)
from ..utils import assert_charge_neutral, assert_dtype_coherent


# Each entry pairs a group factory with the charges it admits. Charge pools are
# kept small: the number of admissible blocks grows combinatorially with order.
_GROUP_CHARGES = {
    "U1": (U1Group, (-1, 0, 1)),
    "Z2": (Z2Group, (0, 1)),
    "SU2": (SU2Group, (0, 1, 2)),
    "U1xSU2": (
        lambda: ProductGroup([U1Group(), SU2Group()]),
        ((0, 0), (0, 1), (1, 0), (1, 1)),
    ),
}

_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)


@st.composite
def indices(draw, group_name: str, order: int) -> List[Index]:
    """Draw a list of `order` indices sharing one symmetry group."""
    factory, charge_pool = _GROUP_CHARGES[group_name]
    group = factory()

    drawn: List[Index] = []
    for _ in range(order):
        direction = draw(st.sampled_from([Direction.IN, Direction.OUT]))
        charges = draw(
            st.lists(st.sampled_from(charge_pool), min_size=1, max_size=3, unique=True)
        )
        dims = draw(st.lists(st.integers(1, 3), min_size=len(charges), max_size=len(charges)))
        sectors = tuple(Sector(charge, dim) for charge, dim in zip(charges, dims))
        drawn.append(Index(direction, group, sectors))
    return drawn


@st.composite
def tensors(draw, min_order: int = 2, max_order: int = 4) -> Tensor:
    """Draw a random non-empty tensor over one of the supported groups."""
    group_name = draw(st.sampled_from(sorted(_GROUP_CHARGES)))
    order = draw(st.integers(min_order, max_order))
    drawn_indices = draw(indices(group_name, order))
    dtype = draw(st.sampled_from(_DTYPES))
    seed = draw(st.integers(0, 2**16))

    itags = [f"i{position}" for position in range(order)]
    tensor = Tensor.random(drawn_indices, dtype=dtype, seed=seed, itags=itags)
    # A charge layout admitting no block makes every invariant below vacuous.
    assume(tensor.data)
    return tensor


@st.composite
def tensor_and_permutation(draw) -> Tuple[Tensor, List[int]]:
    """Draw a tensor together with a permutation of its axes."""
    tensor = draw(tensors())
    order = draw(st.permutations(range(len(tensor.indices))))
    return tensor, list(order)


def _invert(order: Sequence[int]) -> List[int]:
    """Return the permutation that undoes `order`."""
    inverse = [0] * len(order)
    for position, axis in enumerate(order):
        inverse[axis] = position
    return inverse


def _tolerance(dtype: torch.dtype) -> dict:
    """Return comparison tolerances appropriate to a dtype.

    Single precision carries about seven decimal digits, so the defaults of
    `torch.allclose` sit below its round-off for an SU(2) recoupling and would
    report a spurious difference.
    """
    if dtype in (torch.float32, torch.complex64):
        return {"rtol": 1e-4, "atol": 1e-5}
    return {"rtol": 1e-7, "atol": 1e-10}


# Hypothesis' function-scoped-fixture check does not apply here, and tensor
# construction is slow enough that the default deadline is not meaningful.
_SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# ------------------------------------------------------------
#   Structural invariants
# ------------------------------------------------------------

@_SETTINGS
@given(tensors())
def test_random_tensors_are_charge_neutral(tensor: Tensor):
    """Every block of a randomly built tensor conserves charge."""
    assert_charge_neutral(tensor)


@_SETTINGS
@given(tensors())
def test_random_tensors_have_a_coherent_dtype(tensor: Tensor):
    """Blocks and intertwiner weights agree with the declared dtype."""
    assert_dtype_coherent(tensor)


@_SETTINGS
@given(tensors())
def test_clone_equals_the_original(tensor: Tensor):
    """A clone compares equal to its source and is a distinct object."""
    copy = tensor.clone()

    assert copy is not tensor
    assert copy == tensor
    assert allclose(copy, tensor, **_tolerance(tensor.dtype))


# ------------------------------------------------------------
#   Involutions and round trips
# ------------------------------------------------------------

@_SETTINGS
@given(tensor_and_permutation())
def test_permuting_by_the_inverse_order_restores_the_tensor(case):
    """Permuting axes and then applying the inverse permutation is the identity."""
    tensor, order = case
    restored = permute(permute(tensor, order), _invert(order))

    assert restored.itags == tensor.itags
    assert restored.indices == tensor.indices
    assert allclose(restored, tensor, **_tolerance(tensor.dtype))


@_SETTINGS
@given(tensors())
def test_transpose_is_an_involution(tensor: Tensor):
    """Reversing every axis twice returns the original tensor."""
    restored = transpose(transpose(tensor))

    assert restored.itags == tensor.itags
    assert allclose(restored, tensor, **_tolerance(tensor.dtype))


@_SETTINGS
@given(tensors())
def test_conjugation_is_an_involution(tensor: Tensor):
    """Conjugating twice restores both the data and the index directions."""
    restored = conj(conj(tensor))

    assert restored.indices == tensor.indices
    assert allclose(restored, tensor, **_tolerance(tensor.dtype))


@_SETTINGS
@given(tensors())
def test_serialization_round_trip_preserves_the_tensor(tensor: Tensor):
    """A serialize/deserialize round trip reproduces the tensor exactly."""
    restored = deserialize(serialize(tensor))

    assert restored == tensor
    assert restored.itags == tensor.itags
    if tensor.intw is None:
        assert restored.intw is None
    else:
        assert set(restored.intw) == set(tensor.intw)


# ------------------------------------------------------------
#   Norm and scale invariants
# ------------------------------------------------------------

@_SETTINGS
@given(tensor_and_permutation())
def test_permutation_preserves_the_norm(case):
    """Reordering axes moves data without changing the Frobenius norm."""
    tensor, order = case
    permuted = permute(tensor, order)

    assert permuted.norm() == pytest.approx(tensor.norm(), rel=1e-5)


@_SETTINGS
@given(tensors())
def test_trimming_zero_blocks_preserves_the_norm(tensor: Tensor):
    """Dropping negligible blocks leaves the norm of a random tensor intact."""
    before = tensor.norm()
    trimmed = tensor.clone()
    trimmed.trim_zero_blocks()

    assert trimmed.norm() == pytest.approx(before, rel=1e-5)


# ------------------------------------------------------------
#   Operation invariants
# ------------------------------------------------------------

@_SETTINGS
@given(tensors())
def test_direct_sum_with_itself_doubles_every_sector(tensor: Tensor):
    """A direct sum along one axis adds the sector dimensions of both operands."""
    axis = 0
    summed = oplus(tensor, tensor.clone(), axes=axis)

    merged_dims = summed.indices[axis].sector_dim_map()
    for charge, dim in tensor.indices[axis].sector_dim_map().items():
        assert merged_dims[charge] == 2 * dim
    assert_charge_neutral(summed)


@_SETTINGS
@given(tensors())
def test_contracting_with_an_identity_leaves_the_tensor_unchanged(tensor: Tensor):
    """Contracting an axis against the identity on that index is a no-op."""
    axis = 0
    index = tensor.indices[axis]
    ident = identity(index.flip(), dtype=tensor.dtype, itags=(tensor.itags[axis], "_pad_"))
    contracted = contract(tensor, ident, axes=(axis, 0))
    contracted.retag({"_pad_": tensor.itags[axis]})

    tag_to_position = {tag: i for i, tag in enumerate(contracted.itags)}
    contracted.permute([tag_to_position[tag] for tag in tensor.itags], in_place=True)

    assert allclose(contracted, tensor, **_tolerance(tensor.dtype))


@_SETTINGS
@given(tensors(min_order=2, max_order=3))
def test_svd_reconstructs_the_tensor(tensor: Tensor):
    """A full SVD along the first axis multiplies back to the original tensor."""
    U, S, Vh = decomp(tensor, axes=0, mode="SVD")
    reconstructed = contract(U, contract(S, Vh))

    tag_to_position = {tag: i for i, tag in enumerate(reconstructed.itags)}
    reconstructed.permute([tag_to_position[tag] for tag in tensor.itags], in_place=True)

    assert allclose(reconstructed, tensor, rtol=1e-3, atol=1e-4)
