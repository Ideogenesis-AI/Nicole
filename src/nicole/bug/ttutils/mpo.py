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


"""Matrix-product-operator helpers for Nicole-based tensor trains.

This module keeps two responsibilities together because they are tightly
coupled in the BUG codebase:

- constructing and manipulating tensor-train operators; and
- building dense effective Hamiltonians for local Krylov updates.

The implementation remains intentionally conservative. Many routines still take
the dense fallback path because the surrounding tests validate correctness more
than asymptotic performance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from nicole import Direction, Tensor

from ..indices import Ix, has_nontrivial_symmetry
from ..helpers import flatten_fortran, make_tensor, tcontract, to_dense
from .tensortrain import TensorTrain, siteinds as tt_siteinds, tensor_train_from_vector, vector

__all__ = [
    "TensorTrainOperator",
    "contract",
    "identity_op",
    "matrix",
    "tensor_train_operator_from_arrays",
    "tto_direct_sum",
]


@dataclass
class TensorTrainOperator:
    """Ordered list of rank-4 Nicole MPO cores.

    Parameters
    ----------
    cores:
        MPO core tensors ordered from left to right.
    """

    cores: list[Tensor]

    def __len__(self) -> int:
        """Return the number of sites represented by the MPO."""

        return len(self.cores)

    def __getitem__(self, idx: int) -> Tensor:
        """Return one core at a zero-based Python index."""

        return self.cores[idx]

    def __setitem__(self, idx: int, value: Tensor) -> None:
        """Replace one core tensor in-place."""

        self.cores[idx] = value

    def copy(self) -> "TensorTrainOperator":
        """Return a deep copy of the MPO."""

        return TensorTrainOperator([core.clone() for core in self.cores])


def _site_ix(core: Tensor, axis: int) -> Ix:
    """Extract one index from an MPO core.

    Parameters
    ----------
    core:
        Nicole tensor carrying the desired leg.
    axis:
        Integer axis in the tensor.

    Returns
    -------
    The matching :class:`Ix`.
    """

    idx = core.indices[axis]
    return Ix(core.itags[axis], int(idx.dim), idx.direction, idx.sectors, idx.group)


def tensor_train_operator_from_arrays(sites: Sequence[Ix], arrays: Sequence[torch.Tensor | object]) -> TensorTrainOperator:
    """Construct an MPO from explicit rank-4 arrays.

    Parameters
    ----------
    sites:
        Physical site indices.
    arrays:
        Core arrays ordered as ``(bond_left, ket, bra, bond_right)``.

    Returns
    -------
    A :class:`TensorTrainOperator` with matching physical legs.
    """

    if len(sites) != len(arrays):
        raise ValueError("Length of sites and arrays must match.")

    cores: list[Tensor] = []
    for k, (site, arr_like) in enumerate(zip(sites, arrays), start=1):
        arr = torch.as_tensor(arr_like, dtype=torch.complex128)
        if arr.ndim != 4:
            raise ValueError(f"MPO core {k} must be rank-4, got shape {tuple(arr.shape)}.")

        wl, d_ket, d_bra, wr = (int(dim) for dim in arr.shape)
        if d_ket != site.dim or d_bra != site.dim:
            raise ValueError(
                f"MPO core {k} physical dims ({d_ket},{d_bra}) incompatible with site dim {site.dim}."
            )

        w_left = Ix(f"w{k-1}", wl, Direction.IN)
        ket = Ix(site.itag, site.dim, Direction.IN, site.sectors, site.group)
        bra = Ix(f"{site.itag}*", site.dim, Direction.OUT, site.sectors, site.group)
        w_right = Ix(f"w{k}", wr, Direction.OUT)
        cores.append(make_tensor(arr, [w_left, ket, bra, w_right], dtype=torch.complex128))

    return TensorTrainOperator(cores)


def siteinds(W: TensorTrainOperator, plev: int = 0):
    """Return bra or ket physical indices of an MPO.

    Parameters
    ----------
    W:
        Tensor-train operator.
    plev:
        ``0`` for bra/output legs, ``1`` for ket/input legs.

    Returns
    -------
    List of physical site indices at the requested prime level.
    """

    if plev == 0:
        return [_site_ix(core, 2) for core in W.cores]
    if plev == 1:
        return [_site_ix(core, 1) for core in W.cores]
    raise ValueError("plev must be 0 (bra/output) or 1 (ket/input)")


def matrix(W: TensorTrainOperator) -> torch.Tensor:
    """Fully contract an MPO into a dense matrix.

    Parameters
    ----------
    W:
        Tensor-train operator.

    Returns
    -------
    Dense matrix in ``(bra, ket)`` ordering.
    """

    contracted = W[0]
    for core in W.cores[1:]:
        contracted = tcontract(contracted, core)

    left_tag = W[0].itags[0]
    right_tag = W[-1].itags[3]
    bra_tags = [ix.itag for ix in siteinds(W, plev=0)]
    ket_tags = [ix.itag for ix in siteinds(W, plev=1)]
    dense = to_dense(contracted, [left_tag, *bra_tags, *ket_tags, right_tag]).squeeze(0).squeeze(-1)
    d = int(math.prod(ix.dim for ix in siteinds(W, plev=1)))
    return flatten_fortran(dense).reshape(d, d)


def identity_op(sites: Sequence[Ix]) -> TensorTrainOperator:
    """Build the identity MPO on the supplied sites.

    Parameters
    ----------
    sites:
        Physical site indices.

    Returns
    -------
    Identity operator represented as an MPO.
    """

    arrays = [torch.eye(site.dim, dtype=torch.complex128).transpose(0, 1).unsqueeze(0).unsqueeze(-1) for site in sites]
    return tensor_train_operator_from_arrays(sites, arrays)


def contract(W: TensorTrainOperator, psi: TensorTrain) -> TensorTrain:
    """Apply an MPO to an MPS using the dense fallback route.

    Parameters
    ----------
    W:
        Tensor-train operator.
    psi:
        Tensor-train state.

    Returns
    -------
    The state ``W |psi>`` as a tensor train.
    """

    out_vec = matrix(W) @ vector(psi)
    return tensor_train_from_vector(out_vec, tt_siteinds(psi))


def replacelinks_op(W: TensorTrainOperator) -> TensorTrainOperator:
    """Retag every internal MPO bond so two MPOs can coexist safely.

    Parameters
    ----------
    W:
        Operator to copy and retag.

    Returns
    -------
    A copied operator with fresh internal bond tags.
    """

    out = W.copy()
    for j in range(1, len(out)):
        old = out[j - 1].itags[3]
        new = f"{old}#{j}"
        out[j - 1].retag({old: new})
        out[j].retag({old: new})
    return out


def tto_direct_sum(A: TensorTrainOperator, B: TensorTrainOperator) -> TensorTrainOperator:
    """Build the block-diagonal direct sum of two dense MPOs.

    Parameters
    ----------
    A:
        First operator.
    B:
        Second operator.

    Returns
    -------
    MPO representing ``A ⊕ B``.
    """

    if len(A) != len(B):
        raise ValueError("MPO lengths must match for direct sum.")
    if has_nontrivial_symmetry(siteinds(A, plev=1)) or has_nontrivial_symmetry(siteinds(B, plev=1)):
        raise NotImplementedError("Direct-sum MPO assembly is currently supported only for dense/trivial site indices.")

    B_r = replacelinks_op(B.copy())
    n = len(A)
    sites = [_site_ix(core, 1) for core in A.cores]

    arrays: list[torch.Tensor] = []
    for k in range(n):
        a = to_dense(A[k], list(A[k].itags))
        b = to_dense(B_r[k], list(B_r[k].itags))
        a_wl, d, _, a_wr = a.shape
        b_wl, _, _, b_wr = b.shape

        # Each core is assembled in dense form and then lifted back into Nicole.
        if k == 0:
            out = torch.zeros((1, d, d, a_wr + b_wr), dtype=torch.complex128)
            out[0, :, :, :a_wr] = a[0]
            out[0, :, :, a_wr:] = b[0]
        elif k == n - 1:
            out = torch.zeros((a_wl + b_wl, d, d, 1), dtype=torch.complex128)
            out[:a_wl, :, :, 0] = a[:, :, :, 0]
            out[a_wl:, :, :, 0] = b[:, :, :, 0]
        else:
            out = torch.zeros((a_wl + b_wl, d, d, a_wr + b_wr), dtype=torch.complex128)
            out[:a_wl, :, :, :a_wr] = a
            out[a_wl:, :, :, a_wr:] = b
        arrays.append(out)

    return tensor_train_operator_from_arrays(sites, arrays)
