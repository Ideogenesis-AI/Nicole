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


"""Local operator builders and small operator-sum utilities.

This module intentionally stays simple: it provides a tiny spin-1/2 operator
table, an ``OpSum`` container, and a direct ``OpSum -> MPO`` conversion path
used by dense regression tests and small examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from ..indices import Ix, has_nontrivial_symmetry

__all__ = ["OpSum", "mpo_from_opsum", "op"]


def op(name: str, site_or_dim: int | Ix = 2) -> torch.Tensor:
    """Return a dense spin-1/2 local operator matrix by canonical name.

    Parameters
    ----------
    name:
        Operator label such as ``"S+"``, ``"S-"``, ``"Sz"``, or ``"Id"``.
    site_or_dim:
        Either a site index wrapper or the local Hilbert-space
        dimension.

    Returns
    -------
    A dense ``2 x 2`` complex torch matrix.
    """
    dim = site_or_dim.dim if isinstance(site_or_dim, Ix) else int(site_or_dim)
    if dim != 2:
        raise NotImplementedError("Only spin-1/2 local operators are supported.")

    if name == "S+":
        return torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)
    if name == "S-":
        return torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.complex128)
    if name == "Sz":
        return torch.tensor([[0.5, 0.0], [0.0, -0.5]], dtype=torch.complex128)
    if name == "Id":
        return torch.eye(2, dtype=torch.complex128)
    raise ValueError(f"Unknown operator name: {name}")


@dataclass
class OpSum:
    """Lightweight container for operator-sum terms.

    Parameters
    ----------
    terms:
        Stored terms in the form ``(coeff, [(op_name, site), ...])``.
    """

    terms: list[tuple[complex, list[tuple[str, int]]]] = field(default_factory=list)

    def append_term(self, coeff: complex, factors: Sequence[tuple[str, int]]) -> None:
        """Append one operator-sum term.

        Parameters
        ----------
        coeff:
            Scalar prefactor.
        factors:
            Sequence of ``(operator_name, site_number)`` pairs.

        Returns
        -------
        ``None``.
        """
        self.terms.append((complex(coeff), list(factors)))

    def __iadd__(self, item):
        """Append one term using ``osum += (coeff, factors)`` syntax.

        Parameters
        ----------
        item:
            ``(coeff, factors)`` tuple.

        Returns
        -------
        ``self``.
        """
        coeff, factors = item
        self.append_term(coeff, factors)
        return self


def _term_local_matrix(site: int, factors: Sequence[tuple[str, int]], dim: int) -> torch.Tensor:
    """Assemble the single-site factor contributed by one OpSum term.

    Parameters
    ----------
    site:
        One-based site number being assembled.
    factors:
        Operator factors present in the term.
    dim:
        Local Hilbert-space dimension.

    Returns
    -------
    The dense matrix acting on ``site`` for that term.
    """
    mat = torch.eye(dim, dtype=torch.complex128)
    for name, position in factors:
        if position == site:
            mat = op(name, dim) @ mat
    return mat


def mpo_from_opsum(osum: OpSum, sites: Sequence[Ix]):
    """Convert an :class:`OpSum` into an MPO by direct-summing product terms.

    Parameters
    ----------
    osum:
        Operator-sum container.
    sites:
        Physical site indices.

    Returns
    -------
    A :class:`nicole.bug.ttutils.mpo.TensorTrainOperator`.
    """
    from .mpo import TensorTrainOperator, tensor_train_operator_from_arrays, tto_direct_sum

    if has_nontrivial_symmetry(sites):
        for _, factors in osum.terms:
            if any(name in {"S+", "S-"} for name, _ in factors):
                raise NotImplementedError(
                    "U(1)-symmetric OpSum->MPO for charge-changing local factors is not yet encoded. "
                    "Use direct bond Hamiltonian tensors for symmetric real-time evolution."
                )

    total: TensorTrainOperator | None = None
    num_sites = len(sites)
    for coeff, factors in osum.terms:
        arrays: list[torch.Tensor] = []
        for site in range(1, num_sites + 1):
            local = _term_local_matrix(site, factors, sites[site - 1].dim)
            core = local.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
            arrays.append(core)

        # Multiply the scalar prefactor into one core so the term stays an MPO.
        arrays[0] = coeff * arrays[0]
        term_mpo = tensor_train_operator_from_arrays(sites, arrays)
        total = term_mpo if total is None else tto_direct_sum(total, term_mpo)

    if total is None:
        arrays = [torch.zeros((1, site.dim, site.dim, 1), dtype=torch.complex128) for site in sites]
        return tensor_train_operator_from_arrays(sites, arrays)
    return total
