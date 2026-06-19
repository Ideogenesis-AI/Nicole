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


"""Odd/even bug sweeps built on the local BUG/KLS update.

This module is intentionally user-facing. The public entry points are:

- ``bug_xx_bond_gates`` and ``bug_heisenberg_bond_gates`` for constructing
  local two-site Hamiltonian terms.
- ``bug_two_site`` for applying a Lie or Strang odd/even sweep.

The implementation stays close to the mathematics, but the Python surface is
kept small: a single ``BUGOptions`` object controls the sweep behavior, while
the lower-level helpers focus on one job each.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch
from nicole import Direction, Tensor, einsum

from ..indices import Ix, has_nontrivial_symmetry
from ..krylov import with_expv_backend, with_time_prefactor
from ..linalg import lq, qr
from ..helpers import identity_tensor, make_tensor, tcontract, to_dense
from ..ttutils.ops import OpSum, mpo_from_opsum, op
from ..ttutils.tensortrain import TensorTrain, linkinds, orthogonalize_
from .init import BUGInfo, record_s_step_rank
from .kls import _faithful_kls_local_bond_candidate


@dataclass(frozen=True)
class BUGOptions:
    """Runtime options for an odd/even BUG sweep.

    Parameters
    ----------
    maxdim:
        Maximum bond dimension kept after each local SVD.
    order:
        Trotter ordering. Supported values are ``"strang"`` and
        ``"lie"``.
    augment:
        Whether the local KLS update may grow the bond basis.
    aug_krylov_depth:
        Number of K/L Krylov directions stacked before the
        local S-step.
    lanczos_tol:
        Lanczos termination tolerance used inside each local step.
    lanczos_maxiter:
        Maximum number of Lanczos iterations per local step.
    expv_backend:
        Backend name forwarded to :mod:`nicole.bug.krylov`.
    time_prefactor:
        Global evolution prefactor, usually ``-1j``.
    substep_method:
        Reserved compatibility flag for future non-``expv``
        local substeps.
    matrixfree_sstep:
        Reserved compatibility flag for future matrix-free
        S-step variants.
    """

    maxdim: int = 200
    order: str = "strang"
    augment: bool = True
    aug_krylov_depth: int = 1
    lanczos_tol: float = 1e-15
    lanczos_maxiter: int = 30
    expv_backend: str = "auto"
    time_prefactor: complex = -1j
    substep_method: str = "expv"
    matrixfree_sstep: bool = False


_BUG_OPTION_FIELDS = {field.name for field in BUGOptions.__dataclass_fields__.values()}


def _coerce_bug_options(options: BUGOptions | None = None, **kwargs: Any) -> BUGOptions:
    """Build a :class:`BUGOptions` instance from modern or legacy arguments.

    Parameters
    ----------
    options:
        Existing options object to start from.
    **kwargs:
        Legacy keyword overrides such as ``maxdim=64``.

    Returns
    -------
    A normalized :class:`BUGOptions` instance.
    """
    overrides = {name: kwargs.pop(name) for name in list(kwargs) if name in _BUG_OPTION_FIELDS}
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unknown BUG option(s): {unknown}")
    if options is None:
        return BUGOptions(**overrides)
    return replace(options, **overrides)


def _clone_tensor_with_ixs(tensor: Tensor, ixs: Sequence[Ix]) -> Tensor:
    """Reattach richer :class:`Ix` metadata to a Nicole tensor.

    Parameters
    ----------
    tensor:
        Tensor whose block data should be preserved.
    ixs:
        Replacement indices in the same order as ``tensor.indices``.

    Returns
    -------
    A cloned tensor with the same data and the requested ``Ix`` metadata.
    """
    indices = tuple(ix.nicole() for ix in ixs)
    itags = tuple(ix.itag for ix in ixs)
    data = {tuple(key): block.clone() for key, block in tensor.data.items()}
    intw = None if tensor.intw is None else {tuple(key): bridge.clone() for key, bridge in tensor.intw.items()}
    return Tensor(indices=indices, itags=itags, data=data, intw=intw, dtype=tensor.dtype, label=tensor.label)


def _xx_local_matrix(J: float = 1.0) -> torch.Tensor:
    """Return the dense two-site XX Hamiltonian term.

    Parameters
    ----------
    J:
        XX coupling strength.

    Returns
    -------
    A ``4 x 4`` dense torch matrix in the computational basis.
    """
    sp = op("S+")
    sm = op("S-")
    return (J / 2.0) * (torch.kron(sp, sm) + torch.kron(sm, sp))


def _heisenberg_local_matrix(Jxy: float = 1.0, Jz: float = 1.0) -> torch.Tensor:
    """Return the dense two-site Heisenberg Hamiltonian term.

    Parameters
    ----------
    Jxy:
        XY exchange strength.
    Jz:
        Ising ``Sz Sz`` coupling.

    Returns
    -------
    A ``4 x 4`` dense torch matrix in the computational basis.
    """
    sp = op("S+")
    sm = op("S-")
    sz = op("Sz")
    return (Jxy / 2.0) * (torch.kron(sp, sm) + torch.kron(sm, sp)) + Jz * torch.kron(sz, sz)


def _bond_operator_tensor(sites: Sequence[Ix], local: torch.Tensor) -> list[Tensor]:
    """Lift a dense two-site operator onto every nearest-neighbor bond.

    Parameters
    ----------
    sites:
        Physical site indices.
    local:
        Dense two-site operator of shape ``(d*d, d*d)``.

    Returns
    -------
    One Nicole tensor per bond with itags ``(s_i, s_j, s_i*, s_j*)``.
    """
    gates: list[Tensor] = []
    d = sites[0].dim
    gate_data = local.reshape(d, d, d, d).permute(2, 3, 0, 1).contiguous()

    # Each bond uses the same local matrix but different site labels.
    for bond in range(1, len(sites)):
        left_site = sites[bond - 1]
        right_site = sites[bond]
        gate = make_tensor(
            gate_data,
            [
                Ix(left_site.itag, left_site.dim, direction=Direction.IN, sectors=left_site.sectors, group=left_site.group),
                Ix(right_site.itag, right_site.dim, direction=Direction.IN, sectors=right_site.sectors, group=right_site.group),
                Ix(f"{left_site.itag}*", left_site.dim, direction=Direction.OUT, sectors=left_site.sectors, group=left_site.group),
                Ix(f"{right_site.itag}*", right_site.dim, direction=Direction.OUT, sectors=right_site.sectors, group=right_site.group),
            ],
            dtype=torch.complex128,
        )
        gates.append(gate)
    return gates


def bug_xx_bond_gates(sites: Sequence[Ix], J: float = 1.0) -> list[Tensor]:
    """Construct nearest-neighbor XX Hamiltonian terms.

    Parameters
    ----------
    sites:
        Physical site indices.
    J:
        XX coupling strength.

    Returns
    -------
    A list whose ``bond-1`` entry acts on sites ``(bond, bond+1)``.
    """
    return _bond_operator_tensor(sites, _xx_local_matrix(J=J))


def bug_heisenberg_bond_gates(sites: Sequence[Ix], Jxy: float = 1.0, Jz: float = 1.0) -> list[Tensor]:
    """Construct nearest-neighbor Heisenberg Hamiltonian terms.

    Parameters
    ----------
    sites:
        Physical site indices.
    Jxy:
        XY exchange strength.
    Jz:
        Ising ``Sz Sz`` coupling.

    Returns
    -------
    A list whose ``bond-1`` entry acts on sites ``(bond, bond+1)``.
    """
    return _bond_operator_tensor(sites, _heisenberg_local_matrix(Jxy=Jxy, Jz=Jz))


def bug_xx_parity_mpos(sites: Sequence[Ix], J: float = 1.0):
    """Build dense-check parity MPOs for the XX model.

    Parameters
    ----------
    sites:
        Physical site indices. Only dense sites are currently supported.
    J:
        XX coupling strength.

    Returns
    -------
    ``(W_odd, W_even, W_full)`` as tensor-train operators.
    """
    if has_nontrivial_symmetry(sites):
        raise NotImplementedError("Symmetric checkerboard MPO assembly is not implemented; use direct bond gates.")

    os_odd = OpSum()
    os_even = OpSum()
    os_full = OpSum()
    for bond in range(1, len(sites)):
        coeff = J / 2.0
        target = os_odd if bond % 2 == 1 else os_even
        target += (coeff, [("S+", bond), ("S-", bond + 1)])
        target += (coeff, [("S-", bond), ("S+", bond + 1)])
        os_full += (coeff, [("S+", bond), ("S-", bond + 1)])
        os_full += (coeff, [("S-", bond), ("S+", bond + 1)])
    return mpo_from_opsum(os_odd, sites), mpo_from_opsum(os_even, sites), mpo_from_opsum(os_full, sites)


def _bug_bond_snapshot(psi: TensorTrain, bond: int) -> dict[str, Any]:
    """Build the canonical two-site snapshot used by a local KLS update.

    Parameters
    ----------
    psi:
        Matrix product state to sample.
    bond:
        One-based bond index. Bond ``1`` couples sites ``1`` and ``2``.

    Returns
    -------
    A dictionary containing the bond legs, physical site legs, canonical
    left/right isometries, the bond center ``S0_tens``, and the assembled
    two-site tensor ``theta0_tens``.
    """
    left = psi[bond - 1]
    right = psi[bond]

    # Rebuild each leg as an ``Ix`` so the downstream code retains sector data.
    link_l = Ix(left.itags[0], int(left.indices[0].dim), left.indices[0].direction, left.indices[0].sectors, left.indices[0].group)
    site_l = Ix(left.itags[1], int(left.indices[1].dim), left.indices[1].direction, left.indices[1].sectors, left.indices[1].group)
    link_mid = Ix(left.itags[2], int(left.indices[2].dim), left.indices[2].direction, left.indices[2].sectors, left.indices[2].group)
    site_r = Ix(right.itags[1], int(right.indices[1].dim), right.indices[1].direction, right.indices[1].sectors, right.indices[1].group)
    link_r = Ix(right.itags[2], int(right.indices[2].dim), right.indices[2].direction, right.indices[2].sectors, right.indices[2].group)

    # A QR/LQ pair produces the canonical frame expected by the local KLS step.
    U0_tens, S_left_tens, canon_u0 = qr(left, [link_l, site_l], tag=link_mid.itag, positive=False)
    S_right_tens, V0_tens, canon_v0 = lq(right, [site_r, link_r], tag=link_mid.itag)
    S0_tens = tcontract(S_left_tens, S_right_tens)
    theta0_tens = tcontract(tcontract(U0_tens, S0_tens), V0_tens)

    return {
        "link_l": link_l,
        "site_l": site_l,
        "link_mid": link_mid,
        "site_r": site_r,
        "link_r": link_r,
        "U0_tens": U0_tens,
        "V0_tens": V0_tens,
        "S_left_tens": S_left_tens,
        "S_right_tens": S_right_tens,
        "S0_tens": S0_tens,
        "canon_u0": canon_u0,
        "canon_v0": canon_v0,
        "theta0_tens": theta0_tens,
    }


def _bug_local_effective_hamiltonian(gate: Tensor, bond_data: dict[str, Any]) -> Tensor:
    """Dress a two-site gate with trivial bond environments.

    Parameters
    ----------
    gate:
        Bare two-site Hamiltonian term acting on the physical site legs.
    bond_data:
        Output of :func:`_bug_bond_snapshot`.

    Returns
    -------
    The same local operator, but extended with identities on the left and
    right bond legs.
    """
    left_id = identity_tensor(bond_data["link_l"], f"{bond_data['link_l'].itag}*")
    right_id = identity_tensor(bond_data["link_r"], f"{bond_data['link_r'].itag}*")

    # Nicole preserves dtype blockwise, so we promote explicitly for evolution.
    left_id.data = {key: block.to(dtype=torch.complex128) for key, block in left_id.data.items()}
    right_id.data = {key: block.to(dtype=torch.complex128) for key, block in right_id.data.items()}
    return tcontract(tcontract(left_id, gate), right_id)


def _bond_dense_matrix(gate: Tensor) -> torch.Tensor:
    """Convert a local gate tensor into a dense matrix.

    Parameters
    ----------
    gate:
        Nicole tensor with ordering ``(s_i, s_j, s_i*, s_j*)``.

    Returns
    -------
    A dense ``(d*d, d*d)`` torch matrix.
    """
    dense = to_dense(gate, list(gate.itags)).to(torch.complex128)
    local = dense.permute(2, 3, 0, 1).contiguous()
    d = dense.shape[0]
    return local.reshape(d * d, d * d)


def _lift_gate(local: torch.Tensor, bond: int, n: int, d: int = 2) -> torch.Tensor:
    """Embed a two-site dense operator into a full ``n``-site Hilbert space.

    Parameters
    ----------
    local:
        Two-site dense matrix.
    bond:
        One-based bond index.
    n:
        Number of sites.
    d:
        On-site Hilbert-space dimension.

    Returns
    -------
    A dense ``d**n x d**n`` matrix acting on the full chain.
    """
    eye = torch.eye(d, dtype=torch.complex128)
    factors: list[torch.Tensor] = []
    for site in range(1, n + 1):
        if site == bond:
            factors.append(local)
        elif site == bond + 1:
            continue
        else:
            factors.append(eye)

    full = factors[0]
    for factor in factors[1:]:
        full = torch.kron(full, factor)
    return full


def _apply_gate_named(gate: Tensor, theta: Tensor, site_l_tag: str, site_r_tag: str) -> Tensor:
    """Apply a two-site gate to a bond tensor while restoring the original site tags.

    Parameters
    ----------
    gate:
        Two-site Nicole tensor.
    theta:
        Four-leg bond tensor ``(link_l, site_l, site_r, link_r)``.
    site_l_tag:
        Original left-site tag in ``theta``.
    site_r_tag:
        Original right-site tag in ``theta``.

    Returns
    -------
    The evolved bond tensor with the original site tags restored.
    """
    out = einsum("LRlr,aLRb->alrb", gate, theta)
    out.retag({f"{site_l_tag}*": site_l_tag, f"{site_r_tag}*": site_r_tag})
    return out


def _singular_values_from_diag_tensor(S: Tensor) -> torch.Tensor:
    """Extract singular values from a block-diagonal Nicole tensor.

    Parameters
    ----------
    S:
        Diagonal tensor returned by Nicole's SVD.

    Returns
    -------
    A flattened torch vector containing every block diagonal entry.
    """
    diagonals = [torch.diagonal(block) for block in S.data.values()]
    if not diagonals:
        return torch.empty((0,), dtype=torch.complex128)
    return torch.cat(diagonals)


def _parity_bonds(num_sites: int, parity: str) -> list[int]:
    """Return the one-based bond list for one parity sweep.

    Parameters
    ----------
    num_sites:
        Number of MPS sites.
    parity:
        Either ``"odd"`` or ``"even"``.

    Returns
    -------
    A list of one-based bond indices.
    """
    if parity == "odd":
        return list(range(1, num_sites, 2))
    if parity == "even":
        return list(range(2, num_sites, 2))
    raise ValueError("parity must be 'odd' or 'even'")


def _bug_parity_sweep(
    psi: TensorTrain,
    gates: Sequence[Tensor],
    *args,
    options: BUGOptions | None = None,
    **kwargs: Any,
) -> TensorTrain:
    """Apply one odd or even sweep of local BUG/KLS updates.

    Parameters
    ----------
    psi:
        State evolved in place.
    gates:
        Local bond terms; ``gates[bond - 1]`` acts on ``bond``.
    *args:
        Optional legacy positional tail ``(info, dt, parity)``.
    options:
        Optional :class:`BUGOptions` instance.
    **kwargs:
        Legacy keyword overrides for ``options`` plus keyword forms
        of ``info``, ``dt``, and ``parity``.

    Returns
    -------
    The updated ``psi`` object.
    """
    if args:
        if len(args) > 3:
            raise TypeError("_bug_parity_sweep accepts at most (info, dt, parity) after (psi, gates).")
        for name, value in zip(("info", "dt", "parity"), args):
            if name in kwargs:
                raise TypeError(f"Provide {name} either positionally or by keyword, not both.")
            kwargs[name] = value
    try:
        info = kwargs.pop("info")
        dt = kwargs.pop("dt")
        parity = kwargs.pop("parity")
    except KeyError as exc:
        raise TypeError("Missing _bug_parity_sweep input 'info', 'dt', or 'parity'.") from exc

    options = _coerce_bug_options(options, **kwargs)
    sweep_label = f"{parity}_bug"

    for bond in _parity_bonds(len(psi), parity):
        # Bring the orthogonality center onto the active bond before snapshotting.
        orthogonalize_(psi, bond)
        bond_data = _bug_bond_snapshot(psi, bond)
        candidate = _faithful_kls_local_bond_candidate(
            bond_data,
            gate=gates[bond - 1],
            dt=dt,
            maxdim=options.maxdim,
            augment=options.augment,
            aug_krylov_depth=options.aug_krylov_depth,
            lanczos_tol=options.lanczos_tol,
            lanczos_maxiter=options.lanczos_maxiter,
        )

        psi[bond - 1] = candidate["left_core"]
        psi[bond] = candidate["right_core"]
        info.aug_dims_k.append(candidate["n_new_k"])
        info.aug_dims_l.append(candidate["n_new_l"])
        info.lanczos_numops.append(candidate["numops_s"])
        record_s_step_rank(info, sweep_label, bond, candidate["keep"], candidate["svals"])
    return psi


def bug_two_site(
    psi: TensorTrain,
    gates: Sequence[Tensor],
    dt: complex,
    *,
    options: BUGOptions | None = None,
    **kwargs: Any,
) -> BUGInfo:
    """Advance one Lie or Strang odd/even BUG step.

    Parameters
    ----------
    psi:
        State evolved in place.
    gates:
        Local bond Hamiltonian terms.
    dt:
        Physical timestep for the full sweep.
    options:
        Optional :class:`BUGOptions` instance.
    **kwargs:
        Legacy keyword overrides such as ``maxdim=64`` or
        ``order="strang"``.

    Returns
    -------
    A :class:`BUGInfo` object describing the observed bond growth and S-step
    truncation behavior.
    """
    options = _coerce_bug_options(options, **kwargs)
    info = BUGInfo()
    info.bond_dims_before = [ix.dim for ix in linkinds(psi)][1:-1]
    backend = "native_hermitian_lanczos" if options.expv_backend == "auto" else options.expv_backend

    def sweep(tau: complex, parity: str) -> None:
        """Run one parity sweep with the already-selected bug options."""
        _bug_parity_sweep(psi, gates, info, dt=tau, parity=parity, options=options)

    with with_expv_backend(backend), with_time_prefactor(options.time_prefactor):
        if options.order == "strang":
            sweep(dt / 2, "odd")
            sweep(dt, "even")
            sweep(dt / 2, "odd")
        elif options.order == "lie":
            sweep(dt, "odd")
            sweep(dt, "even")
        else:
            raise ValueError("order must be 'strang' or 'lie'")

    info.bond_dims_after = [ix.dim for ix in linkinds(psi)][1:-1]
    return info
