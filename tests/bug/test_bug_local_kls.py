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


from __future__ import annotations

import torch

from nicole.bug.sweep.bug_sweep import _bug_parity_sweep, bug_two_site, bug_xx_bond_gates, bug_xx_parity_mpos
from nicole.bug.indices import siteinds as make_siteinds
from nicole.bug.ttutils import matrix, normalize_, random_tt, vector


def _to_torch(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.complex128)


def bug_infidelity(v: torch.Tensor, ref: torch.Tensor) -> float:
    ov = torch.vdot(ref, v)
    denom = torch.linalg.norm(v) * torch.linalg.norm(ref)
    if denom == 0:
        return 0.0
    fid = abs(ov / denom) ** 2
    return float(max(0.0, 1.0 - fid))


def bug_rank3_state(sites, seed: int):
    psi = random_tt(sites, maxdim=3, seed=seed)
    normalize_(psi)
    return psi


def bug_vec(psi):
    return _to_torch(vector(psi))


def test_single_parity_sweep_matches_exact_odd_evolution():
    sites = make_siteinds(6, d=2)
    psi = bug_rank3_state(sites, seed=10)
    v0 = bug_vec(psi)
    gates = bug_xx_bond_gates(sites, J=1.0)
    W_odd, _, _ = bug_xx_parity_mpos(sites, J=1.0)
    H_odd = _to_torch(matrix(W_odd))
    dt = 0.05

    from nicole.bug.sweep.init import BUGInfo

    _bug_parity_sweep(psi, gates, BUGInfo(), dt=dt, parity="odd", maxdim=64)
    v1 = bug_vec(psi)
    ref = torch.matrix_exp(-1j * dt * H_odd) @ v0
    assert bug_infidelity(v1, ref) < 1e-11


def test_strang_step_matches_exact_strang_product():
    sites = make_siteinds(6, d=2)
    psi = bug_rank3_state(sites, seed=11)
    v0 = bug_vec(psi)
    gates = bug_xx_bond_gates(sites, J=1.0)
    W_odd, W_even, _ = bug_xx_parity_mpos(sites, J=1.0)
    H_odd = _to_torch(matrix(W_odd))
    H_even = _to_torch(matrix(W_even))
    dt = 0.05

    info = bug_two_site(psi, gates, dt, order="strang", maxdim=64, expv_backend="auto", time_prefactor=-1j)
    v1 = bug_vec(psi)

    U_ref = torch.matrix_exp(-1j * (dt / 2) * H_odd) @ torch.matrix_exp(-1j * dt * H_even) @ torch.matrix_exp(-1j * (dt / 2) * H_odd)
    ref = U_ref @ v0
    assert bug_infidelity(v1, ref) < 1e-11
    assert info.backward_correction_calls == 0


def test_parity_sweep_routes_through_kls_local_candidate(monkeypatch):
    sites = make_siteinds(4, d=2)
    psi = bug_rank3_state(sites, seed=12)
    gates = bug_xx_bond_gates(sites, J=1.0)

    import nicole.bug.sweep.bug_sweep as bug_sweep_mod

    calls: list[int] = []
    original = bug_sweep_mod._faithful_kls_local_bond_candidate

    def wrapped(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(bug_sweep_mod, "_faithful_kls_local_bond_candidate", wrapped)

    from nicole.bug.sweep.init import BUGInfo

    _bug_parity_sweep(psi, gates, BUGInfo(), dt=0.05, parity="odd", maxdim=64, augment=True)
    assert len(calls) == 2
