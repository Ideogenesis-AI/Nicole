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

from nicole.bug.sweep.bug_sweep import bug_two_site, bug_xx_bond_gates, bug_xx_parity_mpos
from nicole.bug.indices import siteinds as make_siteinds
from nicole.bug.ttutils import matrix, normalize_, random_tt, vector


def _to_torch(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.complex128)


def bug_norm_error(v: torch.Tensor, ref: torch.Tensor) -> float:
    return float(abs(torch.linalg.norm(v) - torch.linalg.norm(ref)))


def bug_rank3_state(sites, seed: int):
    psi = random_tt(sites, maxdim=3, seed=seed)
    normalize_(psi)
    return psi


def bug_vec(psi):
    return _to_torch(vector(psi))


def _run(order: str, dt: float):
    sites = make_siteinds(6, d=2)
    psi = bug_rank3_state(sites, seed=42)
    gates = bug_xx_bond_gates(sites, J=1.0)
    W_odd, W_even, _ = bug_xx_parity_mpos(sites, J=1.0)
    H = _to_torch(matrix(W_odd)) + _to_torch(matrix(W_even))
    T = 0.5
    steps = int(round(T / dt))
    v0 = bug_vec(psi)
    for _ in range(steps):
        bug_two_site(psi, gates, dt, order=order, maxdim=64, expv_backend="auto", time_prefactor=-1j)
    v = bug_vec(psi)
    ref = torch.matrix_exp(-1j * T * H) @ v0
    err = torch.linalg.norm(v - ref) / torch.linalg.norm(ref)
    nerr = bug_norm_error(v, ref)
    return float(err), nerr


def test_trotter_convergence_orders():
    dts = (0.1, 0.05, 0.025)
    strang_errs = []
    lie_errs = []
    strang_nerrs = []
    lie_nerrs = []

    for dt in dts:
        e_s, n_s = _run("strang", dt)
        e_l, n_l = _run("lie", dt)
        strang_errs.append(e_s)
        lie_errs.append(e_l)
        strang_nerrs.append(n_s)
        lie_nerrs.append(n_l)

    # Monotone decreasing with dt
    assert strang_errs[2] < strang_errs[1] < strang_errs[0]
    assert lie_errs[2] < lie_errs[1] < lie_errs[0]

    # Approximate order ratios.
    r_s1 = strang_errs[0] / strang_errs[1]
    r_s2 = strang_errs[1] / strang_errs[2]
    assert 2.5 <= r_s1 <= 8.0
    assert 2.5 <= r_s2 <= 8.0

    r_l1 = lie_errs[0] / lie_errs[1]
    r_l2 = lie_errs[1] / lie_errs[2]
    assert 1.6 <= r_l1 <= 3.5
    assert 1.6 <= r_l2 <= 3.5

    assert max(strang_errs) < max(lie_errs)
    assert max(strang_nerrs) <= 1e-10
    assert max(lie_nerrs) <= 1e-10
