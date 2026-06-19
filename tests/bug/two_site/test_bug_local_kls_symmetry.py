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

from nicole.bug import product_tt, bug_heisenberg_bond_gates
from nicole.bug.sweep.kls import _faithful_kls_local_bond_candidate
from nicole.bug.sweep.bug_sweep import _bug_bond_snapshot
from nicole.bug.krylov import linear_substep, with_expv_backend
from nicole.bug.indices import siteinds as make_siteinds
from nicole.bug.helpers import reshape_fortran
from nicole.bug.ttutils.tensortrain import vector


def test_local_exp_solver_matches_dense_exp_dt_h_x_for_both_backends():
    n = 16
    torch.manual_seed(0)
    a = torch.randn((n, n), dtype=torch.complex128)
    h = a + a.conj().T
    x = torch.randn((n,), dtype=torch.complex128)

    for dt in (0.05, -0.05, -0.05j, 0.05j):
        ref = torch.matrix_exp(dt * h) @ x
        y, _ = linear_substep(
            h,
            dt,
            x,
            method="expv",
            lanczos_tol=1e-14,
            lanczos_maxiter=n,
            restart=1,
        )
        assert torch.linalg.norm(y - ref) / torch.linalg.norm(ref) < 1e-10

        with with_expv_backend("native_hermitian_lanczos"):
            yn, _ = linear_substep(
                h,
                dt,
                x,
                method="expv",
                lanczos_tol=1e-14,
                lanczos_maxiter=n,
                restart=1,
            )
        assert torch.linalg.norm(yn - ref) / torch.linalg.norm(ref) < 1e-10


def test_symmetric_local_kls_candidate_preserves_site_sector_tables():
    psi = product_tt("uudd", symmetry="u1")
    gates = bug_heisenberg_bond_gates(make_siteinds(4, symmetry="u1"))
    snap = _bug_bond_snapshot(psi, 2)

    candidate = _faithful_kls_local_bond_candidate(
        snap,
        gate=gates[1],
        dt=0.1,
        maxdim=16,
        augment=True,
        aug_krylov_depth=2,
    )
    psi[1] = candidate["left_core"]
    psi[2] = candidate["right_core"]

    assert vector(psi).numel() == 16
    assert candidate["n_new_k"] == 1
    assert candidate["n_new_l"] == 1
    assert candidate["keep"] == 2
    assert len(candidate["left_core"].indices[1].sectors) == 2
    assert len(candidate["right_core"].indices[1].sectors) == 2
    assert [(sec.charge, sec.dim) for sec in candidate["left_core"].indices[2].sectors] == [(-2, 1), (0, 1)]

    dense = reshape_fortran(vector(psi), [2] * 4)
    prob = dense.abs() ** 2
    mz_total = sum(0.5 * (prob.select(axis, 0).sum().item() - prob.select(axis, 1).sum().item()) for axis in range(4))
    assert abs(mz_total) < 1e-10
