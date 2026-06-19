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

from nicole.bug import product_tt, bug_heisenberg_bond_gates
from nicole.bug.sweep import bug_two_site
from nicole.bug.indices import siteinds as make_siteinds
from nicole.bug.helpers import reshape_fortran
from nicole.bug.ttutils.tensortrain import maxlinkdim, vector


def _mz_profile(psi):
    dense = reshape_fortran(vector(psi), [2] * len(psi))
    prob = dense.abs() ** 2
    profile = []
    for axis in range(len(psi)):
        up = prob.select(axis, 0).sum().item()
        down = prob.select(axis, 1).sum().item()
        profile.append(0.5 * (up - down))
    return profile


def test_u1_heisenberg_domain_wall_spreads_and_conserves_sz():
    psi = product_tt("uuuudddd", symmetry="u1")
    gates = bug_heisenberg_bond_gates(make_siteinds(8, symmetry="u1"), Jxy=1.0, Jz=1.0)

    centre_history = []
    maxlink_history = []

    for _ in range(5):
        info = bug_two_site(psi, gates, 0.1, order="strang", maxdim=32, augment=True, aug_krylov_depth=2)
        profile = _mz_profile(psi)

        assert vector(psi).numel() == 256
        assert abs(sum(profile)) < 1e-10
        assert max(abs(profile[j] + profile[-1 - j]) for j in range(len(profile))) < 1e-10

        centre_history.append(profile[3])
        maxlink_history.append(maxlinkdim(psi))

    assert centre_history[0] > centre_history[-1]
    assert all(centre_history[k + 1] < centre_history[k] for k in range(len(centre_history) - 1))
    assert all(maxlink_history[k + 1] >= maxlink_history[k] for k in range(len(maxlink_history) - 1))
    assert maxlink_history[-1] >= 8
    assert info.bond_dims_after[3] >= 8
