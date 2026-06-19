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


from nicole.bug import bug_heisenberg_bond_gates, bug_two_site, product_tt, siteinds
from nicole.bug.helpers import reshape_fortran
from nicole.bug.ttutils.tensortrain import maxlinkdim, vector


def _total_sz(psi) -> float:
    dense = reshape_fortran(vector(psi), [2] * len(psi))
    prob = dense.abs() ** 2
    total = 0.0
    for axis in range(len(psi)):
        up = prob.select(axis, 0).sum().item()
        down = prob.select(axis, 1).sum().item()
        total += 0.5 * (up - down)
    return total


def test_symmetric_reference_strang_step_preserves_norm_and_total_sz():
    psi = product_tt("uudd", symmetry="u1")
    sites = siteinds(4, symmetry="u1")
    gates = bug_heisenberg_bond_gates(sites, Jxy=1.0, Jz=1.0)

    before = vector(psi).norm().item()
    bug_two_site(psi, gates, 0.05, order="strang", maxdim=8, augment=False)

    assert vector(psi).numel() == 16
    assert abs(vector(psi).norm().item() - before) < 1e-12
    assert abs(_total_sz(psi)) < 1e-12
    assert maxlinkdim(psi) <= 8
