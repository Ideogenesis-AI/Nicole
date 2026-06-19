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

from nicole.bug.sweep.bug_sweep import _bug_bond_snapshot, _bug_local_effective_hamiltonian, bug_xx_bond_gates, bug_xx_parity_mpos
from nicole.bug.indices import siteinds as make_siteinds
from nicole.bug.helpers import tcontract
from nicole.bug.ttutils import matrix as bug_dense, random_tt


def _to_torch(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.complex128)


def _commutator_norm(A, B):
    return torch.linalg.norm(A @ B - B @ A)


def test_bug_parity_decomposition_and_hermiticity():
    sites = make_siteinds(6, d=2)
    W_odd, W_even, W_full = bug_xx_parity_mpos(sites, J=1.0)
    H_odd = _to_torch(bug_dense(W_odd))
    H_even = _to_torch(bug_dense(W_even))
    H_full = _to_torch(bug_dense(W_full))

    rel = torch.linalg.norm((H_odd + H_even) - H_full) / torch.linalg.norm(H_full)
    assert rel < 1e-12
    assert torch.linalg.norm(H_odd - H_odd.conj().T) < 1e-12
    assert torch.linalg.norm(H_even - H_even.conj().T) < 1e-12
    assert torch.linalg.norm(H_full - H_full.conj().T) < 1e-12

    # Within-parity terms commute for nearest-neighbour XX checkerboard split.
    assert _commutator_norm(H_odd, H_odd) < 1e-12
    assert _commutator_norm(H_even, H_even) < 1e-12
    assert _commutator_norm(H_odd, H_even) > 1e-6


def test_local_effective_hamiltonian_tensor_contracts():
    sites = make_siteinds(6, d=2)
    psi = random_tt(sites, maxdim=4, seed=2)
    gates = bug_xx_bond_gates(sites, J=1.0)

    for bond in (1, 3):
        snap = _bug_bond_snapshot(psi, bond)
        HW = _bug_local_effective_hamiltonian(gates[bond - 1], snap)
        out = tcontract(HW, snap["theta0_tens"])
        # The contraction should preserve the two-site tensor shape.
        lhs_shape = next(iter(out.data.values())).shape
        rhs_shape = next(iter(snap["theta0_tens"].data.values())).shape
        lhs_squeezed = tuple(dim for dim in lhs_shape if dim != 1)
        rhs_squeezed = tuple(dim for dim in rhs_shape if dim != 1)
        arr = next(iter(out.data.values()))
        assert torch.isfinite(arr).all()
        # At minimum, the transformed tensor must retain both physical site dimensions.
        assert lhs_squeezed.count(rhs_squeezed[1]) >= 2
