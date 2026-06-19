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

from nicole.bug.indices import siteinds as make_siteinds
from nicole.bug.ttutils import OpSum, matrix, mpo_from_opsum


def _to_torch(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.complex128)


def _dense_lrxx_exact(n: int, alpha: float) -> torch.Tensor:
    sp = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)
    sm = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.complex128)
    eye = torch.eye(2, dtype=torch.complex128)

    def kron_many(mats):
        out = mats[0]
        for m in mats[1:]:
            out = torch.kron(out, m)
        return out

    d = 2**n
    h = torch.zeros((d, d), dtype=torch.complex128)
    for i in range(n):
        for j in range(i + 1, n):
            c = 0.5 / abs((i + 1) - (j + 1)) ** alpha
            ops_pm = [eye] * n
            ops_mp = [eye] * n
            ops_pm[i], ops_pm[j] = sp, sm
            ops_mp[i], ops_mp[j] = sm, sp
            h += c * kron_many(ops_pm) + c * kron_many(ops_mp)
    return h


def test_long_range_xx_mpo_matches_dense_reference():
    n = 6
    alpha = 1.5
    sites = make_siteinds(n, d=2)
    osum = OpSum()
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            c = 0.5 / abs(i - j) ** alpha
            osum += (c, [("S+", i), ("S-", j)])
            osum += (c, [("S-", i), ("S+", j)])

    W = mpo_from_opsum(osum, sites)
    dense_mpo = _to_torch(matrix(W))
    dense_ref = _dense_lrxx_exact(n, alpha)

    rel_frob = torch.linalg.norm(dense_mpo - dense_ref) / torch.linalg.norm(dense_ref)
    rel_op = torch.linalg.norm(dense_mpo - dense_ref, ord=2) / torch.linalg.norm(dense_ref, ord=2)
    max_abs = torch.max(torch.abs(dense_mpo - dense_ref))

    assert rel_frob < 1e-12
    assert rel_op < 1e-12
    assert max_abs < 1e-12
