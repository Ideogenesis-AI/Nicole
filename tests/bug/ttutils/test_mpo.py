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

from nicole.bug.ttutils import matrix, normalize_, random_tt, vector


def bug_exact(v: torch.Tensor, H: torch.Tensor, t: complex) -> torch.Tensor:
    return torch.matrix_exp(t * H) @ v


def bug_infidelity(v: torch.Tensor, ref: torch.Tensor) -> float:
    ov = torch.vdot(ref, v)
    denom = torch.linalg.norm(v) * torch.linalg.norm(ref)
    if denom == 0:
        return 0.0
    fid = abs(ov / denom) ** 2
    return float(max(0.0, 1.0 - fid))


def bug_norm_error(v: torch.Tensor, ref: torch.Tensor) -> float:
    return float(abs(torch.linalg.norm(v) - torch.linalg.norm(ref)))


def bug_rank3_state(sites, seed: int):
    psi = random_tt(sites, maxdim=3, seed=seed)
    normalize_(psi)
    return psi


def bug_dense(W):
    return matrix(W)


def bug_vec(psi):
    return vector(psi)

