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


"""BUG/KLS sweep helpers and canonical-form utilities."""

from .init import (
    BUGInfo,
    record_s_step_rank,
)
from .kls import (
    _augmented_left_isometry_from_k,
    _augmented_right_isometry_from_l,
    _faithful_kls_local_bond_candidate,
    _faithful_reverse_kls_local_bond_candidate,
    _pick_left_update,
    _pick_right_update,
    _truncate_quantum_s_step,
    _truncate_quantum_s_step_reverse,
)
from .bug_sweep import (
    _bug_bond_snapshot,
    _bug_local_effective_hamiltonian,
    _bug_parity_sweep,
    bug_two_site,
    bug_heisenberg_bond_gates,
    bug_xx_bond_gates,
    bug_xx_parity_mpos,
)

__all__ = [
    "BUGInfo",
    "_augmented_left_isometry_from_k",
    "_augmented_right_isometry_from_l",
    "_bug_bond_snapshot",
    "_bug_local_effective_hamiltonian",
    "_bug_parity_sweep",
    "_faithful_kls_local_bond_candidate",
    "_faithful_reverse_kls_local_bond_candidate",
    "_pick_left_update",
    "_pick_right_update",
    "_truncate_quantum_s_step",
    "_truncate_quantum_s_step_reverse",
    "bug_heisenberg_bond_gates",
    "bug_two_site",
    "bug_xx_bond_gates",
    "bug_xx_parity_mpos",
    "record_s_step_rank",
]
