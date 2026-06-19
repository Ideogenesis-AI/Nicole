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


from nicole.bug.sweep.init import BUGInfo, record_s_step_rank


def test_buginfo_defaults():
    info = BUGInfo()
    assert info.bond_dims_before == []
    assert info.bond_dims_after == []
    assert info.elapsed == 0.0
    assert info.aug_sizes_k == []
    assert info.aug_sizes_l == []
    assert info.lanczos_numops == []


def test_buginfo_record_helpers():
    info = BUGInfo()
    record_s_step_rank(info, "fwd", 2, 3)
    assert info.s_step_kept_ranks == [3]
    assert info.s_step_bonds == [2]

