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


"""Readable Python BUG helpers backed by Nicole tensors.

The top-level package re-exports the most commonly used state constructors,
index helpers, and two-site evolution entry points.
"""

from .sweep import (
    BUGInfo,
    bug_heisenberg_bond_gates,
    bug_two_site,
    bug_xx_bond_gates,
    bug_xx_parity_mpos,
)
from .indices import Ix, fresh_itag, idx, siteinds
from .ttutils import TensorTrain, TensorTrainOperator, product_tt

__all__ = [
    "BUGInfo",
    "Ix",
    "TensorTrain",
    "TensorTrainOperator",
    "bug_heisenberg_bond_gates",
    "bug_two_site",
    "bug_xx_bond_gates",
    "bug_xx_parity_mpos",
    "fresh_itag",
    "idx",
    "product_tt",
    "siteinds",
]
