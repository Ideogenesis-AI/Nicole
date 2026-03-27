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


"""Nicole public API surface for symmetry-aware tensor utilities."""

from .contract import contract, trace
from .decomp import decomp
from .identity import identity, isometry, isometry_n
from .index import Index, Sector
from .maneuver import conj, permute, transpose
from .maneuver import oplus, diag, inv, capcup
from .maneuver import filter_blocks, merge_axes
from .space import load_space
from .symmetry.abelian import U1Group, Z2Group
from .symmetry.unitary import SU2Group
from .symmetry.product import ProductGroup
from .tensor import Tensor
from .typing import Charge, Direction

__all__ = [
    "Charge",
    "Direction",
    "Sector",
    "Index",
    "Tensor",
    "U1Group",
    "Z2Group",
    "SU2Group",
    "ProductGroup",
    "contract",
    "trace",
    "permute",
    "transpose",
    "conj",
    "diag",
    "inv",
    "oplus",
    "capcup",
    "decomp",
    "identity",
    "isometry",
    "isometry_n",
    "filter_blocks",
    "merge_axes",
    "load_space",
]


