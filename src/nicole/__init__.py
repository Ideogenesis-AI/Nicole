# Copyright (C) 2025-2026 Changkai Zhang.
#
# This file is part of Nicole (TN) library.
#
# Nicole (TN) is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Nicole (TN) is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Nicole (TN). If not, see <https://www.gnu.org/licenses/>.


"""Nicole (TN) public API surface for symmetry-aware tensor utilities."""

from .contract import contract, trace
from .decomp import decomp
from .identity import identity, isometry, isometry_n
from .index import Index, Sector
from .operators import conj, permute, transpose
from .operators import oplus, diag, inv
from .operators import subsector, merge_axes
from .space import load_space
from .symmetry.abelian import U1Group, Z2Group
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
    "ProductGroup",
    "contract",
    "trace",
    "permute",
    "transpose",
    "conj",
    "diag",
    "inv",
    "oplus",
    "decomp",
    "identity",
    "isometry",
    "isometry_n",
    "subsector",
    "merge_axes",
    "load_space",
]


