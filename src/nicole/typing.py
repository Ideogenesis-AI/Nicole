# Copyright (C) 2025 Changkai Zhang.
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


from __future__ import annotations

"""Shared typing utilities for Nicole (TN) symmetry-aware tensor machinery."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Hashable

Charge = Hashable


class Direction(IntEnum):
    """Orientation flag used by tensor indices."""

    IN = -1
    OUT = 1

    def reverse(self) -> Direction:
        """Return the opposite orientation (IN ↔ OUT)."""
        return Direction(-int(self))


@dataclass(frozen=True)
class Sector:
    """Charge sector descriptor pairing a conserved charge with its dimension.

    Attributes
    ----------
    charge:
        The conserved quantity carried by the sector. May be any hashable value.
    dim:
        Positive integer indicating how many states/multiplets the sector spans.
    """

    charge: Charge
    dim: int

    def __post_init__(self) -> None:
        """Validate that the sector has a strictly positive dimension."""
        if self.dim <= 0:
            raise ValueError("Sector dimension must be positive")
