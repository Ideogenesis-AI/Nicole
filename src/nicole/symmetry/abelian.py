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

"""Concrete Abelian symmetry groups shipped with Nicole (TN)."""

from dataclasses import dataclass
from typing import Any

from .base import AbelianGroup


@dataclass(frozen=True)
class U1Group(AbelianGroup):
    """U(1) with integer-valued charges (e.g., particle number)."""

    @property
    def name(self) -> str:
        return "U1"

    @property
    def neutral(self) -> int:
        return 0

    def inverse(self, q: int) -> int:
        self.validate_charge(q)
        return -q

    def fuse(self, *qs: int) -> int:
        s = 0
        for q in qs:
            self.validate_charge(q)
            s += int(q)
        return s

    def equal(self, a: int, b: int) -> bool:
        return int(a) == int(b)

    def validate_charge(self, q: Any) -> None:
        """Ensure a charge lies in the integer lattice."""
        if not isinstance(q, int):
            raise TypeError("U1 charge must be an int")


@dataclass(frozen=True)
class Z2Group(AbelianGroup):
    """Z2 with charges 0/1 (parity)."""

    @property
    def name(self) -> str:
        return "Z2"

    @property
    def neutral(self) -> int:
        return 0

    def inverse(self, q: int) -> int:
        self.validate_charge(q)
        return q & 1

    def fuse(self, *qs: int) -> int:
        acc = 0
        for q in qs:
            self.validate_charge(q)
            acc ^= (q & 1)
        return acc

    def equal(self, a: int, b: int) -> bool:
        return (a & 1) == (b & 1)

    def validate_charge(self, q: Any) -> None:
        """Ensure a charge is a valid Z2 parity (0 or 1)."""
        if not isinstance(q, int):
            raise TypeError("Z2 charge must be an int (0 or 1)")
        if q not in (0, 1):
            raise ValueError("Z2 charge must be 0 or 1")
