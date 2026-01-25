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

"""Abstract base classes for symmetry groups used by Nicole (TN)."""

from abc import ABC, abstractmethod
from typing import Iterable

from ..typing import Charge


class SymmetryGroup(ABC):
    """Abstract protocol that symmetry group implementations must satisfy."""

    @property
    @abstractmethod
    def name(self) -> str:  # pragma: no cover - trivial
        ...

    @property
    @abstractmethod
    def neutral(self) -> Charge:
        ...

    @abstractmethod
    def inverse(self, q: Charge) -> Charge:
        ...

    @abstractmethod
    def fuse(self, *qs: Charge) -> Charge:
        ...

    @abstractmethod
    def equal(self, a: Charge, b: Charge) -> bool:
        ...

    @abstractmethod
    def validate_charge(self, q: Charge) -> None:
        ...

    def fuse_many(self, qs: Iterable[Charge]) -> Charge:
        """Fold a sequence of charges using repeated `fuse` applications."""
        acc = self.neutral
        for q in qs:
            acc = self.fuse(acc, q)
        return acc

    def dual(self, q: Charge) -> Charge:
        """Return the dual (contragredient) of a charge.

        By default, the dual coincides with the inverse element; individual
        symmetry groups can override this if their contragredient differs from
        the group inverse.
        """

        return self.inverse(q)


class AbelianGroup(SymmetryGroup, ABC):
    """Marker base for Abelian groups."""


