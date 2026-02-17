# Copyright (C) 2025 Changkai Zhang.
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

"""Abstract base classes for symmetry groups used by Nicole library."""

from abc import ABC, abstractmethod
from typing import Tuple

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
    def dual(self, q: Charge) -> Charge:
        """Return the dual (contragredient) representation of a charge.
        
        The dual representation is the representation that makes contraction
        well-defined in the presence of symmetries. For Abelian groups, this
        coincides with the group inverse, but conceptually they are different:
        charges label representations, not group elements.
        """
        ...

    @abstractmethod
    def equal(self, a: Charge, b: Charge) -> bool:
        ...

    @abstractmethod
    def validate_charge(self, q: Charge) -> None:
        ...


class AbelianGroup(SymmetryGroup, ABC):
    """Base class for Abelian symmetry groups with deterministic fusion."""

    @abstractmethod
    def fuse_unique(self, *qs: Charge) -> Charge:
        """Fuse charges deterministically (single unique result).
        
        Parameters
        ----------
        *qs:
            Variable number of charges to fuse sequentially.
        
        Returns
        -------
        Charge
            The unique fusion result.
        """
        ...


class UnitaryGroup(SymmetryGroup, ABC):
    """Base class for non-Abelian unitary groups with multi-channel fusion."""

    @abstractmethod
    def fuse_channels(self, q1: Charge, q2: Charge) -> Tuple[Charge, ...]:
        """Fuse two charges pairwise, returning all allowed channels.
        
        Parameters
        ----------
        q1, q2:
            Two charges to fuse pairwise.
        
        Returns
        -------
        Tuple[Charge, ...]
            Tuple of all allowed fusion channels satisfying the group's
            fusion rules (e.g., triangular inequality for SU(2)).
        
        Notes
        -----
        For non-Abelian groups, fusion is not associative in general,
        so only pairwise fusion is supported. Multi-index fusion requires
        an explicit fusion tree structure (handled via Yuzuha protocol).
        """
        ...


