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

    @property
    @abstractmethod
    def is_abelian(self) -> bool:
        """Return True if the group is Abelian, False otherwise.
        
        Abelian groups have unique fusion outcomes, while non-Abelian groups
        (UnitaryGroup and ProductGroup with UnitaryGroup components) can have
        multiple fusion channels.
        """
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

    @abstractmethod
    def irrep_dim(self, q: Charge) -> int:
        """Return the dimension of the irreducible representation labeled by charge q.
        
        For Abelian groups, this is always 1. For non-Abelian groups like SU(2),
        this is the dimension of the spin-j representation (2j+1).
        """
        ...


class AbelianGroup(SymmetryGroup, ABC):
    """Base class for Abelian symmetry groups with deterministic fusion."""

    @property
    def is_abelian(self) -> bool:
        """Return True for all Abelian groups."""
        return True

    def irrep_dim(self, q: Charge) -> int:
        """Return dimension of irreducible representation (always 1 for Abelian groups)."""
        return 1

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

    @property
    def is_abelian(self) -> bool:
        """Return False for all non-Abelian unitary groups."""
        return False

    @abstractmethod
    def irrep_dim(self, q: Charge) -> int:
        """Return dimension of the irreducible representation labeled by charge q.
        
        For non-Abelian groups, this is the dimension of the representation space.
        For example, in SU(2), charge q = 2j labels a spin-j representation with
        dimension 2j+1.
        """
        ...

    @abstractmethod
    def fuse_channels(self, *qs: Charge) -> Tuple[Charge, ...]:
        """Fuse multiple charges, returning all achievable total charge channels.
        
        Returns all total charge values achievable by fusing the given charges,
        independent of the fusion tree structure. The specific fusion tree needed
        to achieve each value is determined via Yuzuha protocol.
        
        Parameters
        ----------
        *qs:
            One or more charges to fuse.
        
        Returns
        -------
        Tuple[Charge, ...]
            Tuple of all achievable total charge channels satisfying the group's
            fusion rules. For example, in SU(2) with the triangular inequality,
            all values from the minimum to maximum in steps of 2.
        
        Notes
        -----
        The returned channels represent all possible outcomes regardless of
        the fusion tree structure. The actual intermediate states and coupling
        coefficients depend on the chosen tree (handled via Yuzuha protocol).
        """
        ...


