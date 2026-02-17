# Copyright (C) 2026 Changkai Zhang.
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

"""Non-Abelian unitary symmetry groups with multi-channel fusion."""

from dataclasses import dataclass
from typing import Any, Tuple

from .base import UnitaryGroup


@dataclass(frozen=True)
class SU2Group(UnitaryGroup):
    """SU(2) symmetry group with integer-valued quantum numbers.
    
    The SU(2) group represents rotational symmetry in quantum mechanics.
    Quantum numbers are non-negative integers representing **twice the physical
    spin** (2j convention): 0, 1, 2, 3, 4, ... corresponding to physical spins
    0, 1/2, 1, 3/2, 2, ...
    
    This convention avoids fractions while preserving all half-integer spins.
    
    Fusion of two spins follows the triangular inequality:
    2j1 ⊗ 2j2 → |2j1 - 2j2|, |2j1 - 2j2| + 2, ..., 2j1 + 2j2 (step size: 2)
    
    Examples
    --------
    >>> group = SU2Group()
    >>> group.neutral
    0
    
    >>> # Spin-1/2 ⊗ spin-1/2 (1 ⊗ 1 in 2j notation) → 0, 2 (spin-0, spin-1)
    >>> group.fuse_channels(1, 1)
    (0, 2)
    
    >>> # Spin-1 ⊗ spin-1 (2 ⊗ 2) → 0, 2, 4 (spin-0, spin-1, spin-2)
    >>> group.fuse_channels(2, 2)
    (0, 2, 4)
    
    >>> # SU(2) representations are self-dual
    >>> group.dual(3)  # spin-3/2
    3
    """

    @property
    def name(self) -> str:
        """Return the group name."""
        return "SU2"

    @property
    def neutral(self) -> int:
        """Return the neutral element (spin-0, represented as 2j=0)."""
        return 0

    def dual(self, two_j: int) -> int:
        """Return the dual representation.
        
        For SU(2), all representations are self-dual (real), meaning the
        dual of a spin-j representation is spin-j itself.
        
        Parameters
        ----------
        two_j:
            Quantum number (2j, where j is the physical spin).
        
        Returns
        -------
        int
            The same quantum number (2j).
        """
        self.validate_charge(two_j)
        return two_j

    def fuse_channels(self, two_j1: int, two_j2: int) -> Tuple[int, ...]:
        """Fuse two spins pairwise, returning all allowed fusion channels.
        
        The fusion of two angular momenta follows the Clebsch-Gordan series
        (triangular inequality). In the 2j convention:
        
        2j1 ⊗ 2j2 → |2j1 - 2j2|, |2j1 - 2j2| + 2, ..., 2j1 + 2j2
        
        All values from |2j1 - 2j2| to 2j1 + 2j2 in steps of 2 are allowed.
        
        Parameters
        ----------
        two_j1, two_j2:
            Quantum numbers (2j1 and 2j2, where j1 and j2 are physical spins).
        
        Returns
        -------
        Tuple[int, ...]
            Tuple of all allowed fusion channels (2j values), sorted from
            smallest to largest.
        
        Examples
        --------
        >>> group = SU2Group()
        >>> # Spin-1/2 ⊗ spin-1/2 (represented as 1 ⊗ 1)
        >>> group.fuse_channels(1, 1)
        (0, 2)
        
        >>> # Spin-1 ⊗ spin-1 (represented as 2 ⊗ 2)
        >>> group.fuse_channels(2, 2)
        (0, 2, 4)
        
        >>> # Spin-2 ⊗ spin-1 (represented as 4 ⊗ 2)
        >>> group.fuse_channels(4, 2)
        (2, 4, 6)
        
        >>> # Spin-3/2 ⊗ spin-1/2 (represented as 3 ⊗ 1)
        >>> group.fuse_channels(3, 1)
        (2, 4)
        """
        self.validate_charge(two_j1)
        self.validate_charge(two_j2)
        
        # Compute min and max allowed 2j values
        two_j_min = abs(two_j1 - two_j2)
        two_j_max = two_j1 + two_j2
        
        # Generate all 2j values from min to max in steps of 2
        # Number of channels is (two_j_max - two_j_min) / 2 + 1
        num_channels = (two_j_max - two_j_min) // 2 + 1
        return tuple(two_j_min + 2 * i for i in range(num_channels))

    def equal(self, a: int, b: int) -> bool:
        """Check if two quantum numbers are equal.
        
        Parameters
        ----------
        a, b:
            Quantum numbers (2j values) to compare.
        
        Returns
        -------
        bool
            True if the quantum numbers are equal.
        """
        return a == b

    def validate_charge(self, two_j: Any) -> None:
        """Validate that a charge is a valid SU(2) quantum number.
        
        Valid SU(2) charges are non-negative integers representing 2j,
        where j is the physical spin: 0, 1, 2, 3, 4, ... corresponding to
        physical spins 0, 1/2, 1, 3/2, 2, ...
        
        Parameters
        ----------
        two_j:
            Charge to validate (should be 2j as an integer).
        
        Raises
        ------
        TypeError
            If charge is not an integer.
        ValueError
            If charge is negative.
        
        Examples
        --------
        >>> group = SU2Group()
        >>> group.validate_charge(0)   # Valid: spin-0 (2j=0)
        >>> group.validate_charge(1)   # Valid: spin-1/2 (2j=1)
        >>> group.validate_charge(2)   # Valid: spin-1 (2j=2)
        >>> group.validate_charge(3)   # Valid: spin-3/2 (2j=3)
        >>> group.validate_charge(-1)  # Raises ValueError
        >>> group.validate_charge(1.5) # Raises TypeError
        """
        if not isinstance(two_j, int):
            raise TypeError(
                f"SU2 quantum number (2j) must be int, got {type(two_j).__name__}"
            )
        
        # Check non-negative
        if two_j < 0:
            raise ValueError(
                f"SU2 quantum number (2j) must be non-negative, got {two_j}"
            )
