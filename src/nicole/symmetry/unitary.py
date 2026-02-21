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

"""Special unitary symmetry groups (N>1) with multi-channel fusion."""

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

    def irrep_dim(self, two_j: int) -> int:
        """Return dimension of spin-j irreducible representation.
        
        For SU(2), the dimension of a spin-j representation is 2j+1.
        
        Parameters
        ----------
        two_j:
            Quantum number (2j, where j is the physical spin).
        
        Returns
        -------
        int
            Dimension of the representation: 2j + 1.
        """
        self.validate_charge(two_j)
        return two_j + 1

    def fuse_channels(self, *two_js: int) -> Tuple[int, ...]:
        """Fuse multiple spins, returning all possible total spin channels.
        
        Returns all total spin values achievable by fusing the given spins,
        independent of the fusion tree structure. For SU(2), any value from
        the minimum to maximum in steps of 2 can be achieved by some choice
        of fusion tree.
        
        In the 2j convention:
        - For two spins: 2j1 ⊗ 2j2 → |2j1 - 2j2|, ..., 2j1 + 2j2 (step: 2)
        - For n spins: 2j_min, 2j_min + 2, ..., 2j_max where:
          - 2j_max = 2j1 + 2j2 + ... + 2jn (all aligned)
          - 2j_min: largest value ≤ |2j_largest - sum_of_others| with matching integrality
        
        Integrality constraint: All achievable total spins must be either all
        integer or all half-integer. In 2j notation, this means all returned
        values have the same parity (even for integer, odd for half-integer).
        
        Parameters
        ----------
        *two_js:
            One or more quantum numbers (2j values) to fuse.
        
        Returns
        -------
        Tuple[int, ...]
            Tuple of all achievable total spin channels (2j values), sorted
            from smallest to largest. The actual fusion tree structure needed
            to achieve each value is determined by external packages.
        
        Examples
        --------
        >>> group = SU2Group()
        >>> # Spin-1/2 ⊗ spin-1/2 (1 ⊗ 1)
        >>> group.fuse_channels(1, 1)
        (0, 2)
        
        >>> # Spin-1 ⊗ spin-1 (2 ⊗ 2)
        >>> group.fuse_channels(2, 2)
        (0, 2, 4)
        
        >>> # Three spin-1/2 (1 ⊗ 1 ⊗ 1)
        >>> group.fuse_channels(1, 1, 1)
        (1, 3)
        
        >>> # Spin-1 ⊗ spin-1 ⊗ spin-1/2 (2 ⊗ 2 ⊗ 1)
        >>> group.fuse_channels(2, 2, 1)
        (1, 3, 5)
        
        >>> # Four spin-1/2 (1 ⊗ 1 ⊗ 1 ⊗ 1)
        >>> group.fuse_channels(1, 1, 1, 1)
        (0, 2, 4)
        """
        if not two_js:
            return (0,)
        
        # Validate all charges
        for two_j in two_js:
            self.validate_charge(two_j)
        
        # Single spin: return itself
        if len(two_js) == 1:
            return (two_js[0],)
        
        # Maximum: all spins aligned (parallel coupling)
        two_j_max = sum(two_js)
        
        # Minimum: maximal cancellation occurs when largest spin opposes all others
        largest = max(two_js)
        sum_others = sum(two_js) - largest
        two_j_min_unconstrained = max(0, largest - sum_others)
        
        # Count down from max by 2s until below min_unconstrained
        # This automatically maintains correct integrality
        num_channels = (two_j_max - two_j_min_unconstrained) // 2 + 1
        return tuple(two_j_max - 2 * i for i in range(num_channels - 1, -1, -1))

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
