# Copyright (C) 2026 Changkai Zhang.
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

"""Product group for multiple independent symmetries."""

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

from .base import AbelianGroup, SymmetryGroup


@dataclass(frozen=True)
class ProductGroup(SymmetryGroup):
    """Product of multiple independent symmetry groups.
    
    Represents the direct product of multiple (currently Abelian) symmetry groups,
    enabling tensors with multiple independent conserved quantities. Charges are
    tuples where each component corresponds to one component group.
    
    Attributes
    ----------
    components:
        Tuple of component SymmetryGroup instances. Currently restricted to
        AbelianGroup instances, but designed for future extension to non-Abelian.
    
    Examples
    --------
    >>> # U(1) particle number × U(1) spin
    >>> group = ProductGroup([U1Group(), U1Group()])
    >>> group.neutral
    (0, 0)
    >>> group.fuse((2, 1), (1, -1))
    (3, 0)
    
    >>> # U(1) × Z(2)
    >>> group = ProductGroup([U1Group(), Z2Group()])
    >>> group.neutral
    (0, 0)
    >>> group.fuse((3, 1), (-1, 0))
    (2, 1)
    """
    
    components: Tuple[SymmetryGroup, ...]
    
    def __init__(self, components: Sequence[SymmetryGroup]) -> None:
        """Initialize ProductGroup with component symmetry groups.
        
        Parameters
        ----------
        components:
            Sequence of SymmetryGroup instances. Must contain at least one group.
            Currently restricted to AbelianGroup instances.
        
        Raises
        ------
        ValueError
            If components is empty or contains non-Abelian groups.
        TypeError
            If components contains non-SymmetryGroup instances.
        """
        if not components:
            raise ValueError("ProductGroup requires at least one component")
        
        # Validate all components are SymmetryGroup instances
        for i, comp in enumerate(components):
            if not isinstance(comp, SymmetryGroup):
                raise TypeError(
                    f"Component {i} is not a SymmetryGroup instance: {type(comp)}"
                )
        
        # For now, restrict to Abelian groups only
        for i, comp in enumerate(components):
            if not isinstance(comp, AbelianGroup):
                raise ValueError(
                    f"Component {i} is not an AbelianGroup. "
                    f"Non-Abelian groups are not yet supported in ProductGroup."
                )
        
        # Use object.__setattr__ since dataclass is frozen
        object.__setattr__(self, 'components', tuple(components))
    
    @property
    def name(self) -> str:
        """Return the product group name, e.g., 'U1×Z2'."""
        return "×".join(comp.name for comp in self.components)
    
    @property
    def neutral(self) -> Tuple[Any, ...]:
        """Return the neutral element as a tuple of component neutrals."""
        return tuple(comp.neutral for comp in self.components)
    
    def inverse(self, q: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Return the inverse of a charge tuple.
        
        Parameters
        ----------
        q:
            Charge tuple with one component per group.
        
        Returns
        -------
        Tuple
            Tuple of inverted charges.
        """
        self.validate_charge(q)
        return tuple(comp.inverse(qi) for comp, qi in zip(self.components, q))
    
    def fuse(self, *qs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Fuse multiple charge tuples component-wise.
        
        Parameters
        ----------
        *qs:
            Variable number of charge tuples to fuse.
        
        Returns
        -------
        Tuple
            Fused charge tuple.
        
        Examples
        --------
        >>> group = ProductGroup([U1Group(), Z2Group()])
        >>> group.fuse((2, 1), (1, 0), (-1, 1))
        (2, 0)
        """
        if not qs:
            return self.neutral
        
        # Validate all charges
        for q in qs:
            self.validate_charge(q)
        
        # Fuse each component independently
        result = []
        for i, comp in enumerate(self.components):
            component_charges = [q[i] for q in qs]
            result.append(comp.fuse(*component_charges))
        
        return tuple(result)
    
    def equal(self, a: Tuple[Any, ...], b: Tuple[Any, ...]) -> bool:
        """Check if two charge tuples are equal component-wise.
        
        Parameters
        ----------
        a, b:
            Charge tuples to compare.
        
        Returns
        -------
        bool
            True if all components are equal.
        """
        self.validate_charge(a)
        self.validate_charge(b)
        return all(comp.equal(ai, bi) for comp, ai, bi in zip(self.components, a, b))
    
    def validate_charge(self, q: Any) -> None:
        """Validate that a charge is a tuple of correct length with valid components.
        
        Parameters
        ----------
        q:
            Charge to validate.
        
        Raises
        ------
        TypeError
            If charge is not a tuple.
        ValueError
            If charge has incorrect length or invalid component charges.
        """
        if not isinstance(q, tuple):
            raise TypeError(
                f"ProductGroup charge must be a tuple, got {type(q).__name__}"
            )
        
        if len(q) != len(self.components):
            raise ValueError(
                f"Charge tuple length {len(q)} does not match "
                f"number of components {len(self.components)}"
            )
        
        # Validate each component charge
        for i, (comp, qi) in enumerate(zip(self.components, q)):
            try:
                comp.validate_charge(qi)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid charge for component {i} ({comp.name}): {e}"
                ) from e
    
    def dual(self, q: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Return the dual (contragredient) of a charge tuple.
        
        Parameters
        ----------
        q:
            Charge tuple.
        
        Returns
        -------
        Tuple
            Tuple of dual charges.
        """
        self.validate_charge(q)
        return tuple(comp.dual(qi) for comp, qi in zip(self.components, q))
    
    @property
    def num_components(self) -> int:
        """Return the number of component groups."""
        return len(self.components)
    
    def get_component(self, i: int) -> SymmetryGroup:
        """Access a specific component group by index.
        
        Parameters
        ----------
        i:
            Component index (0-based).
        
        Returns
        -------
        SymmetryGroup
            The i-th component group.
        
        Raises
        ------
        IndexError
            If index is out of range.
        """
        if i < 0 or i >= len(self.components):
            raise IndexError(
                f"Component index {i} out of range [0, {len(self.components)})"
            )
        return self.components[i]
