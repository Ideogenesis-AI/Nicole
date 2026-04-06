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


"""Product group for multiple independent symmetries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple

from .base import SymmetryGroup, AbelianGroup, UnitaryGroup


@dataclass(frozen=True)
class ProductGroup(SymmetryGroup):
    """Product of multiple independent symmetry groups.
    
    Represents the direct product of multiple symmetry groups, enabling tensors
    with multiple independent conserved quantities. Charges are tuples where each
    component corresponds to one component group.
    
    Supports Abelian groups and at most one UnitaryGroup (non-Abelian) component.
    If a UnitaryGroup is present, it must be the last component.
    
    Attributes
    ----------
    components:
        Tuple of component SymmetryGroup instances. Can contain AbelianGroup
        instances and at most one UnitaryGroup instance (placed at the end).
    
    Examples
    --------
    >>> # U(1) particle number × U(1) spin (all Abelian)
    >>> group = ProductGroup([U1Group(), U1Group()])
    >>> group.neutral
    (0, 0)
    >>> group.fuse_unique((2, 1), (1, -1))
    (3, 0)
    
    >>> # U(1) × Z(2) (all Abelian)
    >>> group = ProductGroup([U1Group(), Z2Group()])
    >>> group.neutral
    (0, 0)
    >>> group.fuse_unique((3, 1), (-1, 0))
    (2, 1)
    
    >>> # U(1) × SU(2) (mixed: has UnitaryGroup)
    >>> group = ProductGroup([U1Group(), SU2Group()])
    >>> group.fuse_channels((1, 1), (0, 1))
    ((1, 0), (1, 2))  # Two fusion channels from SU(2) (spin-1/2 ⊗ spin-1/2)
    """
    
    components: Tuple[SymmetryGroup, ...]
    _has_unitary: bool
    
    def __init__(self, components: Sequence[SymmetryGroup]) -> None:
        """Initialize ProductGroup with component symmetry groups.
        
        Parameters
        ----------
        components:
            Sequence of SymmetryGroup instances. Must contain at least one group.
            Can include AbelianGroup instances and at most one UnitaryGroup instance.
            Nested ProductGroups are not allowed.
        
        Raises
        ------
        ValueError
            If components is empty, contains multiple UnitaryGroups, or has
            UnitaryGroup not at the last position.
        TypeError
            If components contains non-SymmetryGroup instances or nested ProductGroups.
        """
        if not components:
            raise ValueError("ProductGroup requires at least one component")
        
        # Validate all components are SymmetryGroup instances
        for i, comp in enumerate(components):
            if not isinstance(comp, SymmetryGroup):
                raise TypeError(
                    f"Component {i} is not a SymmetryGroup instance: {type(comp)}"
                )
        
        # Reject nested ProductGroups and find UnitaryGroup components
        unitary_indices = []
        for i, comp in enumerate(components):
            if isinstance(comp, ProductGroup):
                raise TypeError(
                    f"Component {i} is a ProductGroup. "
                    f"Nested ProductGroups are not allowed."
                )
            elif isinstance(comp, UnitaryGroup):
                unitary_indices.append(i)
            elif not isinstance(comp, AbelianGroup):
                raise TypeError(
                    f"Component {i} must be AbelianGroup or UnitaryGroup, "
                    f"got {type(comp).__name__}"
                )
        
        # Validate: at most one UnitaryGroup, must be at end
        if len(unitary_indices) > 1:
            raise ValueError(
                f"ProductGroup allows at most one UnitaryGroup component. "
                f"Found at indices: {unitary_indices}"
            )
        
        if unitary_indices and unitary_indices[0] != len(components) - 1:
            raise ValueError(
                f"UnitaryGroup component must be the last component. "
                f"Found at index {unitary_indices[0]}, expected index {len(components) - 1}"
            )
        
        # Use object.__setattr__ since dataclass is frozen
        object.__setattr__(self, 'components', tuple(components))
        object.__setattr__(self, '_has_unitary', bool(unitary_indices))
    
    @property
    def name(self) -> str:
        """Return the product group name, e.g., 'U1×Z2'."""
        return "×".join(comp.name for comp in self.components)
    
    @property
    def neutral(self) -> Tuple[Any, ...]:
        """Return the neutral element as a tuple of component neutrals."""
        return tuple(comp.neutral for comp in self.components)
    
    @property
    def is_abelian(self) -> bool:
        """Return True if all components are Abelian, False if any is non-Abelian.
        
        A ProductGroup is Abelian only when all its components are Abelian groups.
        If it contains a UnitaryGroup component, it is non-Abelian.
        """
        return not self._has_unitary
    
    def fuse_unique(self, *qs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Fuse multiple charge tuples component-wise (Abelian-only).
        
        This method is only available when all components are Abelian groups.
        For ProductGroups containing a UnitaryGroup, use `fuse_channels` instead.
        
        Parameters
        ----------
        *qs:
            Variable number of charge tuples to fuse.
        
        Returns
        -------
        Tuple
            The unique fused charge tuple.
        
        Raises
        ------
        TypeError
            If this ProductGroup contains a UnitaryGroup component.
        
        Examples
        --------
        >>> group = ProductGroup([U1Group(), Z2Group()])
        >>> group.fuse_unique((2, 1), (1, 0), (-1, 1))
        (2, 0)
        """
        if self._has_unitary:
            raise TypeError(
                f"ProductGroup {self.name} contains a non-Abelian component. "
                f"Use fuse_channels() for multi-channel fusion instead."
            )
        
        if not qs:
            return self.neutral
        
        # Validate all charges
        for q in qs:
            self.validate_charge(q)
        
        # Fuse each component independently (all are Abelian)
        fused_components = []
        for i, comp in enumerate(self.components):
            component_charges = [q[i] for q in qs]
            fused_components.append(comp.fuse_unique(*component_charges))
        
        return tuple(fused_components)
    
    def fuse_channels(self, *qs: Tuple[Any, ...]) -> Tuple[Tuple[Any, ...], ...]:
        """Fuse multiple charge tuples, returning all achievable fusion channels.
        
        When all components are Abelian, returns a single-element tuple.
        When a UnitaryGroup component is present (must be at the end), returns
        multiple channels corresponding to the UnitaryGroup's fusion outcomes.
        
        Parameters
        ----------
        *qs:
            One or more charge tuples to fuse.
        
        Returns
        -------
        Tuple[Tuple[Any, ...], ...]
            Tuple of all achievable fusion channel tuples, independent of
            the fusion tree structure.
        
        Examples
        --------
        >>> # All Abelian: single channel
        >>> group = ProductGroup([U1Group(), Z2Group()])
        >>> group.fuse_channels((2, 1), (1, 0))
        ((3, 1),)
        
        >>> # With UnitaryGroup: multiple channels (pairwise)
        >>> group = ProductGroup([U1Group(), SU2Group()])
        >>> group.fuse_channels((1, 1), (0, 1))
        ((1, 0), (1, 2))
        
        >>> # With UnitaryGroup: multiple charges
        >>> group.fuse_channels((1, 1), (0, 1), (2, 1))
        ((3, 1), (3, 3))
        """
        if not qs:
            return (self.neutral,)
        
        # Validate all charges
        for q in qs:
            self.validate_charge(q)
        
        # Single charge: return itself
        if len(qs) == 1:
            return (qs[0],)
        
        if not self._has_unitary:
            # All Abelian: single fusion outcome
            return (self.fuse_unique(*qs),)
        
        # Mixed case: Abelian components fuse uniquely, Unitary has channels
        # Fuse all Abelian components (all except last)
        abelian_results = []
        for i in range(len(self.components) - 1):
            comp = self.components[i]
            # Extract i-th component from each charge tuple and fuse
            comp_charges = tuple(q[i] for q in qs)
            abelian_results.append(comp.fuse_unique(*comp_charges))
        
        # Last component is UnitaryGroup: get all fusion channels
        unitary_comp = self.components[-1]
        if not isinstance(unitary_comp, UnitaryGroup):
            raise RuntimeError("Internal error: expected last component to be UnitaryGroup")
        
        # Extract last component from each charge tuple
        unitary_charges = tuple(q[-1] for q in qs)
        unitary_channels = unitary_comp.fuse_channels(*unitary_charges)
        
        # Combine: each unitary channel creates one ProductGroup charge
        results = []
        for uch in unitary_channels:
            results.append(tuple(abelian_results + [uch]))
        
        return tuple(results)
    
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

    def irrep_dim(self, q: Tuple[Any, ...]) -> int:
        """Return dimension of the irreducible representation.
        
        For ProductGroup, the irrep dimension is the product of constituent
        irrep dimensions.
        
        Parameters
        ----------
        q:
            Charge tuple.
        
        Returns
        -------
        int
            Product of component irrep dimensions.
        """
        self.validate_charge(q)
        dim = 1
        for comp, qi in zip(self.components, q):
            dim *= comp.irrep_dim(qi)
        return dim
    
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
