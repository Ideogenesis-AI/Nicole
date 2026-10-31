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
from .abelian import U1Group, Z2Group
from .unitary import SU2Group


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
    _neutral: Tuple[Any, ...]
    
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
        object.__setattr__(self, '_neutral', tuple(comp.neutral for comp in components))

        self._install_fast_paths()

    def _install_fast_paths(self) -> None:
        """Resolve and install the specialized fusion and dual implementations.

        Selection happens once, here, so that call sites pay no dispatch cost:
        the chosen plain functions go into the instance dict, where they shadow
        the generic class methods. Products of three or more components keep the
        generic methods. See the Fast Paths section at the end of this module.

        The installed names are not dataclass fields, so the generated `__eq__`,
        `__hash__`, and `__repr__` are unaffected.
        """
        components = self.components
        # Only pairs are specialized. Three or more components keep the generic
        # methods and pay their per-component loop in full: unrolling it would
        # take a separate implementation per arity, so no fast path is offered.
        if len(components) != 2:
            return

        # Hand-written arithmetic for the group pairs that occur in practice.
        # Keying on type objects, not names, keeps the lookup an identity hash.
        spec = _REGISTRY.get((type(components[0]), type(components[1])))
        if spec is None:
            # Any other pair, user-defined groups included, is specialized
            # generically from whatever the two components themselves provide.
            spec = _specialize_pair(components[0], components[1], self._has_unitary)

        fuse_unique, fuse_channels, dual = spec
        # A None fuse_unique leaves the generic method in place, which is what
        # products containing a UnitaryGroup need so that it still raises.
        if fuse_unique is not None:
            object.__setattr__(self, 'fuse_unique', fuse_unique)
        object.__setattr__(self, 'fuse_channels', fuse_channels)
        object.__setattr__(self, 'dual', dual)

    
    @property
    def name(self) -> str:
        """Return the product group name, e.g., 'U1×Z2'."""
        return "×".join(comp.name for comp in self.components)
    
    @property
    def neutral(self) -> Tuple[Any, ...]:
        """Return the neutral element as a tuple of component neutrals."""
        return self._neutral
    
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
        return a == b  # all charge types support ==; tuple == is element-wise == in C
    
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


# --- Fast Paths ---
#
# Charge fusion sits in the innermost loops of block enumeration, so the generic
# implementations above — which loop over components, rebuild per-component
# charge lists, and call back into each component group — cost far more than the
# arithmetic they perform. The functions below specialize the three hot methods
# (`fuse_unique`, `fuse_channels`, `dual`) for the product shapes that occur in
# practice. `ProductGroup.__init__` picks one set once and installs it in the
# instance dict, where it shadows the class method, so no dispatch cost is paid
# per call. Every specialization must stay bit-identical to the generic method,
# including the Z2 `& 1` normalization.
#
# Each registry entry is a `(fuse_unique, fuse_channels, dual)` triple. A `None`
# in the `fuse_unique` slot means the generic method stays in place; this is used
# for products containing SU(2), where it must keep raising TypeError.


def _fuse_unique_u1_u1(*qs):
    """Fuse U1xU1 charges component-wise."""
    if len(qs) == 2:
        a, b = qs
        return (a[0] + b[0], a[1] + b[1])
    s0 = 0
    s1 = 0
    for q in qs:
        s0 += q[0]
        s1 += q[1]
    return (s0, s1)


def _fuse_unique_u1_z2(*qs):
    """Fuse U1xZ2 charges component-wise."""
    if len(qs) == 2:
        a, b = qs
        return (a[0] + b[0], (a[1] ^ b[1]) & 1)
    s0 = 0
    p1 = 0
    for q in qs:
        s0 += q[0]
        p1 ^= q[1] & 1
    return (s0, p1)


def _fuse_unique_z2_u1(*qs):
    """Fuse Z2xU1 charges component-wise."""
    if len(qs) == 2:
        a, b = qs
        return ((a[0] ^ b[0]) & 1, a[1] + b[1])
    p0 = 0
    s1 = 0
    for q in qs:
        p0 ^= q[0] & 1
        s1 += q[1]
    return (p0, s1)


def _fuse_unique_z2_z2(*qs):
    """Fuse Z2xZ2 charges component-wise."""
    if len(qs) == 2:
        a, b = qs
        return ((a[0] ^ b[0]) & 1, (a[1] ^ b[1]) & 1)
    p0 = 0
    p1 = 0
    for q in qs:
        p0 ^= q[0] & 1
        p1 ^= q[1] & 1
    return (p0, p1)


# The Abelian fuse_channels functions repeat the two-charge arithmetic instead
# of calling their fuse_unique counterpart: two charges is by far the common
# case, and the saved call is most of the cost of the operation.


def _fuse_channels_u1_u1(*qs):
    """Return the single U1xU1 fusion channel."""
    if len(qs) == 2:
        a, b = qs
        return ((a[0] + b[0], a[1] + b[1]),)
    if len(qs) == 1:
        return (qs[0],)
    return (_fuse_unique_u1_u1(*qs),)


def _fuse_channels_u1_z2(*qs):
    """Return the single U1xZ2 fusion channel."""
    if len(qs) == 2:
        a, b = qs
        return ((a[0] + b[0], (a[1] ^ b[1]) & 1),)
    if len(qs) == 1:
        return (qs[0],)
    return (_fuse_unique_u1_z2(*qs),)


def _fuse_channels_z2_u1(*qs):
    """Return the single Z2xU1 fusion channel."""
    if len(qs) == 2:
        a, b = qs
        return (((a[0] ^ b[0]) & 1, a[1] + b[1]),)
    if len(qs) == 1:
        return (qs[0],)
    return (_fuse_unique_z2_u1(*qs),)


def _fuse_channels_z2_z2(*qs):
    """Return the single Z2xZ2 fusion channel."""
    if len(qs) == 2:
        a, b = qs
        return (((a[0] ^ b[0]) & 1, (a[1] ^ b[1]) & 1),)
    if len(qs) == 1:
        return (qs[0],)
    return (_fuse_unique_z2_z2(*qs),)


def _fuse_channels_u1_su2(*qs):
    """Return all U1xSU2 fusion channels."""
    n = len(qs)
    if n == 2:
        # Two charges: the SU(2) triangular rule spans |ja - jb| .. ja + jb.
        a, b = qs
        charge = a[0] + b[0]
        ja = a[1]
        jb = b[1]
        lo = ja - jb if ja > jb else jb - ja
        return tuple((charge, j) for j in range(lo, ja + jb + 2, 2))
    if n < 2:
        return (qs[0],) if n else ((0, 0),)
    # n charges: accumulate the U1 sum and the SU(2) bounds in one pass.
    charge = 0
    j_sum = 0
    j_largest = 0
    for q in qs:
        charge += q[0]
        j = q[1]
        j_sum += j
        # An inline comparison rather than max(), which would cost a call per
        # charge on a path that runs once per block key.
        if j > j_largest:
            j_largest = j
    # Maximal cancellation has the largest spin opposing all the others.
    lo = max(0, 2 * j_largest - j_sum)
    # Stepping down from the maximum by 2 keeps the integrality of the channels.
    start = j_sum - 2 * ((j_sum - lo) // 2)
    return tuple((charge, j) for j in range(start, j_sum + 2, 2))


def _fuse_channels_z2_su2(*qs):
    """Return all Z2xSU2 fusion channels."""
    n = len(qs)
    if n == 2:
        a, b = qs
        parity = (a[0] ^ b[0]) & 1
        ja = a[1]
        jb = b[1]
        lo = ja - jb if ja > jb else jb - ja
        return tuple((parity, j) for j in range(lo, ja + jb + 2, 2))
    if n < 2:
        return (qs[0],) if n else ((0, 0),)
    parity = 0
    j_sum = 0
    j_largest = 0
    for q in qs:
        parity ^= q[0] & 1
        j = q[1]
        j_sum += j
        if j > j_largest:
            j_largest = j
    lo = max(0, 2 * j_largest - j_sum)
    start = j_sum - 2 * ((j_sum - lo) // 2)
    return tuple((parity, j) for j in range(start, j_sum + 2, 2))


def _dual_u1_u1(q):
    """Return the dual of a U1xU1 charge."""
    return (-q[0], -q[1])


def _dual_u1_z2(q):
    """Return the dual of a U1xZ2 charge."""
    return (-q[0], q[1] & 1)


def _dual_z2_u1(q):
    """Return the dual of a Z2xU1 charge."""
    return (q[0] & 1, -q[1])


def _dual_z2_z2(q):
    """Return the dual of a Z2xZ2 charge."""
    return (q[0] & 1, q[1] & 1)


def _dual_u1_su2(q):
    """Return the dual of a U1xSU2 charge (SU(2) irreps are self-dual)."""
    return (-q[0], q[1])


def _dual_z2_su2(q):
    """Return the dual of a Z2xSU2 charge (SU(2) irreps are self-dual)."""
    return (q[0] & 1, q[1])


# Keyed on the tuple of component *types*, so selection is an identity-hashed
# dict lookup rather than a comparison of group name strings.
_REGISTRY: dict = {
    (U1Group, U1Group): (_fuse_unique_u1_u1, _fuse_channels_u1_u1, _dual_u1_u1),
    (U1Group, Z2Group): (_fuse_unique_u1_z2, _fuse_channels_u1_z2, _dual_u1_z2),
    (Z2Group, U1Group): (_fuse_unique_z2_u1, _fuse_channels_z2_u1, _dual_z2_u1),
    (Z2Group, Z2Group): (_fuse_unique_z2_z2, _fuse_channels_z2_z2, _dual_z2_z2),
    (U1Group, SU2Group): (None, _fuse_channels_u1_su2, _dual_u1_su2),
    (Z2Group, SU2Group): (None, _fuse_channels_z2_su2, _dual_z2_su2),
}


# --- Generic Pairs ---
#
# Fallback for two-component products whose component types the registry does
# not list, user-defined groups included. Their arithmetic is unknown here and
# so cannot be inlined; instead each implementation is a closure built once per
# group that captures the component bound methods. The call path still avoids
# the attribute lookups and the per-component loop that the generic methods pay,
# at the cost of one call per component. The triples follow the registry
# convention, so `fuse_unique` is None whenever the second component is a
# UnitaryGroup. Products of three or more components get no fast path at all.


def _specialize_pair(first, second, has_unitary):
    """Build specialized fusion and dual functions for any two-component product.

    The returned closures capture the component bound methods, so the call path
    is free of attribute lookups and of the per-component loop that the generic
    methods run. This covers component types that `_REGISTRY` does not list,
    including user-defined groups.

    Parameters
    ----
    first, second:
        The two component groups, in order.
    has_unitary:
        Whether `second` is a `UnitaryGroup`, in which case fusion is
        multi-channel and no `fuse_unique` is provided.

    Returns
    ----
    tuple
        A `(fuse_unique, fuse_channels, dual)` triple, with `fuse_unique` set to
        `None` when `has_unitary` is true.
    """
    dual_first = first.dual
    dual_second = second.dual

    def dual(q):
        return (dual_first(q[0]), dual_second(q[1]))

    if has_unitary:
        return (*_specialize_unitary_channels(first, second), dual)
    else:
        return (*_specialize_abelian_fusion(first, second), dual)


def _specialize_unitary_channels(first, second):
    """Build fuse_unique and fuse_channels for an Abelian group times a UnitaryGroup."""
    fuse_first = first.fuse_unique
    channels_second = second.fuse_channels
    neutral = (first.neutral, second.neutral)

    def fuse_channels(*qs):
        if len(qs) < 2:
            return (qs[0],) if qs else (neutral,)
        fused_first = fuse_first(*[q[0] for q in qs])
        return tuple(
            (fused_first, ch) for ch in channels_second(*[q[1] for q in qs])
        )

    # Fusion is multi-channel here, so there is no unique outcome to specialize:
    # a None fuse_unique leaves the generic method in place, where it raises.
    return None, fuse_channels


def _specialize_abelian_fusion(first, second):
    """Build fuse_unique and fuse_channels for a pair of Abelian groups."""
    fuse_first = first.fuse_unique
    fuse_second = second.fuse_unique

    def fuse_unique(*qs):
        if len(qs) == 2:
            a, b = qs
            return (fuse_first(a[0], b[0]), fuse_second(a[1], b[1]))
        return (fuse_first(*[q[0] for q in qs]), fuse_second(*[q[1] for q in qs]))

    def fuse_channels(*qs):
        if len(qs) == 2:
            a, b = qs
            return ((fuse_first(a[0], b[0]), fuse_second(a[1], b[1])),)
        if len(qs) == 1:
            return (qs[0],)
        return (fuse_unique(*qs),)

    return fuse_unique, fuse_channels
