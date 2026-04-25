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


"""Bridge to delegate SU(2) Clebsch-Gordan tensor manipulation to yuzuha package.

This module provides the `Bridge` class, which serves as a storage container for
SU(2) Clebsch-Gordan tensor data using the yuzuha package's canonical bases.
The canonical bases are defined with respect to the outer multiplicity (OM) space,
which represents all valid fusion tree configurations.

Module-level `serialize` and `deserialize` convert `Bridge` instances to and from
plain dicts compatible with `torch.save` / `torch.load(..., weights_only=True)`.

Also provides thin delegates to yuzuha's recoupling routines: `fs_phase` produces
the Frobenius-Schur phase, `compute_xsymbol` computes the X-symbol for tensor
contraction, and `compute_rsymbol` computes the R-symbol for edge permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
import yuzuha

from ..typing import Charge, Direction
from .base import SymmetryGroup
from .product import ProductGroup
from .unitary import SU2Group


@dataclass(frozen=True)
class Bridge:
    """Storage container for SU(2) CG tensor data using yuzuha canonical bases.
    
    The Bridge class stores a CGSpec (Clebsch-Gordan specification) that identifies
    the canonical bases for the outer multiplicity (OM) space, along with a weight
    matrix that represents coefficients expanded according to these canonical bases.
    
    Attributes
    ----------
    cgspec : yuzuha.CGSpec
        Identifies the canonical bases for the OM space. Encapsulates:
        - External edge list (spins and directions)
        - All valid internal-spin tuples (α) for left-associative fusion trees
        - The full OM space dimension
    weights : torch.Tensor
        Coefficient matrix with shape (num_components, om_dimension).
        Each row represents coefficients expanded in the canonical basis.
        The canonical bases are normalized with respect to the OM space.
        num_components can be any positive integer (typically starts at 1).
        om_dimension is obtained from cgspec.om_dimension().
    
    device : torch.device
        Device where the weight matrix is stored.
    om_dimension : int
        Dimension of the outer multiplicity space (number of fusion tree configurations).
    num_components : int
        Number of component rows in the weight matrix (weights.shape[0]).
    num_external : int
        Number of external edges in the CGSpec.
    
    Examples
    --------
    Create a Bridge for a simple three-edge SU(2) tensor:
    
    >>> import yuzuha
    >>> import torch
    >>> from nicole.symmetry.delegate import Bridge
    >>> 
    >>> # Define edges: two incoming spin-1/2, one outgoing spin-1
    >>> j_half = yuzuha.Spin(1)  # 2j=1 for j=1/2
    >>> j_one = yuzuha.Spin(2)   # 2j=2 for j=1
    >>> edges = [
    ...     yuzuha.Edge.incoming(j_half),
    ...     yuzuha.Edge.incoming(j_half),
    ...     yuzuha.Edge.outgoing(j_one),
    ... ]
    >>> 
    >>> # Create CGSpec
    >>> cgspec = yuzuha.CGSpec.from_edges(edges)
    >>> 
    >>> # Initialize weight matrix (1 component)
    >>> om_dim = cgspec.om_dimension()
    >>> weights = torch.zeros(1, om_dim, dtype=torch.float64)
    >>> 
    >>> # Create Bridge
    >>> bridge = Bridge(cgspec, weights)
    >>> bridge.om_dimension
    1
    >>> bridge.num_components
    1
    >>> bridge.num_external
    3
    
    Notes
    -----
    - The OM space and canonical bases are fully managed by yuzuha.
    - Nicole's Direction.IN (value +1) maps to yuzuha.Edge.incoming (+1).
    - Nicole's Direction.OUT (value -1) maps to yuzuha.Edge.outgoing (-1).
    - Both Nicole and yuzuha use the 2j integer convention for SU(2) spins.
    """
    
    cgspec: yuzuha.CGSpec
    weights: torch.Tensor
    
    def __post_init__(self) -> None:
        """Validate weight matrix shape and type at construction time."""
        if not isinstance(self.weights, torch.Tensor):
            raise TypeError(f"weights must be a torch.Tensor, got {type(self.weights)}")
        
        if self.weights.ndim != 2:
            raise ValueError(
                f"weights must be a 2D tensor, got shape {self.weights.shape}"
            )
        
        if self.weights.shape[0] < 1:
            raise ValueError(
                f"weights must have at least 1 component (row), got shape {self.weights.shape}"
            )
        
        om_dim = self.cgspec.om_dimension()
        if self.weights.shape[1] != om_dim:
            raise ValueError(
                f"weights shape[1] must match OM dimension {om_dim}, "
                f"got shape {self.weights.shape}"
            )

    def __eq__(self, other: object) -> bool:
        """Return True if both cgspec and weights are exactly equal."""
        if not isinstance(other, Bridge):
            return NotImplemented
        return self.cgspec == other.cgspec and torch.equal(self.weights, other.weights)

    # Note: @dataclass(frozen=True) generates __hash__ regardless of an explicit
    # __eq__. The generated hash is identity-based for the torch.Tensor field and
    # therefore not consistent with value-based __eq__, but Bridge is only ever a
    # mapping value (in intw), never a key, so this causes no practical issues.

    @property
    def om_dimension(self) -> int:
        """Return the dimension of the outer multiplicity space."""
        return self.cgspec.om_dimension()
    
    @property
    def num_components(self) -> int:
        """Return the number of component rows in the weight matrix."""
        return self.weights.shape[0]
    
    @property
    def num_external(self) -> int:
        """Return the number of external edges in the CGSpec."""
        return self.cgspec.num_external()
    
    @property
    def device(self) -> torch.device:
        """Return the device where the weight matrix is stored."""
        return self.weights.device
    
    def to(
        self, 
        device: Union[str, torch.device], 
        dtype: Optional[torch.dtype] = None
    ) -> Bridge:
        """Move Bridge to a new device and optionally convert dtype.
        
        Parameters
        ----------
        device : str or torch.device
            Target device ('cpu', 'cuda', 'mps', etc.)
        dtype : torch.dtype, optional
            Target dtype. If None, preserves current dtype.
        
        Returns
        -------
        Bridge
            New Bridge instance with weights on the target device/dtype.
            If already on target device with target dtype, returns self.
        
        Examples
        --------
        >>> bridge = Bridge(cgspec, weights)
        >>> bridge_gpu = bridge.to('cuda')
        >>> bridge_gpu.device.type
        'cuda'
        """
        device = torch.device(device)
        
        # Check if already on target device/dtype
        if dtype is None:
            dtype = self.weights.dtype
        
        if device == self.device and dtype == self.weights.dtype:
            return self
        
        # Move weights to new device/dtype
        new_weights = self.weights.to(device, dtype=dtype)
        return Bridge(cgspec=self.cgspec, weights=new_weights)
    
    def clone(self) -> Bridge:
        """Create a deep copy of this Bridge with cloned weights.
        
        Returns
        -------
        Bridge
            New Bridge instance with cloned weight matrix.
        
        Examples
        --------
        >>> bridge = Bridge(cgspec, weights)
        >>> bridge_copy = bridge.clone()
        >>> bridge_copy.weights is bridge.weights
        False
        """
        return Bridge(cgspec=self.cgspec, weights=self.weights.clone())
    
    def conj(self) -> Bridge:
        """Return a new Bridge with flipped edge directions (conjugation).
        
        Creates a new Bridge instance where all edge directions in the CGSpec are
        flipped (incoming <-> outgoing). The cumulated Frobenius-Schur phase
        returned by `yuzuha.compute_conjugate` is multiplied into the weight
        matrix, so the resulting weights may differ from the original by a
        global factor of ±1.
        
        Returns
        -------
        Bridge
            New Bridge instance with flipped edge directions and weights scaled
            by the cumulated FS phase (±1).
        
        Examples
        --------
        >>> bridge = Bridge(cgspec, weights)
        >>> bridge_conj = bridge.conj()
        >>> # All edge directions are flipped
        >>> for orig, conj in zip(bridge.cgspec.edges, bridge_conj.cgspec.edges):
        ...     assert orig.dir == conj.dir.flip()
        """
        phase, conj_spec = yuzuha.compute_conjugate(self.cgspec)
        new_weights = self.weights * phase
        return Bridge(cgspec=conj_spec, weights=new_weights)
    
    def conj_phase(self) -> float:
        """Return the conjugation phase for this Bridge.
        
        Computes and returns the cumulated Frobenius-Schur phase (±1) between
        the conjugate of a CG basis and the canonical basis. This phase arises
        from the coupling tree structure and is essential for correct scalar
        contractions of SU(2) tensors.
        
        Returns
        -------
        float
            The cumulated FS phase, either +1.0 or -1.0.
        
        Examples
        --------
        >>> bridge = Bridge(cgspec, weights)
        >>> phase = bridge.conj_phase()
        >>> assert phase in [1.0, -1.0]
        """
        phase, _ = yuzuha.compute_conjugate(self.cgspec)
        return phase
    
    def invert_edges(self, positions: Sequence[int]) -> Bridge:
        """Return a new Bridge with edge directions inverted at the specified positions.
        
        Rebuilds the CGSpec with the selected edges' directions inverted, keeping
        the weight matrix unchanged. This is the building block for `Tensor.invert()`,
        which must be its own inverse (double application restores the original state).
        
        Parameters
        ----------
        positions : Sequence[int]
            0-based edge indices whose directions should be inverted.
        
        Returns
        -------
        Bridge
            New Bridge instance with the same spins and weights but with the
            specified edge directions inverted.
        """
        new_cgspec = self.cgspec.with_inverted_axes(list(positions))
        return Bridge(cgspec=new_cgspec, weights=self.weights)

    def insert_edge(self, position: int, direction: Direction) -> Bridge:
        """Return a new Bridge with a trivial neutral edge inserted at the given position.
        
        Inserts a neutral (spin-0) edge into the CGSpec at `position`. The
        insertion is performed in three steps:

        1. Insert the new edge at position 0 to obtain a well-defined CGSpec.
        2. Compute the R-symbol for the permutation that moves position 0 to
           `position` while keeping all other edges in their original order.
        3. Apply `new_weights = old_weights @ R` and return the permuted Bridge.

        Because the neutral representation does not participate in coupling,
        the OM dimension is preserved exactly.
        This is the building block for `Tensor.insert_index()` on SU(2) tensors.

        **Developer's note: Why insert at position 0 first, then permute?**

        In the SU(2) CG fusion tree the terminal edge (the last one) plays a
        distinguished role: it represents the total coupled representation of all
        preceding (leading) edges. Consequently, the CG coefficients depend on
        whether an edge is leading or terminal.

        Inserting the trivial j=0 edge directly at any leading position leaves the
        terminal edge unchanged, so no extra phase arises. Inserting it directly at
        the terminal position, however, demotes the previous terminal edge to a
        leading role, which can introduce a non-trivial recoupling phase (equal to
        `fs_phase(j_last)` when the old terminal is fermionic).

        By always inserting at position 0 first — where j=0 simply becomes the
        outermost leading edge and nothing else changes — we obtain a well-defined
        starting CGSpec with no ambiguity. We then compute the R-symbol for the
        permutation that slides the j=0 edge from position 0 to the requested
        `position`. This R-symbol captures exactly the phase (or lack thereof)
        produced by the change in leading/terminal status, making the result correct
        for every target position.

        Parameters
        ----------
        position : int
            0-based index at which to insert the new edge.
        direction : Direction
            Direction of the new neutral edge (Direction.IN or Direction.OUT).
        
        Returns
        -------
        Bridge
            New Bridge instance with the neutral edge inserted and weights
            updated by the appropriate R-symbol.
        """
        neutral_charge = yuzuha.Spin(0)
        if direction == Direction.IN:
            new_edge = yuzuha.Edge.incoming(neutral_charge)
        else:
            new_edge = yuzuha.Edge.outgoing(neutral_charge)

        # Step 1: insert at position 0
        edges_at_0 = [new_edge] + list(self.cgspec.edges)
        cgspec_at_0 = yuzuha.CGSpec.from_edges(edges_at_0)

        if position == 0:
            return Bridge(cgspec=cgspec_at_0, weights=self.weights)

        # Step 2: permutation [1, 2, ..., position, 0, position+1, ..., N-1]
        N = len(edges_at_0)
        perm = list(range(1, position + 1)) + [0] + list(range(position + 1, N))
        r_array, cgspec_final = yuzuha.compute_rsymbol(cgspec_at_0, perm)
        r_symbol = torch.from_numpy(r_array)
        r_symbol = r_symbol.to(device=self.weights.device, dtype=self.weights.dtype)

        # Step 3: apply R-symbol  (weights: num_components × om_dim)
        new_weights = self.weights @ r_symbol
        return Bridge(cgspec=cgspec_final, weights=new_weights)

    @staticmethod
    def from_block(
        group: SymmetryGroup,
        key: Tuple[Charge, ...],
        directions: Sequence[Direction],
        weights: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ) -> Bridge:
        """Construct a Bridge from a BlockKey and corresponding directions.
        
        This factory method creates a Bridge by extracting SU(2) charges from a
        BlockKey and creating the corresponding yuzuha CGSpec. Useful for converting
        Nicole tensor blocks into the yuzuha canonical basis representation.
        
        Parameters
        ----------
        group : SymmetryGroup
            The symmetry group. Must be either SU2Group or ProductGroup with
            SU2Group as the last component.
        key : Tuple[Charge, ...]
            BlockKey containing one charge per edge. For SU2Group, charges are
            integers (2j values). For ProductGroup with SU(2), charges are tuples
            where the last element is the 2j value.
        directions : Sequence[Direction]
            One direction per edge in the BlockKey. Length must match len(key).
        weights : torch.Tensor, optional
            Pre-initialized weight matrix. If provided, must have shape
            (num_components, om_dimension). If None, weights are initialized
            with shape (1, om_dimension) with the first element set to 1 and
            all other elements set to 0.
        dtype : torch.dtype, optional
            Data type for the weight matrix. Only used if weights is None.
            Defaults to torch.float64.
        device : torch.device or str, optional
            Device for the weight matrix. Only used if weights is None.
            If None, defaults to `torch.get_default_device()`.
        
        Returns
        -------
        Bridge
            New Bridge instance with CGSpec constructed from the block key and
            weights either provided or initialized with first element as 1.
        
        Raises
        ------
        TypeError
            If group is not SU2Group or ProductGroup with SU2Group, or if
            provided weights is not a torch.Tensor (via Bridge.__post_init__).
        ValueError
            If the number of directions doesn't match the number of charges,
            if provided weights has incorrect shape (via Bridge.__post_init__),
            or if yuzuha rejects the edge configuration.
        
        Examples
        --------
        >>> from nicole import SU2Group, Direction
        >>> from nicole.symmetry.delegate import Bridge
        >>> 
        >>> # Pure SU(2) group with default weights
        >>> group = SU2Group()
        >>> key = (1, 1, 2)  # Three edges: spin-1/2, spin-1/2, spin-1
        >>> directions = [Direction.IN, Direction.IN, Direction.OUT]
        >>> bridge = Bridge.from_block(group, key, directions)
        >>> bridge.weights[0, 0]  # First element is 1
        tensor(1.)
        >>> 
        >>> # ProductGroup with SU(2) and custom weights
        >>> import torch
        >>> from nicole import ProductGroup, U1Group
        >>> group = ProductGroup([U1Group(), SU2Group()])
        >>> key = ((0, 1), (-1, 1), (1, 2))  # Three edges with U1 and SU(2) charges
        >>> custom_weights = torch.randn(3, 1)
        >>> bridge = Bridge.from_block(group, key, directions, weights=custom_weights)
        >>> bridge.num_components
        3
        """
        # Validate inputs
        if len(key) != len(directions):
            raise ValueError(
                f"Number of charges in key ({len(key)}) must match "
                f"number of directions ({len(directions)})"
            )
        
        # Extract SU(2) charges based on group type
        if isinstance(group, SU2Group):
            # Pure SU(2): charges are already 2j values
            two_js = key
        elif isinstance(group, ProductGroup) and group._has_unitary:
            # ProductGroup with SU(2) at the end: extract last component of each charge
            two_js = tuple(charge[-1] if isinstance(charge, tuple) else charge for charge in key)
        else:
            raise TypeError(
                f"Bridge.from_block requires SU2Group or ProductGroup with SU2Group, "
                f"got {type(group).__name__}"
            )
        
        # Create yuzuha edges
        edges = []
        for two_j, direction in zip(two_js, directions):
            spin = yuzuha.Spin(two_j)
            # Nicole: Direction.IN = +1, Direction.OUT = -1
            # Yuzuha: incoming = +1, outgoing = -1
            if direction == Direction.IN:
                edges.append(yuzuha.Edge.incoming(spin))
            else:  # Direction.OUT
                edges.append(yuzuha.Edge.outgoing(spin))
        
        # Create CGSpec
        cgspec = yuzuha.CGSpec.from_edges(edges)
        om_dim = cgspec.om_dimension()
        
        # Initialize weight matrix if not provided
        if weights is None:
            if device is None:
                device = torch.get_default_device()
            device = torch.device(device)
            # Default: 1 component with first element as 1, rest as 0
            weights = torch.zeros(1, om_dim, dtype=dtype, device=device)
            weights[0, 0] = 1.0
        
        return Bridge(cgspec, weights)


# --- Serialization ---

def serialize(bridge: Bridge) -> dict:
    """Serialize a Bridge to a plain dict of primitives and a `torch.Tensor`.

    The returned dict is safe to pass to `torch.save` /
    `torch.load(..., weights_only=True)`.

    Parameters
    ----------
    bridge : Bridge
        The Bridge to serialize.

    Returns
    -------
    dict
        `{"edges": tuple of (two_j: int, dir_sign: int), "weights": torch.Tensor}`
        where each `(two_j, dir_sign)` pair encodes the doubled SU(2) spin
        and the edge direction (+1 incoming, -1 outgoing).
    """
    spins = bridge.cgspec.get_spins()
    dirs = bridge.cgspec.get_directions()
    return {
        "edges": tuple((int(s), int(d)) for s, d in zip(spins, dirs)),
        "weights": bridge.weights,
    }


def deserialize(
    d: dict,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
) -> Bridge:
    """Reconstruct a Bridge from a dict produced by `serialize`.

    Parameters
    ----------
    d : dict
        Dict that includes the entries produced by `serialize` (i.e. must
        contain at least `"edges"` and `"weights"`).
    device : str or torch.device, optional
        Device to place the weight tensor on. Defaults to `"cpu"`.
    dtype : torch.dtype, optional
        Dtype to cast the weight tensor to. If `None`, the tensor's
        existing dtype is preserved.

    Returns
    -------
    Bridge
        Reconstructed Bridge with weights on `device`.
    """
    edges = []
    for two_j, dir_sign in d["edges"]:
        spin = yuzuha.Spin(two_j)
        if dir_sign == 1:
            edges.append(yuzuha.Edge.incoming(spin))
        else:
            edges.append(yuzuha.Edge.outgoing(spin))

    cgspec = yuzuha.CGSpec.from_edges(edges)
    weights = d["weights"].to(device=device, dtype=dtype) if dtype is not None \
        else d["weights"].to(device=device)
    return Bridge(cgspec=cgspec, weights=weights)


# --- Backend Relays ---

def fs_phase(group: SymmetryGroup, charge: Charge) -> float:
    """Return the Frobenius-Schur phase (-1)^{2j} for the SU(2) part of a charge.

    For Abelian groups, returns 1.0. For SU2Group or ProductGroup with SU2Group,
    extracts the 2j value and returns (-1)^{2j}.

    Parameters
    ----------
    group : SymmetryGroup
        The symmetry group used to interpret the charge.
    charge : Charge
        A single block charge (at one index position). For SU2Group, an integer
        (2j). For ProductGroup with SU2Group, a tuple whose last element is 2j.

    Returns
    -------
    float
        The FS phase: +1.0 or -1.0.
    """
    if group.is_abelian:
        return 1.0
    if isinstance(group, SU2Group):
        two_j = charge
    elif isinstance(group, ProductGroup) and group._has_unitary:
        two_j = charge[-1] if isinstance(charge, tuple) else charge
    else:
        return 1.0
    return float(yuzuha.fs_phase_for_spin(yuzuha.Spin(two_j)))


def compute_xsymbol(
    bridge_a: Bridge,
    bridge_b: Bridge,
    axes_a: Sequence[int],
    axes_b: Sequence[int],
) -> Tuple[torch.Tensor, yuzuha.CGSpec]:
    """Compute X-symbol for contracting two CG tensors.
    
    The X-symbol encodes recoupling coefficients for contracting two CG tensors.
    Given two Bridges with OM bases α and β, and a contraction specification,
    this returns X^γ_{αβ} coefficients and the output CGSpec.
    
    Parameters
    ----------
    bridge_a : Bridge
        First Bridge to contract.
    bridge_b : Bridge
        Second Bridge to contract.
    axes_a : Sequence[int]
        Edge indices from bridge_a to contract (0-indexed).
    axes_b : Sequence[int]
        Edge indices from bridge_b to contract (0-indexed).
        Must have the same length as axes_a, with matching spins.
    
    Returns
    -------
    x_symbol : torch.Tensor
        X-symbol array with shape (om_a, om_b, om_c) as a torch tensor.
    spec_c : yuzuha.CGSpec
        CGSpec for the output (uncontracted edges).
    
    Raises
    ------
    ValueError
        If contraction is invalid (mismatched spins, invalid axes).
    RuntimeError
        On yuzuha internal errors (e.g., cache issues).
    
    Examples
    --------
    >>> from nicole import SU2Group, Direction
    >>> from nicole.symmetry.delegate import Bridge, compute_xsymbol
    >>> 
    >>> group = SU2Group()
    >>> 
    >>> # Bridge A: two spin-1/2 in, one spin-1 out
    >>> key_a = (1, 1, 2)
    >>> dirs_a = [Direction.IN, Direction.IN, Direction.OUT]
    >>> bridge_a = Bridge.from_block(group, key_a, dirs_a)
    >>> 
    >>> # Bridge B: one spin-1 in, one spin-1/2 in, one spin-1/2 out
    >>> key_b = (2, 1, 1)
    >>> dirs_b = [Direction.IN, Direction.IN, Direction.OUT]
    >>> bridge_b = Bridge.from_block(group, key_b, dirs_b)
    >>> 
    >>> # Contract on spin-1 edges
    >>> x_symbol, spec_c = compute_xsymbol(bridge_a, bridge_b, [2], [0])
    >>> x_symbol.shape
    torch.Size([1, 1, 1])
    """
    contraction = yuzuha.Contraction(list(axes_a), list(axes_b))
    x_array, spec_c = yuzuha.compute_xsymbol(bridge_a.cgspec, bridge_b.cgspec, contraction)
    
    x_symbol = torch.from_numpy(x_array)
    x_symbol = x_symbol.to(device=bridge_a.weights.device, dtype=bridge_a.weights.dtype)
    return x_symbol, spec_c


def compute_rsymbol(
    bridge: Bridge,
    permutation: Sequence[int],
) -> Tuple[torch.Tensor, yuzuha.CGSpec]:
    """Compute R-symbol for permuting external edges of a CG tensor.
    
    The R-symbol describes how OM indices transform when external edges are
    permuted. Returns a unitary matrix R^β_α and the permuted CGSpec.
    
    Parameters
    ----------
    bridge : Bridge
        Bridge whose edges to permute.
    permutation : Sequence[int]
        Permutation of edge indices. Must be a valid permutation of
        [0, 1, ..., num_external-1], with each index appearing exactly once.
    
    Returns
    -------
    r_symbol : torch.Tensor
        R-symbol array with shape (om_original, om_permuted) as a torch tensor.
        For SU(2) this is real and unitary (R†R = I, RR† = I).
    spec_permuted : yuzuha.CGSpec
        CGSpec with edges permuted according to the permutation.
    
    Raises
    ------
    ValueError
        If permutation is invalid (wrong length, duplicate indices, out of range).
    RuntimeError
        On yuzuha internal errors.
    
    Examples
    --------
    >>> from nicole import SU2Group, Direction
    >>> from nicole.symmetry.delegate import Bridge, compute_rsymbol
    >>> 
    >>> group = SU2Group()
    >>> key = (1, 1, 2)
    >>> directions = [Direction.IN, Direction.IN, Direction.OUT]
    >>> bridge = Bridge.from_block(group, key, directions)
    >>> 
    >>> # Swap first two edges
    >>> r_symbol, spec_perm = compute_rsymbol(bridge, [1, 0, 2])
    >>> r_symbol.shape
    torch.Size([1, 1])
    """
    r_array, spec_permuted = yuzuha.compute_rsymbol(bridge.cgspec, list(permutation))
    
    r_symbol = torch.from_numpy(r_array)
    r_symbol = r_symbol.to(device=bridge.weights.device, dtype=bridge.weights.dtype)
    return r_symbol, spec_permuted
