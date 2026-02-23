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

"""Bridge to delegate SU(2) Clebsch-Gordan tensor manipulation to yuzuha package.

This module provides the Bridge class, which serves as a storage container for
SU(2) Clebsch-Gordan tensor data using the yuzuha package's canonical bases.
The canonical bases are defined with respect to the outer multiplicity (OM) space,
which represents all valid fusion tree configurations.
"""

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
    
    Properties
    ----------
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
        flipped (incoming <-> outgoing), while the weight matrix is shared (not cloned).
        This corresponds to complex conjugation of the physical tensor in the sense
        of flipping all index directions.
        
        Returns
        -------
        Bridge
            New Bridge instance with flipped edge directions and the same weight tensor.
        
        Examples
        --------
        >>> bridge = Bridge(cgspec, weights)
        >>> bridge_conj = bridge.conj()
        >>> # All edge directions are flipped
        >>> for orig, conj in zip(bridge.cgspec.edges, bridge_conj.cgspec.edges):
        ...     assert orig.dir == conj.dir.flip()
        >>> # Weights are the same object (shared)
        >>> assert bridge.weights is bridge_conj.weights
        """
        # Flip all edge directions
        flipped_edges = []
        for edge in self.cgspec.edges:
            if edge.dir.is_incoming():  # incoming
                flipped_edges.append(yuzuha.Edge.outgoing(edge.j))
            else:  # outgoing
                flipped_edges.append(yuzuha.Edge.incoming(edge.j))
        
        # Create new CGSpec with flipped directions
        new_cgspec = yuzuha.CGSpec.from_edges(flipped_edges)
        
        # Return new Bridge with flipped cgspec and same weights
        return Bridge(cgspec=new_cgspec, weights=self.weights)
    
    @staticmethod
    def from_block(
        group: SymmetryGroup,
        key: Tuple[Charge, ...],
        directions: Sequence[Direction],
        weights: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float64,
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
            # Default: 1 component with first element as 1, rest as 0
            weights = torch.zeros(1, om_dim, dtype=dtype)
            weights[0, 0] = 1.0
        
        return Bridge(cgspec, weights)


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
    return r_symbol, spec_permuted
