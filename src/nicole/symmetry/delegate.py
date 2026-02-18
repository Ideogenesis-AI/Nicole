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
from typing import Optional, Sequence, Tuple

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
