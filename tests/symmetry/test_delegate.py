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


"""Tests for Bridge class and yuzuha integration."""

import pytest
import torch
import yuzuha

from nicole import Direction, ProductGroup, SU2Group, U1Group
from nicole.symmetry.delegate import Bridge, compute_xsymbol, compute_rsymbol


def test_bridge_basic_instantiation():
    """Test basic Bridge instantiation with valid CGSpec and weights."""
    j_half = yuzuha.Spin(1)
    j_one = yuzuha.Spin(2)
    
    edges = [
        yuzuha.Edge.incoming(j_half),
        yuzuha.Edge.incoming(j_half),
        yuzuha.Edge.outgoing(j_one),
    ]
    
    cgspec = yuzuha.CGSpec.from_edges(edges)
    om_dim = cgspec.om_dimension()
    
    weights = torch.zeros(1, om_dim, dtype=torch.float64)
    bridge = Bridge(cgspec, weights)
    
    assert bridge.cgspec == cgspec
    assert bridge.weights.shape == (1, om_dim)
    assert bridge.om_dimension == om_dim
    assert bridge.num_components == 1
    assert bridge.num_external == 3


def test_bridge_multiple_components():
    """Test Bridge with multiple component rows."""
    j_one = yuzuha.Spin(2)
    
    edges = [
        yuzuha.Edge.incoming(j_one),
        yuzuha.Edge.incoming(j_one),
        yuzuha.Edge.outgoing(j_one),
    ]
    
    cgspec = yuzuha.CGSpec.from_edges(edges)
    om_dim = cgspec.om_dimension()
    
    weights = torch.randn(5, om_dim, dtype=torch.float64)
    bridge = Bridge(cgspec, weights)
    
    assert bridge.num_components == 5
    assert bridge.om_dimension == om_dim
    assert bridge.weights.shape == (5, om_dim)


def test_bridge_various_dtypes():
    """Test Bridge with different torch dtypes."""
    j_half = yuzuha.Spin(1)
    j_one = yuzuha.Spin(2)
    
    edges = [
        yuzuha.Edge.incoming(j_half),
        yuzuha.Edge.incoming(j_half),
        yuzuha.Edge.outgoing(j_one),
    ]
    
    cgspec = yuzuha.CGSpec.from_edges(edges)
    om_dim = cgspec.om_dimension()
    
    for dtype in [torch.float32, torch.float64, torch.complex64, torch.complex128]:
        weights = torch.zeros(1, om_dim, dtype=dtype)
        bridge = Bridge(cgspec, weights)
        assert bridge.weights.dtype == dtype


def test_bridge_invalid_weights_not_tensor():
    """Test Bridge raises TypeError when weights is not a tensor."""
    j_half = yuzuha.Spin(1)
    j_one = yuzuha.Spin(2)
    edges = [yuzuha.Edge.incoming(j_half), yuzuha.Edge.incoming(j_half), yuzuha.Edge.outgoing(j_one)]
    cgspec = yuzuha.CGSpec.from_edges(edges)
    
    with pytest.raises(TypeError, match="weights must be a torch.Tensor"):
        Bridge(cgspec, [[1.0, 2.0]])


def test_bridge_invalid_weights_not_2d():
    """Test Bridge raises ValueError when weights is not 2D."""
    j_half = yuzuha.Spin(1)
    j_one = yuzuha.Spin(2)
    edges = [yuzuha.Edge.incoming(j_half), yuzuha.Edge.incoming(j_half), yuzuha.Edge.outgoing(j_one)]
    cgspec = yuzuha.CGSpec.from_edges(edges)
    om_dim = cgspec.om_dimension()
    
    # 1D tensor
    weights_1d = torch.zeros(om_dim, dtype=torch.float64)
    with pytest.raises(ValueError, match="weights must be a 2D tensor"):
        Bridge(cgspec, weights_1d)
    
    # 3D tensor
    weights_3d = torch.zeros(1, om_dim, 2, dtype=torch.float64)
    with pytest.raises(ValueError, match="weights must be a 2D tensor"):
        Bridge(cgspec, weights_3d)


def test_bridge_invalid_weights_zero_components():
    """Test Bridge raises ValueError when weights has zero rows."""
    j_half = yuzuha.Spin(1)
    j_one = yuzuha.Spin(2)
    edges = [yuzuha.Edge.incoming(j_half), yuzuha.Edge.incoming(j_half), yuzuha.Edge.outgoing(j_one)]
    cgspec = yuzuha.CGSpec.from_edges(edges)
    om_dim = cgspec.om_dimension()
    
    weights = torch.zeros(0, om_dim, dtype=torch.float64)
    with pytest.raises(ValueError, match="weights must have at least 1 component"):
        Bridge(cgspec, weights)


def test_bridge_invalid_weights_wrong_om_dimension():
    """Test Bridge raises ValueError when weights shape doesn't match OM dimension."""
    j_half = yuzuha.Spin(1)
    j_one = yuzuha.Spin(2)
    edges = [yuzuha.Edge.incoming(j_half), yuzuha.Edge.incoming(j_half), yuzuha.Edge.outgoing(j_one)]
    cgspec = yuzuha.CGSpec.from_edges(edges)
    om_dim = cgspec.om_dimension()
    
    # Wrong OM dimension
    weights = torch.zeros(1, om_dim + 5, dtype=torch.float64)
    with pytest.raises(ValueError, match=f"weights shape\\[1\\] must match OM dimension {om_dim}"):
        Bridge(cgspec, weights)


def test_bridge_complex_edges():
    """Test Bridge with more complex edge configurations."""
    j_half = yuzuha.Spin(1)
    j_one = yuzuha.Spin(2)
    j_three_half = yuzuha.Spin(3)
    
    # Valid configuration: 1/2 ⊗ 3/2 ⊗ 1 can couple to 0
    # Possible paths: (1/2 ⊗ 3/2) = 1,2 then 1⊗1=0,1,2 or 2⊗1=1,2,3
    edges = [
        yuzuha.Edge.incoming(j_half),
        yuzuha.Edge.incoming(j_three_half),
        yuzuha.Edge.incoming(j_one),
        yuzuha.Edge.outgoing(j_one),
    ]
    
    cgspec = yuzuha.CGSpec.from_edges(edges)
    om_dim = cgspec.om_dimension()
    
    weights = torch.ones(3, om_dim, dtype=torch.float64)
    bridge = Bridge(cgspec, weights)
    
    assert bridge.num_external == 4
    assert bridge.num_components == 3
    assert bridge.om_dimension == om_dim


def test_bridge_frozen():
    """Test that Bridge is immutable (frozen dataclass)."""
    j_half = yuzuha.Spin(1)
    j_one = yuzuha.Spin(2)
    edges = [yuzuha.Edge.incoming(j_half), yuzuha.Edge.incoming(j_half), yuzuha.Edge.outgoing(j_one)]
    cgspec = yuzuha.CGSpec.from_edges(edges)
    om_dim = cgspec.om_dimension()
    
    weights = torch.zeros(1, om_dim, dtype=torch.float64)
    bridge = Bridge(cgspec, weights)
    
    with pytest.raises(AttributeError):
        bridge.weights = torch.ones(1, om_dim, dtype=torch.float64)


def test_bridge_from_block_su2group():
    """Test from_block constructor with pure SU2Group."""
    group = SU2Group()
    
    # BlockKey: three edges with spins 1/2, 1/2, 1 (2j: 1, 1, 2)
    key = (1, 1, 2)
    directions = [Direction.IN, Direction.IN, Direction.OUT]
    
    bridge = Bridge.from_block(group, key, directions)
    
    assert bridge.num_external == 3
    assert bridge.num_components == 1
    assert bridge.om_dimension == 1  # Only one way to couple to j=0
    assert bridge.weights.shape == (1, 1)
    assert bridge.weights.dtype == torch.float64


def test_bridge_from_block_product_group():
    """Test from_block constructor with ProductGroup(U1×SU2)."""
    group = ProductGroup([U1Group(), SU2Group()])
    
    # BlockKey: three edges with (U1, SU2) charges
    # (0, 1), (1, 1), (-1, 2) - U1 charges and SU(2) 2j values
    key = ((0, 1), (1, 1), (-1, 2))
    directions = [Direction.IN, Direction.IN, Direction.OUT]
    
    bridge = Bridge.from_block(group, key, directions)
    
    assert bridge.num_external == 3
    assert bridge.num_components == 1
    assert bridge.om_dimension == 1
    assert bridge.weights.dtype == torch.float64
    
    # Verify CGSpec has correct spins (extracted from SU(2) part)
    spins = bridge.cgspec.get_spins()
    assert spins == [1, 1, 2]


def test_bridge_from_block_custom_dtype():
    """Test from_block with custom dtype."""
    group = SU2Group()
    key = (1, 1, 2)
    directions = [Direction.IN, Direction.IN, Direction.OUT]
    
    bridge = Bridge.from_block(group, key, directions, dtype=torch.complex128)
    
    assert bridge.num_components == 1
    assert bridge.weights.dtype == torch.complex128
    assert bridge.weights.shape == (1, 1)


def test_bridge_from_block_invalid_group():
    """Test from_block raises TypeError for non-SU(2) groups."""
    group = U1Group()
    key = (0, 1, -1)
    directions = [Direction.IN, Direction.IN, Direction.OUT]
    
    with pytest.raises(TypeError, match="Bridge.from_block requires SU2Group or ProductGroup with SU2Group"):
        Bridge.from_block(group, key, directions)


def test_bridge_from_block_mismatched_lengths():
    """Test from_block raises ValueError when key and directions lengths differ."""
    group = SU2Group()
    key = (1, 1, 2)
    directions = [Direction.IN, Direction.OUT]  # Only 2 directions for 3 charges
    
    with pytest.raises(ValueError, match="Number of charges in key .* must match number of directions"):
        Bridge.from_block(group, key, directions)


def test_bridge_from_block_various_directions():
    """Test from_block with different direction combinations."""
    group = SU2Group()
    
    # All IN
    key = (2, 2, 2, 2)  # Four spin-1 edges coupling to j=0
    directions = [Direction.IN, Direction.IN, Direction.IN, Direction.IN]
    bridge1 = Bridge.from_block(group, key, directions)
    assert bridge1.num_external == 4
    
    # Mixed directions
    key = (1, 1, 2)
    directions = [Direction.OUT, Direction.OUT, Direction.IN]
    bridge2 = Bridge.from_block(group, key, directions)
    assert bridge2.num_external == 3


def test_bridge_from_block_default_weights():
    """Test from_block default weights initialization (first element 1, rest 0)."""
    group = SU2Group()
    key = (2, 2, 2, 2)  # Four spin-1 edges
    directions = [Direction.IN, Direction.IN, Direction.IN, Direction.IN]
    
    bridge = Bridge.from_block(group, key, directions)
    
    # Check shape: always 1 component by default
    assert bridge.num_components == 1
    
    # Check first element is 1, rest are 0
    assert bridge.weights[0, 0] == 1.0
    
    # Check all other elements are 0
    if bridge.om_dimension > 1:
        assert torch.all(bridge.weights[0, 1:] == 0.0)


def test_bridge_from_block_provided_weights():
    """Test from_block with provided custom weights."""
    group = SU2Group()
    key = (1, 1, 2)
    directions = [Direction.IN, Direction.IN, Direction.OUT]
    
    # Create custom weights
    custom_weights = torch.tensor([[0.5, 0.8]], dtype=torch.float64)
    
    # Since om_dimension for this config might be 1, let's first check
    edges = [yuzuha.Edge.incoming(yuzuha.Spin(1)), 
             yuzuha.Edge.incoming(yuzuha.Spin(1)), 
             yuzuha.Edge.outgoing(yuzuha.Spin(2))]
    cgspec = yuzuha.CGSpec.from_edges(edges)
    om_dim = cgspec.om_dimension()
    
    # Create appropriate custom weights
    custom_weights = torch.randn(2, om_dim, dtype=torch.float64)
    
    bridge = Bridge.from_block(group, key, directions, weights=custom_weights)
    
    assert bridge.num_components == 2
    assert torch.equal(bridge.weights, custom_weights)


def test_bridge_from_block_provided_weights_validation():
    """Test from_block validates provided weights shape."""
    group = SU2Group()
    key = (1, 1, 2)
    directions = [Direction.IN, Direction.IN, Direction.OUT]
    
    # Get actual om_dimension
    edges = [yuzuha.Edge.incoming(yuzuha.Spin(1)), 
             yuzuha.Edge.incoming(yuzuha.Spin(1)), 
             yuzuha.Edge.outgoing(yuzuha.Spin(2))]
    cgspec = yuzuha.CGSpec.from_edges(edges)
    om_dim = cgspec.om_dimension()
    
    # Wrong shape: wrong om_dimension
    bad_weights = torch.zeros(1, om_dim + 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="weights shape\\[1\\] must match OM dimension"):
        Bridge.from_block(group, key, directions, weights=bad_weights)
    
    # Wrong shape: 1D
    bad_weights_1d = torch.zeros(om_dim, dtype=torch.float64)
    with pytest.raises(ValueError, match="weights must be a 2D tensor"):
        Bridge.from_block(group, key, directions, weights=bad_weights_1d)
    
    # Wrong type
    with pytest.raises(TypeError, match="weights must be a torch.Tensor"):
        Bridge.from_block(group, key, directions, weights=[[1.0, 2.0]])


def test_compute_xsymbol_basic():
    """Test compute_xsymbol for basic contraction."""
    group = SU2Group()
    
    # Bridge A: two spin-1/2 in, one spin-1 out
    key_a = (1, 1, 2)
    dirs_a = [Direction.IN, Direction.IN, Direction.OUT]
    bridge_a = Bridge.from_block(group, key_a, dirs_a)
    
    # Bridge B: one spin-1 in, one spin-1/2 in, one spin-1/2 out
    key_b = (2, 1, 1)
    dirs_b = [Direction.IN, Direction.IN, Direction.OUT]
    bridge_b = Bridge.from_block(group, key_b, dirs_b)
    
    # Contract spin-1 edges (A's edge 2 with B's edge 0)
    x_symbol, spec_c = compute_xsymbol(bridge_a, bridge_b, [2], [0])
    
    # Check result properties
    assert isinstance(x_symbol, torch.Tensor)
    assert x_symbol.ndim == 3
    assert x_symbol.shape[0] == bridge_a.om_dimension
    assert x_symbol.shape[1] == bridge_b.om_dimension
    assert x_symbol.shape[2] == spec_c.om_dimension()
    
    # Check output CGSpec
    assert spec_c.num_external() == 4  # 2 from A + 2 from B (minus 2 contracted)


def test_compute_xsymbol_multiple_edges():
    """Test compute_xsymbol contracting multiple edges."""
    group = SU2Group()
    
    # Bridge A: three spin-1 in, one spin-1 out
    key_a = (2, 2, 2, 2)
    dirs_a = [Direction.IN, Direction.IN, Direction.IN, Direction.OUT]
    bridge_a = Bridge.from_block(group, key_a, dirs_a)
    
    # Bridge B: two spin-1 in, two spin-1 out
    key_b = (2, 2, 2, 2)
    dirs_b = [Direction.IN, Direction.IN, Direction.OUT, Direction.OUT]
    bridge_b = Bridge.from_block(group, key_b, dirs_b)
    
    # Contract two spin-1 edges
    x_symbol, spec_c = compute_xsymbol(bridge_a, bridge_b, [0, 1], [2, 3])
    
    assert isinstance(x_symbol, torch.Tensor)
    assert x_symbol.ndim == 3
    assert spec_c.num_external() == 4  # 2 from A + 2 from B


def test_compute_rsymbol_basic():
    """Test compute_rsymbol for basic permutation."""
    group = SU2Group()
    
    key = (1, 1, 2)
    directions = [Direction.IN, Direction.IN, Direction.OUT]
    bridge = Bridge.from_block(group, key, directions)
    
    # Identity permutation
    r_symbol, spec_perm = compute_rsymbol(bridge, [0, 1, 2])
    
    assert isinstance(r_symbol, torch.Tensor)
    assert r_symbol.ndim == 2
    assert r_symbol.shape[0] == bridge.om_dimension
    assert r_symbol.shape[1] == spec_perm.om_dimension()
    
    # For identity permutation, R should be identity matrix
    assert torch.allclose(r_symbol, torch.eye(bridge.om_dimension, dtype=r_symbol.dtype))


def test_compute_rsymbol_swap():
    """Test compute_rsymbol for edge swap."""
    group = SU2Group()
    
    # Use a configuration with non-trivial OM space
    key = (2, 2, 2, 2)
    directions = [Direction.IN, Direction.IN, Direction.IN, Direction.IN]
    bridge = Bridge.from_block(group, key, directions)
    
    # Swap first two edges
    r_symbol, spec_perm = compute_rsymbol(bridge, [1, 0, 2, 3])
    
    assert isinstance(r_symbol, torch.Tensor)
    assert r_symbol.shape[0] == bridge.om_dimension
    assert r_symbol.shape[1] == spec_perm.om_dimension()
    
    # R-symbol should be unitary (R†R = I)
    if r_symbol.shape[0] > 1:
        identity = torch.matmul(r_symbol.T, r_symbol)
        assert torch.allclose(identity, torch.eye(r_symbol.shape[1], dtype=r_symbol.dtype), atol=1e-10)


def test_compute_rsymbol_invalid_permutation():
    """Test compute_rsymbol raises on invalid permutation."""
    group = SU2Group()
    key = (1, 1, 2)
    directions = [Direction.IN, Direction.IN, Direction.OUT]
    bridge = Bridge.from_block(group, key, directions)
    
    # Wrong length
    with pytest.raises(ValueError):
        compute_rsymbol(bridge, [0, 1])
    
    # Duplicate index
    with pytest.raises(ValueError):
        compute_rsymbol(bridge, [0, 0, 2])
