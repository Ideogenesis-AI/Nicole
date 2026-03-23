# Copyright (C) 2025-2026 Changkai Zhang.
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


"""Shared test utilities and helper functions for the Nicole test suite.

This module provides convenience functions for performing common assertions
on symmetry-aware tensor structures.

Functions
---------
assert_charge_neutral:
    Verify that all blocks in a tensor satisfy charge neutrality.
assert_blocks_equal:
    Compare two tensors for identical block structure and numerical contents.
assert_data_weights_equal:
    Assert that two SU(2) tensors have identical data and weights block-by-block.
assert_physical_tensors_equal:
    Assert that two SU(2) tensors have identical physical tensors R@W block-by-block.

Comparison Guidelines
---------------------
For tensor comparisons in tests, use PyTorch's native comparison functions:

- torch.allclose(a, b): Preferred for most assertions. Uses default tolerances
  (rtol=1e-05, atol=1e-08) unless explicitly specified. Specify stricter
  tolerances only where higher precision is required (e.g., atol=1e-14 for
  identity matrices, atol=1e-10 for eigenvalue decompositions).

- torch.isclose(a, b): For element-wise comparisons returning boolean tensors.

- math.isclose(a, b): For comparing Python scalars (floats). Cleaner than
  converting to tensors. Uses rel_tol=1e-09, abs_tol=0.0 by default

- torch.equal(a, b): For exact equality checks (no tolerance).

Avoid using abs() for manual comparisons. Use PyTorch's built-in comparison
functions which handle edge cases (NaN, inf) correctly and provide consistent
numerical behavior.
"""

from __future__ import annotations

import torch

from nicole import Tensor
from nicole.blocks import BlockSchema
from nicole.symmetry.delegate import Bridge


def assert_charge_neutral(tensor: Tensor) -> None:
    """Verify that all blocks in a tensor satisfy charge neutrality.

    Parameters
    ----------
    tensor:
        Tensor to validate for charge neutrality.
    """
    if not tensor.indices:
        return
    
    for key in tensor.data:
        assert BlockSchema.charges_conserved(tensor.indices, key)


def assert_blocks_equal(a: Tensor, b: Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> None:
    """Verify that two tensors have identical block structure and numerical contents.

    Parameters
    ----------
    a, b:
        Tensors to compare for block equality.
    rtol, atol:
        Relative and absolute tolerances passed to torch.allclose.
        Defaults match PyTorch's own defaults (rtol=1e-05, atol=1e-08).
    """
    assert set(a.data.keys()) == set(b.data.keys())
    for key in a.data:
        assert torch.allclose(a.data[key], b.data[key], rtol=rtol, atol=atol)


def assert_data_weights_equal(tensor1: Tensor, tensor2: Tensor, rtol: float = 1e-10, atol: float = 1e-12, msg: str = "") -> None:
    """Assert that two SU(2) tensors have identical data and weights block-by-block.
    
    Parameters
    ----------
    tensor1, tensor2:
        Tensors to compare
    rtol, atol:
        Relative and absolute tolerances for torch.allclose
    msg:
        Optional message prefix for assertion errors
    """
    prefix = f"{msg}: " if msg else ""
    
    assert set(tensor1.data.keys()) == set(tensor2.data.keys()), \
        f"{prefix}Block keys should match"
    
    for key in tensor1.data.keys():
        assert torch.allclose(tensor1.intw[key].weights, tensor2.intw[key].weights,
                            rtol=rtol, atol=atol), \
            f"{prefix}Block {key} weights should match"
        
        assert torch.allclose(tensor1.data[key], tensor2.data[key],
                            rtol=rtol, atol=atol), \
            f"{prefix}Block {key} data should match"


def assert_physical_tensors_equal(tensor1: Tensor, tensor2: Tensor, rtol: float = 1e-10, atol: float = 1e-12, msg: str = "") -> None:
    """Assert that two SU(2) tensors have identical physical tensors R@W block-by-block.
    
    This comparison is appropriate when weights may differ due to gauge freedom,
    but the physical tensor (reduced tensor times weights) should be preserved.
    
    Parameters
    ----------
    tensor1, tensor2:
        Tensors to compare
    rtol, atol:
        Relative and absolute tolerances for torch.allclose
    msg:
        Optional message prefix for assertion errors
    """
    prefix = f"{msg}: " if msg else ""
    
    assert set(tensor1.data.keys()) == set(tensor2.data.keys()), \
        f"{prefix}Block keys should match"
    
    for key in tensor1.data.keys():
        block1 = tensor1.data[key]
        block2 = tensor2.data[key]
        weights1 = tensor1.intw[key].weights
        weights2 = tensor2.intw[key].weights
        
        # Flatten spatial dimensions: (d₁, ..., dₙ, r) → (d, r)
        block1_flat = block1.flatten(0, -2)
        block2_flat = block2.flatten(0, -2)
        
        # Compute physical tensors: (d, om_dim)
        physical1 = block1_flat @ weights1
        physical2 = block2_flat @ weights2
        
        assert torch.allclose(physical1, physical2, rtol=rtol, atol=atol), \
            f"{prefix}Block {key} physical tensors R@W should match"


def populate_random_weights(tensor: Tensor, seed: int, min_components: int = 3, max_components: int = 6) -> None:
    """Populate SU(2) tensor with random component counts and weights.
    
    Expands each block to have multiple components (between min_components and max_components)
    with randomized weights. Uses a local generator to avoid affecting global random state.

    Parameters
    ----------
    tensor:
        SU(2) tensor to populate with random weights. Must have intw populated.
    seed:
        Random seed for reproducibility.
    min_components:
        Minimum number of components per block (default: 3).
    max_components:
        Maximum number of components per block (exclusive, default: 6).
    """
    if tensor.intw is None:
        raise ValueError("Tensor must have intw populated (non-Abelian tensor)")
    
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    for key in tensor.data.keys():
        # Random component count for this block
        num_comp = torch.randint(min_components, max_components, (1,), generator=gen).item()
        
        # Expand block to new component count
        block = tensor.data[key]
        phys_shape = list(block.shape[:-1])
        tensor.data[key] = torch.randn(phys_shape + [num_comp], dtype=tensor.dtype, generator=gen)
        
        # Create new weights for this block
        bridge = tensor.intw[key]
        weights = torch.randn(num_comp, bridge.om_dimension, dtype=tensor.dtype, generator=gen)
        tensor.intw[key] = Bridge(bridge.cgspec, weights)


