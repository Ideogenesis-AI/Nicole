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


def assert_blocks_equal(a: Tensor, b: Tensor) -> None:
    """Verify that two tensors have identical block structure and numerical contents.
    
    Uses torch.allclose() with default PyTorch tolerances (rtol=1e-05, atol=1e-08)
    for numerical comparison of block values.

    Parameters
    ----------
    a, b:
        Tensors to compare for block equality.
    """
    assert set(a.data.keys()) == set(b.data.keys())
    for key in a.data:
        assert torch.allclose(a.data[key], b.data[key])


