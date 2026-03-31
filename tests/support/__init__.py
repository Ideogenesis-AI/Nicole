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


"""Tests for utilities and secondary features.

This module contains tests for supporting functionality that enhances
usability and developer experience, including autograd control, cloning
behavior, device management, display formatting, and type definitions.

Test Modules
------------
test_autograd.py
    Tests for autograd control and gradient computation:
    - Autograd disabled globally by default
    - torch.enable_grad() context manager integration
    - requires_grad property getter and setter
    - Gradient computation and backward() through scalar operations
    - Autograd interaction with device management

test_helpers.py
    Tests for tensor helper operations:
    - clone(): deep copy semantics, data independence, intw cloning
    - sorted_keys, key(i), block(i): deterministic block access
    - filter_blocks(): block subsetting, metadata and intw preservation
    - regularize(): Bridge weight normalization for SU(2) tensors

test_device.py
    Tests for device management:
    - Default device placement and override
    - Device transfer (cpu(), cuda(), to())
    - CUDA availability checks and cross-device behavior

test_display.py
    Tests for tensor display and formatting:
    - Internal helper functions (_format_bytes, _format_count_list,
      _charge_components, _group_signature, _format_single_value)
    - tensor_summary output: norm, dtype, block structure, label

test_types.py
    Tests for core type definitions:
    - Direction enum (IN/OUT values and reverse())
    - Sector construction, immutability, equality, and hashability

Key Features Tested
-------------------
- Autograd control and gradient flow through tensor operations
- Clone semantics and data independence
- Device management and CPU/GPU compatibility
- Tensor display and human-readable formatting
- Core type correctness (Direction, Sector)

Developer Experience
--------------------
These tests ensure that Nicole provides:
- Seamless PyTorch integration with autograd support
- Flexible device management for CPU/GPU workflows
- Intuitive cloning behavior without surprises
- Clear visualization of tensor structure
- Strong typing for better IDE support
- Helpful error messages and debugging tools
"""
