# Copyright (C) 2025-2026 Changkai Zhang.
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


"""Tests for utilities and secondary features.

This module contains tests for supporting functionality that enhances
usability and developer experience, including copying behavior, display
formatting, and type definitions. While not central to tensor algebra,
these features are important for practical use of the library.

Test Modules
------------
test_autograd.py
    Tests for automatic differentiation:
    - Gradient computation through tensor operations
    - Backpropagation compatibility
    - Integration with PyTorch autograd
    - Gradient flow through contractions
    - Differentiable decompositions
    - Gradient checking and validation

test_copy_access.py
    Tests for tensor copying and element access:
    - Deep copy vs. shallow copy behavior
    - Copy semantics for tensors and indices
    - Element-wise access to tensor blocks
    - Data sharing and mutation safety
    - Memory management and ownership
    - Indexing into block structure

test_device.py
    Tests for device management:
    - CPU and GPU device placement
    - Device transfer operations
    - Mixed device computations
    - CUDA availability and compatibility
    - Memory management across devices
    - Device-specific optimizations

test_display.py
    Tests for tensor display and formatting:
    - Pretty-printing tensor information
    - Displaying index structure and charges
    - Block structure visualization
    - Summary statistics (shape, charge, blocks)
    - Formatting for different output contexts
    - Readable representation of symmetry info
    - Debug output and diagnostics

test_types.py
    Tests for type definitions and enumerations:
    - Direction enum (IN, OUT)
    - Type annotations and validation
    - Custom type definitions
    - Type checking and inference
    - Compatibility with type checkers
    - Runtime type validation

Key Features Tested
-------------------
- Automatic differentiation and gradient flow
- Copy semantics and data safety
- Device management and GPU acceleration
- User-friendly tensor inspection
- Clear error messages and diagnostics
- Type safety and validation
- API consistency and ergonomics

Developer Experience
--------------------
These tests ensure that Nicole provides:
- Seamless PyTorch integration with autograd support
- Flexible device management for CPU/GPU workflows
- Intuitive copying behavior without surprises
- Clear visualization of tensor structure
- Strong typing for better IDE support
- Helpful error messages and debugging tools
"""
