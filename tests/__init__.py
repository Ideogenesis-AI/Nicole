# Copyright (C) 2025 Changkai Zhang.
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


"""Test suite for the Nicole (TN) symmetry-aware tensor network library.

This package contains comprehensive unit tests and integration tests for all
components of the Nicole library, including:

  - Tensor construction and manipulation
  - Symmetry group operations (U(1), Z(2), product groups)
  - Index and sector management
  - Tensor contractions and traces
  - Decompositions (SVD, eigendecomposition)
  - Diagonal matrix operations (creation and inversion)
  - Block structure and charge conservation
  - Identity and isometry tensors
  - Arithmetic operations (addition, subtraction, direct sum)

Test Organization
-----------------
  - test_arithmetic.py: Tensor addition, subtraction, scaling
  - test_blocks.py: Block schema and charge validation
  - test_construction.py: Tensor creation and initialization
  - test_contract.py: Tensor contractions and traces
  - test_copy_access.py: Tensor copying and element access
  - test_decomp.py: SVD, eigendecomposition, and other decompositions
  - test_diag_inv.py: Diagonal matrix creation (diag) and inversion (inv)
  - test_display.py: Tensor display and formatting
  - test_group_elem.py: Elementary symmetry group operations
  - test_group_prod.py: Product group operations
  - test_identity.py: Identity and isometry tensor construction
  - test_index.py: Index operations and fusion
  - test_integration.py: End-to-end workflow tests
  - test_manipulation.py: Permutation, transposition, conjugation, merging
  - test_oplus.py: Direct sum operations
  - test_types.py: Type definitions and enumerations
  - utils.py: Shared test utilities and helpers

Running Tests
-------------
Execute all tests:
    $ pytest tests/

Run specific test file:
    $ pytest tests/test_contract.py

Run with verbose output:
    $ pytest tests/ -v

Run tests matching a pattern:
    $ pytest tests/ -k "svd"
"""
