# Copyright (C) 2025 Changkai Zhang.
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


"""Test suite for the Nicole symmetry-aware tensor network library.

This package contains comprehensive unit tests and integration tests for all
components of the Nicole library, organized into thematic subdirectories.

Test Organization
-----------------
The tests are organized into four main categories:

primary/
    Core data structures and tensor construction:
    - test_index.py: Index operations, fusion, and validation
    - test_space.py: Space operations and charge structure
    - test_blocks.py: Block schema and charge conservation
    - test_construction.py: Tensor creation and initialization
    - test_identity.py: Identity and isometry tensor construction

operations/
    Tensor operations and transformations:
    - test_arithmetic.py: Addition, subtraction, multiplication, division
    - test_contract.py: Tensor contractions and traces
    - test_decomp.py: SVD, eigendecomposition, and other decompositions
    - test_diag_inv.py: Diagonal matrix creation (diag) and inversion (inv)
    - test_manipulation.py: Permutation, transposition, conjugation, merging
    - test_oplus.py: Direct sum operations

symmetry/
    Symmetry group operations:
    - test_group_elem.py: Elementary symmetry group operations (U1, Z2)
    - test_group_prod.py: Product group operations and multi-symmetry

support/
    Utilities and secondary features:
    - test_helpers.py: Tensor cloning and element access
    - test_display.py: Tensor display and formatting
    - test_types.py: Type definitions and enumerations

Root level:
    - test_integration.py: End-to-end workflow tests
    - utils.py: Shared test utilities and helpers

Running Tests
-------------
Execute all tests:
    $ pytest tests/

Run tests in a specific category:
    $ pytest tests/primary/
    $ pytest tests/operations/
    $ pytest tests/symmetry/
    $ pytest tests/support/

Run specific test file:
    $ pytest tests/operations/test_contract.py

Run with verbose output:
    $ pytest tests/ -v

Run tests matching a pattern:
    $ pytest tests/ -k "svd"

Run a specific test function:
    $ pytest tests/operations/test_decomp.py::test_svd_basic
"""
