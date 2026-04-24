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


"""Test suite for the Nicole symmetry-aware tensor network library.

This package contains comprehensive unit tests and integration tests for all
components of the Nicole library, organized into thematic subdirectories.

Test Organization
-----------------
The tests are organized into five main categories:

primary/
    Core data structures and tensor construction:
    - test_index.py: Index operations, fusion, and validation
    - test_space.py: Space operations and charge structure
    - test_blocks.py: Block schema and charge conservation
    - test_construction.py: Tensor creation and initialization
    - test_identity.py: Identity and isometry tensor construction

operations/
    Tensor operations and transformations:
    - test_arithmetic.py: Addition, subtraction, scalar multiplication
    - test_contract.py: Tensor contractions and traces
    - test_decomp.py: SVD, eigendecomposition, and other decompositions
    - test_factorize.py: SVD- and QR-based tensor factorization
    - test_diag_inv.py: Diagonal matrix creation (diag) and inversion (inv)
    - test_einsum.py: einsum-based contractions, permutations, and traces
    - test_maneuver.py: Permutation, transposition, conjugation, merging
    - test_oplus.py: Direct sum operations
    - test_capcup.py: Bond direction inversion (capcup)

symmetry/
    Symmetry group operations:
    - test_group_elem.py: Elementary symmetry group operations (U1, Z2)
    - test_group_prod.py: Product group operations and multi-symmetry
    - test_group_su2.py: SU(2) group operations and recoupling
    - test_delegate.py: Bridge and CG tensor delegation

support/
    Utilities and secondary features:
    - test_helpers.py: Tensor cloning, element access, filter_blocks, regularize
    - test_display.py: Tensor display and formatting
    - test_types.py: Type definitions and enumerations
    - test_autograd.py: Automatic differentiation support
    - test_device.py: Multi-device tensor operations

integration/
    Multi-component interaction tests:
    - test_consistency.py: Algebraic consistency (distributivity, linearity)
    - test_unitarity.py: Unitarity properties (isometries, conjugation)
    - test_integration.py: End-to-end workflow tests
    - test_spectrum.py: Physically correct spectra for known models
    - test_propagation.py: Device argument propagation (accelerator required)

Root level:
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
    $ pytest tests/integration/

Run specific test file:
    $ pytest tests/operations/test_contract.py

Run with verbose output:
    $ pytest tests/ -v

Run tests matching a pattern:
    $ pytest tests/ -k "svd"

Run a specific test function:
    $ pytest tests/operations/test_decomp.py::test_svd_basic
"""
