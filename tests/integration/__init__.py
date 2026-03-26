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


"""Integration tests for multi-component interactions in Nicole.

This module contains integration-level tests that verify correct behavior
when multiple tensor operations interact. These tests ensure that operations
compose correctly and maintain mathematical consistency properties across
the library.

Test Modules
------------
test_consistency.py
    Tests for algebraic consistency properties:
    - Identity contraction preservation (A ⊗ I = A)
    - Distributivity of contraction over addition
    - Scalar multiplication linearity
    - Permutation commutativity with contraction
    - Bilinearity of inner products

test_unitarity.py
    Tests for unitarity and orthogonality properties:
    - Isometry fusion-unfusion roundtrips (V† V A ≈ A)
    - Isometry linearity (V(A + B) = VA + VB)
    - Isometry scalar multiplication commutation
    - Conjugate self-contraction (<A|A> = ||A||²)
    - Phase correctness in SU(2) contractions

test_integration.py
    Tests for complex multi-operation workflows:
    - Composition of multiple contractions
    - Decomposition followed by reconstruction
    - Index manipulation with contractions
    - End-to-end tensor network algorithms
    - Cross-operation compatibility

Key Properties Tested
---------------------
- Gauge invariance in SU(2) tensor contractions
- Physical tensor equality (R@W) vs reduced space equality
- Charge conservation across operation sequences
- Numerical stability in operation chains
- Correctness of X-symbols and recoupling coefficients
"""
