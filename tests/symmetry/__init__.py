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


"""Tests for symmetry group operations.

This module contains tests for Abelian symmetry groups and their operations,
including single symmetries (U1, Z2) and product groups combining multiple
symmetries. These tests ensure correct implementation of group theory and
charge conservation rules.

Test Modules
------------
test_group_elem.py
    Tests for elementary symmetry groups:
    - U1Group: Continuous U(1) symmetry (integer charges)
      * Charge arithmetic and fusion rules
      * Identity and inverse elements
      * Group multiplication and composition
      
    - Z2Group: Binary Z(2) symmetry (0/1 charges)
      * Modular arithmetic (mod 2)
      * Parity conservation
      * Group operations
      
    - General Abelian group interface:
      * Identity element properties
      * Inverse element properties
      * Associativity and commutativity
      * Charge validation

test_group_prod.py
    Tests for product group operations:
    - ProductGroup: Combining multiple symmetries
      * U1 × U1: Multiple particle number conservation
      * U1 × Z2: Charge and parity conservation
      * Z2 × Z2: Multiple parity sectors
      
    - Product group mechanics:
      * Charge tuple operations
      * Component-wise fusion rules
      * Identity and inverse in product space
      * Tensor operations with product groups
      
    - Advanced features:
      * Mixing different group types
      * Charge validation across components
      * Performance with multiple symmetries

Key Concepts Tested
-------------------
- Abelian group axioms (associativity, identity, inverses)
- Charge fusion rules and conservation laws
- Group homomorphisms and representations
- Tensor network compatibility with symmetries
- Efficient charge arithmetic implementation

Symmetry Applications
---------------------
These symmetries are crucial for:
- Quantum number conservation (particle number, spin, etc.)
- Reducing computational complexity via block structure
- Enforcing physical conservation laws
- Exploiting symmetries in quantum many-body systems
"""
