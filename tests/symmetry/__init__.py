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


"""Tests for symmetry group implementations.

This module contains tests for all symmetry groups supported by Nicole:
Abelian groups (U1, Z2), non-Abelian SU(2), product groups, and the
yuzuha integration layer for SU(2) Clebsch-Gordan data.

Test Modules
------------
test_group_elem.py
    Tests for elementary Abelian group operations:
    - Neutral element, dual, and fusion rules for U1 and Z2
    - Charge validation and group equality
    - Cross-group distinctness and instance equality

test_group_prod.py
    Tests for ProductGroup combining multiple symmetries:
    - Creation, validation, and component access
    - Neutral element, dual, and irrep dimension in product space
    - Fusion rules for all-Abelian and mixed (Abelian × SU2) groups
    - Charge validation and group equality

test_group_su2.py
    Tests for SU2Group non-Abelian symmetry:
    - Neutral element, dual (self-dual), and irrep dimension (2j+1)
    - Multi-channel fusion rules and triangular inequality
    - Multi-particle fusion bounds and integrality consistency

test_delegate.py
    Tests for Bridge class and yuzuha integration:
    - Bridge instantiation, validation, and immutability
    - from_block constructor for SU2Group and ProductGroup
    - Device and dtype management (clone, to)
    - X-symbol and R-symbol computation correctness
    - conj(), invert_edges(), and insert_edge() structural operations
    - fs_phase() for Abelian and SU(2) groups
    - Serialization and deserialization roundtrips
    - Value-based equality comparison

Key Concepts Tested
-------------------
- Abelian (U1, Z2) and non-Abelian (SU2) symmetry implementations
- ProductGroup combining Abelian and non-Abelian components
- SU(2) Clebsch-Gordan fusion channels and triangular inequality
- Bridge data structure for intertwiner weight storage
- Charge validation and group identity/equality
"""
