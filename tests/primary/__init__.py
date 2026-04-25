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


"""Primary tests for core data structures and tensor construction.

This module contains tests for the fundamental building blocks of Nicole:
the Index and Tensor classes, along with their construction methods.
These are the foundational concepts that all other operations depend on.

Test Modules
------------
test_index.py
    Tests for Index class operations:
    - Index creation with sectors and directions
    - Index fusion, splitting, and union
    - Index validation and error handling
    - Direction flipping and charge conjugation

test_space.py
    Tests for physical Hilbert space and operator construction:
    - Spin systems with U(1) and SU(2) symmetry
    - Spinless fermion systems with U(1) and Z2 symmetry
    - Spinful fermion (Band) systems with product symmetries
    - Correctness of operator matrix elements and charge assignments

test_blocks.py
    Tests for block structure and charge conservation:
    - BlockSchema creation and validation
    - Charge neutrality enforcement and block shape computation
    - Block indexing and iteration
    - Intertwiner collinearity and block addition for non-Abelian groups
    - Block compression for collinear intertwiners

test_construction.py
    Tests for Tensor construction methods:
    - Creating zero-filled, random, and scalar tensors
    - In-place random fill and sector normalization
    - Validation of charge conservation, index compatibility, and group consistency
    - Sector pruning and ProductGroup integration

test_identity.py
    Tests for identity and isometry tensor construction:
    - identity(): 2nd-order identity tensors
    - isometry(): 2-to-1 fusion isometries
    - isometry_n(): N-to-1 general isometries
    - Orthonormality, charge neutrality, and norm correctness
    - Abelian and non-Abelian (SU2, ProductGroup) variants

Key Concepts Tested
-------------------
- Charge conservation and fusion rules
- Abelian (U1, Z2) and non-Abelian (SU2) symmetry implementations
- ProductGroup combining Abelian and non-Abelian components
- Block-sparse tensor structure and intertwiner objects
- Index orientation (IN/OUT directions) and duality
- Tensor construction and initialization
"""
