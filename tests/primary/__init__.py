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


"""Primary tests for core data structures and tensor construction.

This module contains tests for the fundamental building blocks of Nicole:
the Index, Space, and Tensor classes, along with their construction methods.
These are the foundational concepts that all other operations depend on.

Test Modules
------------
test_index.py
    Tests for Index class operations:
    - Index creation with sectors and directions
    - Index fusion and charge conservation
    - Index validation and error handling
    - Direction flipping and conjugation

test_space.py
    Tests for Space (physical Hilbert space) operations:
    - Space creation from sectors
    - Space tensor product and direct sum
    - Charge structure validation
    - Space equality and compatibility checks

test_blocks.py
    Tests for block structure and charge conservation:
    - BlockSchema creation and validation
    - Charge neutrality enforcement
    - Block indexing and iteration
    - Allowed block structure based on fusion rules

test_construction.py
    Tests for Tensor construction methods:
    - zeros(): Creating zero-initialized tensors
    - random(): Creating random tensors
    - Tensor initialization from arrays
    - Block-wise tensor construction
    - Custom symmetry tensor creation

test_identity.py
    Tests for identity and isometry tensor construction:
    - identity(): 2-leg identity tensors
    - isometry(): 2-to-1 fusion isometries
    - isometry_n(): N-to-1 general isometries
    - Symmetry-preserving properties
    - Use in index merging operations

Key Concepts Tested
-------------------
- Charge conservation and fusion rules
- Abelian symmetry implementation (U1, Z2, ProductGroup)
- Block-sparse tensor structure
- Index orientation (IN/OUT directions)
- Tensor construction and initialization
"""
