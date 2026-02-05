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


"""Tests for tensor operations and transformations.

This module contains tests for all operations that can be performed on
tensors, including arithmetic, contractions, decompositions, and
manipulations. These operations form the core computational functionality
of Nicole for tensor network algorithms.

Test Modules
------------
test_arithmetic.py
    Tests for basic arithmetic operations:
    - Addition and subtraction of tensors
    - Scalar multiplication and division
    - Negation and sign operations
    - Broadcasting and shape compatibility
    - Charge-preserving arithmetic

test_contract.py
    Tests for tensor contraction operations:
    - contract(): General tensor contractions
    - trace(): Partial and full traces
    - Multi-index contractions
    - Charge conservation during contraction
    - Einstein summation notation support

test_decomp.py
    Tests for tensor decomposition operations:
    - decomp(): High-level decomposition interface
    - svd(): Singular value decomposition
    - eig(): Eigenvalue decomposition
    - Truncation and bond dimension control
    - Charge-preserving decompositions

test_diag_inv.py
    Tests for diagonal matrix operations:
    - diag(): Creating diagonal tensors from vectors
    - inv(): Matrix inversion for diagonal tensors
    - Iterative diagonal construction
    - Charge structure preservation
    - Numerical stability and error handling

test_manipulation.py
    Tests for tensor manipulation and reshaping:
    - permute(): Axis permutation
    - transpose(): Simple two-axis swap
    - conj(): Complex conjugation with direction flip
    - merge_axes(): Combining multiple axes
    - retag(): Changing index tags
    - subsector(): Extracting charge subspaces

test_oplus.py
    Tests for direct sum operations:
    - oplus(): Direct sum of tensors
    - Increasing bond dimensions
    - Charge space expansion
    - Compatibility with other operations
    - Multi-tensor direct sums

Key Operations Tested
---------------------
- Charge conservation throughout all operations
- Numerical accuracy and stability
- Edge cases and error handling
- Performance with large bond dimensions
- Compatibility with different symmetry groups
"""
