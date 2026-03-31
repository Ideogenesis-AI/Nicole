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


"""Tests for tensor operations and transformations.

This module contains tests for all operations that can be performed on
tensors, including arithmetic, contractions, decompositions, and
structural manipulations. These operations form the core computational
functionality of Nicole for tensor network algorithms.

Test Modules
------------
test_arithmetic.py
    Tests for tensor arithmetic:
    - Addition and subtraction, including non-overlapping and partially
      overlapping sector structures
    - Scalar multiplication with integer, float, and complex scalars
    - Norm computation and scaling properties
    - SU(2) addition and subtraction with same or different weights,
      including collinear weight detection and compression

test_capcup.py
    Tests for the capcup maneuver (bond direction inversion):
    - Direction updates on both tensors
    - Frobenius-Schur phase applied to B for SU(2) groups
    - Absence of phase for Abelian groups
    - Contraction invariance for U1, SU(2), and ProductGroup (U1×U1, Z2×SU2, U1×SU2)

test_contract.py
    Tests for tensor contraction and trace:
    - Pairwise and multi-index contractions
    - Partial and full traces
    - Charge conservation and index structure
    - Abelian and non-Abelian (SU2, ProductGroup) variants

test_decomp.py
    Tests for the high-level decomp() interface:
    - All decomposition modes (SVD, UR, LV, QR)
    - Multi-axis decomposition and bond index flow control
    - Truncation by singular value threshold and bond dimension
    - SU(2) and ProductGroup compatibility

test_factorize.py
    Tests for low-level factorization routines:
    - svd(): singular value decomposition, isometry checks, truncation
    - qr(): QR factorization and orthogonality
    - eig(): eigenvalue decomposition, sorting, and Hermitian variant
    - SU(2) and ProductGroup compatibility

test_diag_inv.py
    Tests for diagonal tensor operations:
    - diag(): creating diagonal tensors from reduced blocks
    - inv(): inverting diagonal tensors block-wise
    - Charge structure preservation and error handling

test_maneuver.py
    Tests for structural tensor maneuvers:
    - conj(), permute(), transpose()
    - retag(), invert(), insert_index()
    - merge_axes(), trim_zero_blocks()

test_oplus.py
    Tests for direct sum operations:
    - oplus(): direct sum expanding bond dimensions and charge spaces
    - Compatibility with contractions and other operations

Key Operations Tested
---------------------
- Charge conservation throughout all operations
- Numerical accuracy and stability
- Abelian (U1, Z2) and non-Abelian (SU2) symmetry compatibility
- ProductGroup combining Abelian and non-Abelian components
- Edge cases and error handling
- High-order (5th/6th order) stress tests across contraction
  decomposition, and maneuver operations
"""
