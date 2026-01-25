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


"""Shared test utilities and helper functions for the Nicole test suite.

This module provides convenience functions for performing common assertions
on symmetry-aware tensor structures.

Functions
---------
assert_charge_neutral:
    Verify that all blocks in a tensor satisfy charge neutrality.
assert_blocks_equal:
    Compare two tensors for identical block structure and contents.
"""

from __future__ import annotations

import numpy as np

from nicole import Tensor
from nicole.blocks import BlockSchema


def assert_charge_neutral(tensor: Tensor) -> None:
    """Verify that all blocks in a tensor satisfy charge neutrality.

    Parameters
    ----------
    tensor:
        Tensor to validate for charge neutrality.
    """
    for key in tensor.data:
        totals = BlockSchema.charge_totals(tensor.indices, key)
        for group, total in totals.items():
            assert group.equal(total, group.neutral)


def assert_blocks_equal(a: Tensor, b: Tensor) -> None:
    """Verify that two tensors have identical block structure and numerical contents.

    Parameters
    ----------
    a, b:
        Tensors to compare for block equality.
    """
    assert set(a.data.keys()) == set(b.data.keys())
    for key in a.data:
        np.testing.assert_allclose(a.data[key], b.data[key])


