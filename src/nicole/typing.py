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


from __future__ import annotations

"""Shared typing utilities for Nicole (TN) symmetry-aware tensor machinery."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Hashable, Union

import torch

# Charge can be any hashable value. For single symmetry groups (U1Group, Z2Group),
# charges are typically integers. For ProductGroup (multiple symmetries), charges
# are tuples of hashable values, one per component group.
#
# Examples:
#   - U1Group charge: 2, -1, 0 (integers)
#   - Z2Group charge: 0, 1 (integers 0 or 1)
#   - ProductGroup(U1, U1) charge: (2, -1), (0, 0) (tuples of integers)
#   - ProductGroup(U1, Z2) charge: (3, 1), (-2, 0) (tuples: int, 0/1)
Charge = Hashable


class Direction(IntEnum):
    """Orientation flag used by tensor indices."""

    IN = -1
    OUT = 1

    def reverse(self) -> Direction:
        """Return the opposite orientation (IN ↔ OUT)."""
        return Direction(-int(self))


@dataclass(frozen=True)
class Sector:
    """Charge sector descriptor pairing a conserved charge with its dimension.

    Attributes
    ----------
    charge:
        The conserved quantity carried by the sector. May be any hashable value.
    dim:
        Positive integer indicating how many states/multiplets the sector spans.
    """

    charge: Charge
    dim: int

    def __post_init__(self) -> None:
        """Validate that the sector has a strictly positive dimension."""
        if self.dim <= 0:
            raise ValueError("Sector dimension must be positive")


def normalize_dtype_for_device(dtype: torch.dtype, device: Union[str, torch.device]) -> torch.dtype:
    """Normalize dtype to be compatible with the target device.
    
    Some devices have limited dtype support:
    - MPS (Apple Silicon) doesn't support float64/complex128
    
    Parameters
    ----------
    dtype : torch.dtype
        Requested data type
    device : str or torch.device
        Target device
        
    Returns
    -------
    torch.dtype
        Compatible dtype for the device (may be downgraded if needed)
        
    Notes
    -----
    This helper automatically converts:
    - float64 -> float32 on MPS
    - complex128 -> complex64 on MPS
    """
    device = torch.device(device)
    
    if device.type == 'mps':
        # MPS doesn't support float64 or complex128
        if dtype == torch.float64:
            return torch.float32
        elif dtype == torch.complex128:
            return torch.complex64
    
    return dtype
