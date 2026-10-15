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


"""Shared typing utilities for Nicole symmetry-aware tensor machinery."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING
from typing import Dict, Hashable, Tuple, Union

import torch

if TYPE_CHECKING:
    # Only for annotations: `tensor` imports from here, so the dependency has
    # to stay one-way at runtime.
    from .tensor import Tensor

# Charge can be any hashable value. For single symmetry groups (U1Group, Z2Group,
# SU2Group), charges are typically integers. For ProductGroup (multiple symmetries),
# charges are tuples of hashable values, one per component group.
#
# Examples:
#   - U1Group charge: 2, -1, 0 (integers)
#   - Z2Group charge: 0, 1 (integers 0 or 1)
#   - SU2Group charge: 0, 1, 2, 3, ... (integers representing 2j values)
#   - ProductGroup(U1, U1) charge: (2, -1), (0, 0) (tuples of integers)
#   - ProductGroup(U1, Z2) charge: (3, 1), (-2, 0) (tuples: int, 0/1)
#   - ProductGroup(U1, SU2) charge: (2, 4), (-1, 0) (tuples: U1 int, 2j int)
Charge = Hashable


class Direction(IntEnum):
    """Orientation flag used by tensor indices."""

    IN = +1
    OUT = -1

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


# Real/complex dtype pairings, used to decide the dtype of an operation
# without asking torch. `torch.result_type` answers the same question but
# costs around 400 ns per call against roughly 20 ns for a dict lookup, which
# is a measurable fraction of an elementary tensor operation.
_COMPLEX_OF: Dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.complex32,
    torch.bfloat16: torch.complex64,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
    torch.complex32: torch.complex32,
    torch.complex64: torch.complex64,
    torch.complex128: torch.complex128,
}

_REAL_OF: Dict[torch.dtype, torch.dtype] = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def complex_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """Return the complex dtype that pairs with `dtype`.

    A complex dtype is returned unchanged. Integer and boolean dtypes have no
    natural complex counterpart of the same width and map to `complex128`,
    matching how torch promotes them against a Python complex scalar.

    Parameters
    ----
    dtype:
        Real or complex dtype to promote.

    Returns
    ----
    torch.dtype
        Complex dtype of the corresponding precision.
    """
    return _COMPLEX_OF.get(dtype, torch.complex128)


def real_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """Return the real dtype that pairs with `dtype`.

    A real dtype is returned unchanged, so this is safe to apply to any dtype
    when only the real component's precision matters.

    Parameters
    ----
    dtype:
        Real or complex dtype to demote.

    Returns
    ----
    torch.dtype
        Real dtype of the corresponding precision.
    """
    return _REAL_OF.get(dtype, dtype)


def promote_tensors(*tensors: Tensor) -> Tuple[Tensor, ...]:
    """Cast tensors to their common dtype, leaving them be when it is shared.

    Tensors that already agree are returned as given, so a real-valued
    operation neither casts nor allocates.

    Parameters
    ----
    tensors:
        One or more tensors to bring to a common dtype.

    Returns
    ----
    Tuple
        The tensors, each cast only if its own dtype differs from the common one.
    """
    first = tensors[0].dtype
    # Torch dtypes are singletons, so `is` settles the matching case without
    # asking torch for a promotion it does not need.
    for other in tensors:
        if other.dtype is not first:
            dtype = reduce(torch.promote_types, (t.dtype for t in tensors))
            return tuple(t if t.dtype is dtype else t.to(dtype=dtype) for t in tensors)
    return tensors


def scalar_result_dtype(dtype: torch.dtype, scalar: Union[int, float, complex]) -> torch.dtype:
    """Return the dtype a tensor of `dtype` takes on after meeting `scalar`.

    Only a complex scalar can widen a floating-point dtype, and it widens to
    the complex dtype of matching precision, so a `float32` tensor scaled by a
    complex number becomes `complex64` instead of `complex128`.

    Parameters
    ----
    dtype:
        Current dtype of the tensor.
    scalar:
        Python scalar the tensor is combined with.

    Returns
    ----
    torch.dtype
        Dtype the operation should produce.
    """
    if isinstance(scalar, complex):
        return _COMPLEX_OF.get(dtype, torch.complex128)
    if dtype.is_floating_point or dtype.is_complex:
        return dtype
    # An integer tensor widens to the default real dtype under a float scalar,
    # following torch.
    return torch.get_default_dtype() if isinstance(scalar, float) else dtype


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
    # Callers have usually resolved the device already, and re-wrapping one
    # costs more than the rest of this function put together.
    if not isinstance(device, torch.device):
        device = torch.device(device)
    
    if device.type == 'mps':
        # MPS doesn't support float64 or complex128
        if dtype == torch.float64:
            return torch.float32
        elif dtype == torch.complex128:
            return torch.complex64
    
    return dtype
