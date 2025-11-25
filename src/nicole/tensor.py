# Copyright (C) 2025 Changkai Zhang.
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


from __future__ import annotations

"""Tensor container for block-symmetric data structures.

This module defines the `Tensor` dataclass, which stores symmetry-aware tensor
indices alongside a dictionary of dense NumPy blocks. Helper constructors create
zero-filled or random tensors, while arithmetic and structural operations respect
charge conservation dictated by the index metadata.
"""

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from .blocks import BlockKey, BlockSchema
from .display import tensor_summary
from .index import Index


@dataclass
class Tensor:
    """Block-sparse tensor backed by symmetry-aware indices and dense blocks.

    Each `Tensor` pairs an ordered tuple of `Index` instances with a mapping from
    block keys (one charge per axis) to dense NumPy arrays. Arithmetic operations
    are defined in a way that preserves charge conservation, and helper methods
    provide convenient constructors and transformations.

    Attributes
    ----------
    indices:
        Ordered tuple of `Index` instances defining the symmetry structure of the tensor.
    itags:
        Ordered tuple of human-readable labels for each index.
    data:
        Mapping from block keys (one charge per axis) to dense NumPy arrays.
    dtype:
        Data type for the dense blocks. Defaults to double precision real values.
    label:
        Human-readable label for the tensor. Defaults to "Tensor".

    Methods
    -------
    zeros()
        Create a symmetry-aware tensor with admissible zero-filled blocks.
    random()
        Create a tensor filled with random values for each admissible block.
    norm()
        Compute the Frobenius norm aggregated across all dense blocks.
    conj()
        In-place: Complex conjugate every dense block, and revert all index directions.
    permute()
        In-place: Permute tensor axes according to the provided reordering.
    transpose()
        In-place: Transpose tensor axes; defaults to reversing the index order.
    retag()
        Retag indices: update specific tags by name/index, or replace all tags.
    
    Notes
    -----
    For functional (non-mutating) versions of conj, permute, and transpose that return
    new tensor instances, use the standalone functions from `nicole.operations`.
    """

    indices: Tuple[Index, ...]
    itags: Tuple[str, ...]
    data: MutableMapping[BlockKey, np.ndarray]
    dtype: np.dtype = np.float64
    label: str = "Tensor"

    def __post_init__(self) -> None:
        """Validate the provided block dictionary against the index schema."""
        if len(self.itags) != len(self.indices):
            raise ValueError(
                f"Number of itags ({len(self.itags)}) must match number of indices ({len(self.indices)})"
            )
        BlockSchema.validate_blocks(self.indices, self.data)
        for key in self.data:
            if not BlockSchema.charges_conserved(self.indices, key):
                raise ValueError(
                    f"Block {key} violates charge conservation for assigned index directions"
                )

    @classmethod
    def zeros(cls, indices: Sequence[Index], dtype=np.float64, itags: Optional[Sequence[str]] = None) -> Tensor:
        """Create a symmetry-aware tensor with admissible zero-filled blocks."""
        # Normalise input to an immutable tuple for downstream utilities.
        indices_tuple = tuple(indices)
        if itags is None:
            itags_tuple = tuple(f"_init_" for _ in indices_tuple)
        else:
            itags_tuple = tuple(itags)
        data: Dict[BlockKey, np.ndarray] = {}
        # Iterate over all admissible charge assignments for the provided indices.
        for key in BlockSchema.iter_admissible_keys(indices_tuple):
            if not BlockSchema.charges_conserved(indices_tuple, key):
                continue
            # Determine the dense shape implied by the current key and allocate zeros.
            shape = BlockSchema.shape_for_key(indices_tuple, key)
            data[key] = np.zeros(shape, dtype=dtype)
        return cls(indices=indices_tuple, itags=itags_tuple, data=data, dtype=dtype)

    @classmethod
    def random(
        cls, indices: Sequence[Index], dtype=np.float64, seed: Optional[int] = None, itags: Optional[Sequence[str]] = None
    ) -> Tensor:
        """Create a tensor filled with random values for each admissible block."""
        # Initialise the random number generator.
        rng = np.random.default_rng(seed)
        indices_tuple = tuple(indices)
        if itags is None:
            itags_tuple = tuple(f"_init_" for _ in indices_tuple)
        else:
            itags_tuple = tuple(itags)
        data: Dict[BlockKey, np.ndarray] = {}
        target_dtype = np.dtype(dtype)
        # Walk through admissible blocks in the same fashion as `zeros`.
        for key in BlockSchema.iter_admissible_keys(indices_tuple):
            if not BlockSchema.charges_conserved(indices_tuple, key):
                continue
            shape = BlockSchema.shape_for_key(indices_tuple, key)
            if np.issubdtype(target_dtype, np.complexfloating):
                real = rng.standard_normal(shape)
                imag = rng.standard_normal(shape)
                arr = real + 1j * imag
            else:
                arr = rng.standard_normal(shape)
            data[key] = arr.astype(target_dtype, copy=False)
        return cls(indices=indices_tuple, itags=itags_tuple, data=data, dtype=target_dtype)

    def __str__(self) -> str:
        """Return a formatted multiline summary generated by `tensor_summary`."""
        return tensor_summary(self.indices, self.itags, self.data, self.dtype, self.label, self.norm())

    __repr__ = __str__

    def norm(self) -> float:
        """Compute the Frobenius norm aggregated across all dense blocks."""
        if not self.data:
            return 0.0
        return float(
            np.sqrt(sum(np.sum(np.abs(block) ** 2) for block in self.data.values()))
        )

    def _align_for_binary(self, other: "Tensor") -> Tuple["Tensor", "Tensor"]:
        """Ensure two tensors are compatible for element-wise binary operations."""
        if len(self.indices) != len(other.indices):
            raise ValueError("Cannot add/sub tensors with different order")
        if any((a.group != b.group) or (a.direction != b.direction) for a, b in zip(self.indices, other.indices)):
            raise ValueError("Indices groups and directions must match")
        if any(a.sector_dim_map() != b.sector_dim_map() for a, b in zip(self.indices, other.indices)):
            raise ValueError("Index sector structures must match")
        return self, other

    def __add__(self, other: Tensor) -> Tensor:
        """Element-wise addition while preserving symmetry metadata."""
        self._align_for_binary(other)
        keys = set(self.data.keys()) | set(other.data.keys())
        new_data: Dict[BlockKey, np.ndarray] = {}
        for k in keys:
            a = self.data.get(k)
            b = other.data.get(k)
            if a is None:
                new_data[k] = (+b)
            elif b is None:
                new_data[k] = (+a)
            else:
                new_data[k] = a + b
        return Tensor(
            indices=self.indices,
            itags=self.itags,
            data=new_data,
            dtype=np.result_type(self.dtype, other.dtype),
            label=self.label,
        )

    def __sub__(self, other: Tensor) -> Tensor:
        """Element-wise subtraction while preserving symmetry metadata."""
        self._align_for_binary(other)
        keys = set(self.data.keys()) | set(other.data.keys())
        new_data: Dict[BlockKey, np.ndarray] = {}
        for k in keys:
            a = self.data.get(k)
            b = other.data.get(k)
            if a is None:
                new_data[k] = -b
            elif b is None:
                new_data[k] = +a
            else:
                new_data[k] = a - b
        return Tensor(
            indices=self.indices,
            itags=self.itags,
            data=new_data,
            dtype=np.result_type(self.dtype, other.dtype),
            label=self.label,
        )

    def __mul__(self, scalar: Union[int, float, complex]) -> Tensor:
        """Scale every dense block by a scalar."""
        new_data = {k: (v * scalar) for k, v in self.data.items()}
        return Tensor(
            indices=self.indices,
            itags=self.itags,
            data=new_data,
            dtype=np.result_type(self.dtype, type(scalar)),
            label=self.label,
        )

    __rmul__ = __mul__

    def conj(self) -> None:
        """Complex conjugate every dense block if dtype is complex, and revert all index directions."""
        # Only conjugate data if dtype is complex
        if np.issubdtype(self.dtype, np.complexfloating):
            for k in self.data:
                self.data[k] = np.conjugate(self.data[k])
        # Flip all index directions
        self.indices = tuple(idx.flip() for idx in self.indices)

    def permute(self, order: Sequence[int]) -> None:
        """Permute tensor axes according to the provided reordering."""
        if sorted(order) != list(range(len(self.indices))):
            raise ValueError("Invalid permutation order")
        
        # Update indices and itags
        self.indices = tuple(self.indices[i] for i in order)
        self.itags = tuple(self.itags[i] for i in order)
        
        # Update data blocks
        new_data = {}
        for key, arr in self.data.items():
            new_key = tuple(key[i] for i in order)
            new_data[new_key] = np.transpose(arr, axes=order)
        self.data = new_data

    def transpose(self, *order: int) -> None:
        """Transpose tensor axes; defaults to reversing the index order."""
        if not order:
            order = tuple(reversed(range(len(self.indices))))
        self.permute(order)

    def retag(self, mapping_or_indices: Union[Mapping[str, str], Sequence[str], Sequence[int]], 
              new_tags: Optional[Sequence[str]] = None) -> None:
        """Retag indices using one of three modes.
        
        Parameters
        ----------
        mapping_or_indices:
            Can be one of:
            - Mapping[str, str]: Dictionary mapping old tags to new tags
            - Sequence[str]: Complete list of new tags (must match number of indices)
            - Sequence[int]: List of index positions to update (requires new_tags)
        new_tags:
            New tags to use when mapping_or_indices is a sequence of integers.
            Must have the same length as mapping_or_indices.
        
        Examples
        --------
        # Mode 1: Mapping (update specific tags by name)
        tensor.retag({"a": "left", "b": "right"})
        
        # Mode 2: Full replacement (replace all tags)
        tensor.retag(["left", "middle", "right"])
        
        # Mode 3: Selective update by position
        tensor.retag([0, 2], ["left", "right"])
        """
        if isinstance(mapping_or_indices, Mapping):
            # Mode 1: Mapping dictionary
            self.itags = tuple(mapping_or_indices.get(tag, tag) for tag in self.itags)
        elif new_tags is not None:
            # Mode 3: Update specific indices
            if not isinstance(mapping_or_indices, Sequence):
                raise TypeError("When new_tags is provided, first argument must be a sequence of integers")
            indices = list(mapping_or_indices)
            if len(indices) != len(new_tags):
                raise ValueError("Number of indices must match number of new tags")
            if not all(isinstance(i, int) for i in indices):
                raise TypeError("Index positions must be integers")
            if any(i < 0 or i >= len(self.itags) for i in indices):
                raise IndexError("Index position out of range")
            
            # Convert to list for mutation, then back to tuple
            new_itags = list(self.itags)
            for idx, tag in zip(indices, new_tags):
                new_itags[idx] = tag
            self.itags = tuple(new_itags)
        else:
            # Mode 2: Full replacement
            if not isinstance(mapping_or_indices, Sequence):
                raise TypeError("Expected a sequence of strings for full replacement")
            if len(mapping_or_indices) != len(self.itags):
                raise ValueError(f"Number of new tags ({len(mapping_or_indices)}) must match number of indices ({len(self.itags)})")
            self.itags = tuple(mapping_or_indices)
