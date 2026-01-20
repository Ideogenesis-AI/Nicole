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

from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from .blocks import BlockKey, BlockSchema
from .display import tensor_summary
from .index import Index
from .typing import Direction, Sector
from .symmetry.base import SymmetryGroup


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
    copy()
        Create a deep copy of this tensor with independent block data.
    rand_fill()
        In-place: Fill all data blocks with random values.
    insert_index()
        In-place: Insert a trivial index (neutral charge, dimension 1) at a position.
    group
        Property returning the symmetry group of this tensor.
    sorted_keys
        Property returning block keys in display order (cached).
    key()
        Get the BlockKey for the i-th block (1-indexed, matching display).
    block()
        Access the i-th block by integer index (1-indexed, matching display).
    show()
        Display selected blocks without max_line limits.
    conj()
        In-place: Complex conjugate every dense block, and revert all index directions.
    permute()
        In-place: Permute tensor axes according to the provided reordering.
    transpose()
        In-place: Transpose tensor axes; defaults to reversing the index order.
    flip()
        In-place: Flip the direction of specified index/indices (uses dual to maintain charge conservation).
    retag()
        Retag indices: update specific tags by name/index, or replace all tags.
    
    Notes
    -----
    For functional (non-mutating) versions of conj, permute, and transpose that return
    new tensor instances, use the standalone functions from `nicole.operators`.
    """

    indices: Tuple[Index, ...]
    itags: Tuple[str, ...]
    data: MutableMapping[BlockKey, np.ndarray]
    dtype: np.dtype = np.float64
    label: str = "Tensor"
    _sorted_keys: Optional[Tuple[BlockKey, ...]] = field(default=None, repr=False, compare=False)

    # ------------------------------------------------------------
    #   Post-initialization validation
    # ------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate the provided block dictionary against the index schema."""
        # Allow 0 indices (scalars) or >= 2 indices
        # Disallow exactly 1 index (breaks symmetry semantics)
        if len(self.indices) == 1:
            raise ValueError(
                "Tensors with exactly 1 index cannot properly enforce symmetry constraints. "
                "Use 0 indices for scalars or >= 2 indices for tensors."
            )
        if len(self.itags) != len(self.indices):
            raise ValueError(
                f"Number of itags ({len(self.itags)}) must match number of indices ({len(self.indices)})"
            )
        # Validate all indices share the same symmetry group
        if len(self.indices) >= 2:
            first_group = self.indices[0].group
            for i, idx in enumerate(self.indices[1:], start=1):
                if idx.group != first_group:
                    raise ValueError(
                        f"All indices must share the same symmetry group. "
                        f"Index 0 has {type(first_group).__name__}, "
                        f"but index {i} has {type(idx.group).__name__}"
                    )
        # For scalars, ensure only neutral charge block exists
        if len(self.indices) == 0:
            if len(self.data) > 1:
                raise ValueError("Scalar tensors can only have one block (neutral charge)")
            if self.data and () not in self.data:
                raise ValueError("Scalar tensor must have empty tuple () as key")
            # Set default label for scalars if still using the default "Tensor" label
            if self.label == "Tensor":
                object.__setattr__(self, 'label', "Scalar")
        BlockSchema.validate_blocks(self.indices, self.data)
        for key in self.data:
            if not BlockSchema.charges_conserved(self.indices, key):
                raise ValueError(
                    f"Block {key} violates charge conservation for assigned index directions"
                )

    # ------------------------------------------------------------
    #   Constructors: zero and random tensors
    # ------------------------------------------------------------

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
        
        # normalize indices to only include sectors that actually appear in the data
        normalized_indices = cls._prune_unused_sectors(indices_tuple, data)
        return cls(indices=normalized_indices, itags=itags_tuple, data=data, dtype=dtype)

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
        
        # Prune indices to only include sectors that actually appear in the data
        normalized_indices = cls._prune_unused_sectors(indices_tuple, data)
        return cls(indices=normalized_indices, itags=itags_tuple, data=data, dtype=target_dtype)

    @staticmethod
    def _prune_unused_sectors(indices: Tuple[Index, ...], data: Dict[BlockKey, np.ndarray]) -> Tuple[Index, ...]:
        """Remove sectors from indices that don't appear in any block."""
        if not data:
            # No blocks, return empty indices
            return tuple(Index(idx.direction, idx.group, sectors=()) for idx in indices)
        
        # Collect which charges appear in each axis
        charges_per_axis = [set() for _ in indices]
        for block_key in data.keys():
            for axis, charge in enumerate(block_key):
                charges_per_axis[axis].add(charge)
        
        # Build new indices with only used sectors
        normalized_indices = []
        for axis, idx in enumerate(indices):
            used_charges = charges_per_axis[axis]
            # Filter sectors to only those whose charges appear
            used_sectors = tuple(
                sector for sector in idx.sectors
                if sector.charge in used_charges
            )
            normalized_indices.append(
                Index(idx.direction, idx.group, sectors=used_sectors)
            )
        
        return tuple(normalized_indices)

    # ------------------------------------------------------------
    #   Constructors: scalar as 0D tensor
    # ------------------------------------------------------------

    @classmethod
    def from_scalar(cls, value: Union[int, float, complex], dtype=np.float64, label: str = "Scalar") -> Tensor:
        """Create a scalar (0D tensor) with a single value."""
        data = {(): np.array(value, dtype=dtype)}
        return cls(indices=(), itags=(), data=data, dtype=dtype, label=label)

    def is_scalar(self) -> bool:
        """Check if this tensor is a scalar (0D)."""
        return len(self.indices) == 0

    def item(self) -> Union[int, float, complex]:
        """Extract the scalar value from a 0D tensor."""
        if not self.is_scalar():
            raise ValueError(
                f"item() can only be called on scalars (0D tensors), got {len(self.indices)} indices"
            )
        if len(self.data) != 1 or () not in self.data:
            raise ValueError("Scalar tensor must have exactly one block with empty key ()")
        value = self.data[()]
        if value.shape != ():
            raise ValueError(f"Scalar tensor block must be 0D, got shape {value.shape}")
        return value.item()

    # ------------------------------------------------------------
    #   String representation and display
    # ------------------------------------------------------------

    def __str__(self) -> str:
        """Return a formatted multiline summary generated by `tensor_summary`."""
        return tensor_summary(self.indices, self.itags, self.data, self.dtype, 
                              self.label, self.norm(), self.sorted_keys)

    __repr__ = __str__

    def show(self, block_indices: Sequence[int]) -> None:
        """Display selected blocks without max_line limits.
        """
        # Convert block indices to their corresponding keys
        selected_keys = [self.key(i) for i in block_indices]
        
        # Call tensor_summary with selected keys, original block numbers, and no max_lines limit
        print(tensor_summary(self.indices, self.itags, self.data, self.dtype, self.label, self.norm(),
                             sorted_keys=selected_keys, max_lines=None, block_numbers=list(block_indices)))

    # ------------------------------------------------------------
    #   Utility methods: norm, copy, and sector access
    # ------------------------------------------------------------

    def norm(self) -> float:
        """Compute the Frobenius norm aggregated across all dense blocks."""
        if not self.data:
            return 0.0
        return float(
            np.sqrt(sum(np.sum(np.abs(block) ** 2) for block in self.data.values()))
        )

    def copy(self) -> Tensor:
        """Create a deep copy of this tensor."""
        new_data = {k: v.copy() for k, v in self.data.items()}
        return Tensor(
            indices=self.indices,
            itags=self.itags,
            data=new_data,
            dtype=self.dtype,
            label=self.label,
        )

    def _invalidate_sorted_keys(self) -> None:
        """Clear the sorted keys cache (call after modifying data)."""
        object.__setattr__(self, '_sorted_keys', None)

    @property
    def sorted_keys(self) -> Tuple[BlockKey, ...]:
        """Return block keys sorted in display order (cached)."""
        if self._sorted_keys is None:
            object.__setattr__(self, '_sorted_keys', 
                               tuple(sorted(self.data.keys(), key=str)))
        return self._sorted_keys

    def key(self, i: int) -> BlockKey:
        """Get the BlockKey for the i-th block (1-indexed, matching display)."""
        keys = self.sorted_keys
        if i < 1 or i > len(keys):
            raise IndexError(f"Block index {i} out of range [1, {len(keys)}]")
        return keys[i - 1]

    def block(self, i: int) -> np.ndarray:
        """Access the i-th block by integer index (1-indexed, matching display)."""
        return self.data[self.key(i)]

    # ------------------------------------------------------------
    #   Utility methods: rand_fill, insert_index
    # ------------------------------------------------------------

    @property
    def group(self) -> SymmetryGroup:
        """Fetch the symmetry group of this tensor."""
        if len(self.indices) == 0:
            raise ValueError("Scalar tensor has no symmetry group")
        return self.indices[0].group

    def rand_fill(self, seed: Optional[int] = None) -> None:
        """Fill all data blocks with random values in-place."""
        rng = np.random.default_rng(seed)
        for key in self.data:
            shape = self.data[key].shape
            if np.issubdtype(self.dtype, np.complexfloating):
                real = rng.standard_normal(shape)
                imag = rng.standard_normal(shape)
                self.data[key] = (real + 1j * imag).astype(self.dtype, copy=False)
            else:
                self.data[key] = rng.standard_normal(shape).astype(self.dtype, copy=False)

    def insert_index(self, position: int, direction: Direction, itag: Optional[str] = None) -> None:
        """Insert a trivial index (neutral charge, dimension 1) at a specified position.
        
        Parameters
        ----------
        position:
            Position where the new index should be inserted (0-indexed).
            Must be in range [0, len(self.indices)].
        direction:
            Direction for the new index (Direction.IN or Direction.OUT).
        itag:
            Optional tag for the new index. If None, uses "_init_".
        
        Notes
        -----
        This operation modifies the tensor in-place by:
        - Inserting a new index with a single sector (neutral charge, dimension 1)
        - Adding a singleton dimension to all data blocks at the corresponding axis
        - Updating block keys to include the neutral charge at the new position
        
        The symmetry group for the new index is taken from the existing indices.
        """
        # Validate position
        n = len(self.indices)
        if position < 0 or position > n:
            raise ValueError(f"Position {position} out of range [0, {n}]")
        
        # Get the symmetry group from existing indices
        if n == 0:
            raise ValueError("Cannot insert index into scalar tensor")
        group = self.indices[0].group
        
        # Create trivial index with neutral charge and dimension 1
        neutral_charge = group.neutral
        trivial_sector = Sector(neutral_charge, 1)
        new_index = Index(direction, group, sectors=(trivial_sector,))
        
        # Insert the new index
        indices_list = list(self.indices)
        indices_list.insert(position, new_index)
        self.indices = tuple(indices_list)
        
        # Insert the new itag
        if itag is None:
            itag = "_init_"
        itags_list = list(self.itags)
        itags_list.insert(position, itag)
        self.itags = tuple(itags_list)
        
        # Update data blocks: insert neutral charge in keys and add singleton dimension
        new_data = {}
        for key, arr in self.data.items():
            # Insert neutral charge at the appropriate position in the key
            key_list = list(key)
            key_list.insert(position, neutral_charge)
            new_key = tuple(key_list)
            
            # Add singleton dimension at the appropriate axis
            new_data[new_key] = np.expand_dims(arr, axis=position)
        
        self.data = new_data
        self._invalidate_sorted_keys()

    # ------------------------------------------------------------
    #   Binary operations: add, sub, mul
    # ------------------------------------------------------------

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
        # Special case for scalar + scalar
        if self.is_scalar() and other.is_scalar():
            value = self.item() + other.item()
            return Tensor.from_scalar(
                value, 
                dtype=np.result_type(self.dtype, other.dtype),
                label=self.label
            )
        
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
        # Special case for scalar - scalar
        if self.is_scalar() and other.is_scalar():
            value = self.item() - other.item()
            return Tensor.from_scalar(
                value,
                dtype=np.result_type(self.dtype, other.dtype),
                label=self.label
            )
        
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
        # Special case for scalar tensor * scalar value
        if self.is_scalar():
            value = self.item() * scalar
            return Tensor.from_scalar(
                value,
                dtype=np.result_type(self.dtype, type(scalar)),
                label=self.label
            )
        
        new_data = {k: (v * scalar) for k, v in self.data.items()}
        return Tensor(
            indices=self.indices,
            itags=self.itags,
            data=new_data,
            dtype=np.result_type(self.dtype, type(scalar)),
            label=self.label,
        )

    __rmul__ = __mul__

    # ------------------------------------------------------------
    #   Tensor operations: conj, permute, transpose
    # ------------------------------------------------------------

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
        self._invalidate_sorted_keys()

    def transpose(self, *order: int) -> None:
        """Transpose tensor axes; defaults to reversing the index order."""
        if not order:
            order = tuple(reversed(range(len(self.indices))))
        self.permute(order)

    def flip(self, positions: Union[int, Sequence[int]]) -> None:
        """Flip the direction of specified index/indices while maintaining charge conservation.
        
        Parameters
        ----------
        positions:
            Index position(s) to flip. Can be a single int or a sequence of ints.
            Positions are 0-indexed.
        
        Notes
        -----
        This operation uses Index.dual() to flip both the direction and conjugate
        the charges, ensuring charge conservation is maintained. The tensor data
        blocks are unchanged, but the index metadata (direction and charges) are
        updated to reflect the flipped arrow convention.
        
        This differs from Index.flip() which only reverses direction without
        conjugating charges. For tensors, we need charge conjugation to maintain
        proper charge conservation rules.
        
        Examples
        --------
        # Flip a single index at position 0
        tensor.flip(0)
        
        # Flip multiple indices at positions 0 and 2
        tensor.flip([0, 2])
        """
        # Normalize to a sequence
        if isinstance(positions, int):
            positions = [positions]
        
        # Validate positions
        n = len(self.indices)
        for pos in positions:
            if pos < 0 or pos >= n:
                raise IndexError(f"Index position {pos} out of range [0, {n})")
        
        # Create new indices with dual (flipped direction + conjugated charges) at specified positions
        indices_list = list(self.indices)
        for pos in positions:
            indices_list[pos] = indices_list[pos].dual()
        self.indices = tuple(indices_list)

    # ------------------------------------------------------------
    #   itag manipulations: multiple modes of retagging
    # ------------------------------------------------------------

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
