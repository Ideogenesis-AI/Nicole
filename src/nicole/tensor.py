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


"""Tensor container for block-symmetric data structures.

This module defines the `Tensor` dataclass, which stores symmetry-aware tensor
indices alongside a dictionary of dense PyTorch tensor blocks. Helper constructors create
zero-filled or random tensors, while arithmetic and structural operations respect
charge conservation dictated by the index metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple, Union, Optional

import torch

from .blocks import BlockKey, BlockSchema
from .index import Index, union_indices
from .typing import Direction, Sector
from .typing import normalize_dtype_for_device
from .symmetry import SymmetryGroup
from .symmetry import delegate as dg

# Disable autograd by default for performance (tensor networks rarely need gradients)
torch.set_grad_enabled(False)

# Set default device to CPU (users can change via torch.set_default_device if needed)
torch.set_default_device('cpu')


@dataclass
class Tensor:
    """Block-sparse tensor backed by symmetry-aware indices and dense blocks.

    Each `Tensor` pairs an ordered tuple of `Index` instances with a mapping from
    block keys (one charge per axis) to dense PyTorch tensors. Arithmetic operations
    are defined in a way that preserves charge conservation, and helper methods
    provide convenient constructors and transformations.

    Attributes
    ----------
    indices:
        Ordered tuple of `Index` instances defining the symmetry structure of the tensor.
    itags:
        Ordered tuple of human-readable labels for each index.
    data:
        Mapping from block keys (one charge per axis) to dense PyTorch tensors.
    intw:
        Mapping from block keys to intertwiners delegated to Yuzuha protocol.
    dtype:
        Data type for the dense blocks. Defaults to double precision real values.
    label:
        Human-readable label for the tensor. Defaults to "Tensor".
    device:
        Device where tensor blocks are stored (CPU or GPU).

    Methods
    -------
    zeros()
        Create a symmetry-aware tensor with admissible zero-filled blocks.
    random()
        Create a tensor filled with random values for each admissible block.
    from_scalar()
        Create a scalar (0D tensor) with a single value.
    is_scalar()
        Check if this tensor is a scalar (0D).
    item()
        Extract the scalar value from a 0D tensor.
    norm()
        Compute the Frobenius norm aggregated across all dense blocks.
    clone()
        Create a deep clone of this tensor with independent block data.
    rand_fill()
        In-place: Fill all data blocks with random values.
    insert_index()
        In-place: Insert a trivial index (neutral charge, dimension 1) at a position.
    normalize_sectors()
        In-place: Remove sectors from each index that do not appear in any block.
    trim_zero_blocks()
        In-place: Remove blocks whose magnitude is negligible relative to the norm.
    device
        Property returning the device where tensor blocks are stored.
    to()
        Move tensor to specified device (CPU, CUDA, MPS, etc.).
    cpu()
        Move tensor to CPU.
    cuda()
        Move tensor to CUDA device.
    requires_grad
        Property for checking/setting gradient tracking.
    backward()
        Compute gradients by backpropagating through the computational graph (scalars only).
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
    compress()
        In-place: Reduce redundant intertwiner components via SVD truncation.
    regularize()
        In-place: Canonicalize or regularize Bridge weights.
    conj()
        Complex conjugate every dense block, and revert all index directions.
    permute()
        Permute tensor axes according to the provided reordering.
    transpose()
        Transpose by reversing all tensor axes.
    invert()
        In-place: Invert the direction of specified index/indices.
    retag()
        Retag indices: update specific tags by name/index, or replace all tags.
    
    Notes
    -----
    Standalone functions in `nicole.maneuver` deep-clone all data blocks for full
    isolation. Method forms default to `in_place=False` and share storage (torch views).
    """

    indices: Tuple[Index, ...]
    itags: Tuple[str, ...]
    data: MutableMapping[BlockKey, torch.Tensor]
    intw: Optional[MutableMapping[BlockKey, dg.Bridge]] = field(default=None)
    dtype: torch.dtype = torch.float64
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
        # Validate intertwiners (intw) and block shapes for generic groups
        if len(self.indices) > 0:
            group = self.indices[0].group
            if not group.is_abelian:
                if self.intw is None:
                    raise ValueError("Generic (non-Abelian) tensors must have intertwiner (intw) populated")
                # Validate each Bridge has correct number of external edges
                for key, bridge in self.intw.items():
                    if bridge.num_external != len(self.indices):
                        raise ValueError(
                            f"Bridge for key {key} has {bridge.num_external} edges, "
                            f"expected {len(self.indices)}"
                        )
        
        # Validate block shapes (includes intw validation for generic groups)
        BlockSchema.validate_blocks(self.indices, self.data, self.intw)
        for key in self.data:
            if not BlockSchema.charges_conserved(self.indices, key):
                raise ValueError(
                    f"Block {key} violates charge conservation for assigned index directions"
                )

    # ------------------------------------------------------------
    #   Constructors: zero and random tensors
    # ------------------------------------------------------------

    @classmethod
    def zeros(
        cls, 
        indices: Sequence[Index], 
        dtype: torch.dtype = torch.float64, 
        itags: Optional[Sequence[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        requires_grad: bool = False
    ) -> Tensor:
        """Create a symmetry-aware tensor with admissible zero-filled blocks.
        
        Parameters
        ----------
        indices : Sequence[Index]
            Sequence of Index objects defining the tensor structure
        dtype : torch.dtype, optional
            Data type for the tensor blocks (default: torch.float64)
        itags : Sequence[str], optional
            Tags for each index (default: "_init_" for all)
        device : str or torch.device, optional
            Device to place tensors on (default: current default device)
        requires_grad : bool, optional
            If True, enables gradient tracking for this tensor (default: False)
            
        Notes
        -----
        Gradient tracking follows PyTorch's default behavior. Set requires_grad=True
        to enable autograd for this tensor. Use torch.no_grad() context to temporarily
        disable gradient computation during operations.
        
        MPS (Apple Silicon) doesn't support float64/complex128. If creating on MPS with
        these dtypes, they will be automatically downgraded to float32/complex64.
        
        For generic symmetry groups (e.g., SU2), intertwiners (intw) are automatically
        populated with Bridge objects containing default Clebsch-Gordan specifications.
        """
        if device is None:
            device = torch.get_default_device()
        device = torch.device(device)
        
        # Normalize dtype for device compatibility (e.g., MPS doesn't support float64)
        dtype = normalize_dtype_for_device(dtype, device)
        
        # Normalise input to an immutable tuple for downstream utilities.
        indices_tuple = tuple(indices)
        if itags is None:
            itags_tuple = tuple(f"_init_" for _ in indices_tuple)
        else:
            itags_tuple = tuple(itags)
        
        # Create intertwiner (intw) for generic groups first
        intw: Optional[Dict[BlockKey, dg.Bridge]] = None
        if indices_tuple and not indices_tuple[0].group.is_abelian:
            intw: Dict[BlockKey, dg.Bridge] = {}
            directions = [idx.direction for idx in indices_tuple]
            for key in BlockSchema.iter_admissible_keys(indices_tuple):
                if not BlockSchema.charges_conserved(indices_tuple, key):
                    continue
                intw[key] = dg.Bridge.from_block(group=indices_tuple[0].group,
                    key=key, directions=directions, dtype=dtype, device=device)
        
        data: Dict[BlockKey, torch.Tensor] = {}
        # Iterate over all admissible charge assignments for the provided indices.
        for key in BlockSchema.iter_admissible_keys(indices_tuple):
            if not BlockSchema.charges_conserved(indices_tuple, key):
                continue
            # Determine the dense shape implied by the current key and allocate zeros.
            if intw is not None and key in intw:
                # Non-Abelian: append trailing reduced multiplicity dimension
                shape = BlockSchema.shape_for_key(indices_tuple, key, num_components=intw[key].num_components)
            else: # Abelian: no trailing dimension
                shape = BlockSchema.shape_for_key(indices_tuple, key)
            block = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
            data[key] = block
        
        # normalize indices to only include sectors that actually appear in the data
        normalized_indices = cls._prune_unused_sectors(indices_tuple, data)
        return cls(indices=normalized_indices, itags=itags_tuple, data=data, intw=intw, dtype=dtype)

    @classmethod
    def random(
        cls, 
        indices: Sequence[Index], 
        dtype: torch.dtype = torch.float64, 
        seed: Optional[int] = None, 
        itags: Optional[Sequence[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        requires_grad: bool = False
    ) -> Tensor:
        """Create a tensor filled with random values for each admissible block.
        
        Parameters
        ----------
        indices : Sequence[Index]
            Sequence of Index objects defining the tensor structure
        dtype : torch.dtype, optional
            Data type for the tensor blocks (default: torch.float64)
        seed : int, optional
            Random seed for reproducibility
        itags : Sequence[str], optional
            Tags for each index (default: "_init_" for all)
        device : str or torch.device, optional
            Device to place tensors on (default: current default device)
        requires_grad : bool, optional
            If True, enables gradient tracking for this tensor (default: False)
            
        Notes
        -----
        Gradient tracking follows PyTorch's default behavior. Set requires_grad=True
        to enable autograd for this tensor. Use torch.no_grad() context to temporarily
        disable gradient computation during operations.
        
        MPS (Apple Silicon) doesn't support float64/complex128. If creating on MPS with
        these dtypes, they will be automatically downgraded to float32/complex64.
        
        For generic symmetry groups (e.g., SU2), intertwiners (intw) are automatically
        populated with Bridge objects containing default Clebsch-Gordan specifications.
        """
        if device is None:
            device = torch.get_default_device()
        device = torch.device(device)
        
        # Normalize dtype for device compatibility (e.g., MPS doesn't support float64)
        dtype = normalize_dtype_for_device(dtype, device)
        
        # Initialise the random number generator.
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
        else:
            gen = None
            
        indices_tuple = tuple(indices)
        if itags is None:
            itags_tuple = tuple(f"_init_" for _ in indices_tuple)
        else:
            itags_tuple = tuple(itags)
        
        # Create intertwiner (intw) for generic (non-Abelian) groups first
        intw: Optional[Dict[BlockKey, dg.Bridge]] = None
        if indices_tuple and not indices_tuple[0].group.is_abelian:
            intw: Dict[BlockKey, dg.Bridge] = {}
            directions = [idx.direction for idx in indices_tuple]
            for key in BlockSchema.iter_admissible_keys(indices_tuple):
                if not BlockSchema.charges_conserved(indices_tuple, key):
                    continue
                intw[key] = dg.Bridge.from_block(group=indices_tuple[0].group,
                    key=key, directions=directions, dtype=dtype, device=device)
        
        data: Dict[BlockKey, torch.Tensor] = {}
        # Walk through admissible blocks in the same fashion as `zeros`.
        for key in BlockSchema.iter_admissible_keys(indices_tuple):
            if not BlockSchema.charges_conserved(indices_tuple, key):
                continue
            # Determine the dense shape implied by the current key
            if intw is not None and key in intw:
                # Non-Abelian: append trailing reduced multiplicity dimension
                shape = BlockSchema.shape_for_key(indices_tuple, key, num_components=intw[key].num_components)
            else: # Abelian: no trailing dimension
                shape = BlockSchema.shape_for_key(indices_tuple, key)
            
            if dtype.is_complex:
                real = torch.randn(shape, generator=gen, device=device,
                    dtype=torch.float64 if dtype == torch.complex128 else torch.float32)
                imag = torch.randn(shape, generator=gen, device=device,
                    dtype=torch.float64 if dtype == torch.complex128 else torch.float32)
                arr = torch.complex(real, imag)
                if requires_grad:
                    arr.requires_grad_(True)
            else:
                arr = torch.randn(shape, generator=gen, device=device, dtype=dtype,
                    requires_grad=requires_grad)
            data[key] = arr
        
        # Prune indices to only include sectors that actually appear in the data
        normalized_indices = cls._prune_unused_sectors(indices_tuple, data)
        return cls(indices=normalized_indices, itags=itags_tuple, data=data, intw=intw, dtype=dtype)

    @staticmethod
    def _prune_unused_sectors(
        indices: Tuple[Index, ...],
        data: MutableMapping[BlockKey, torch.Tensor]
    ) -> Tuple[Index, ...]:
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
    def from_scalar(
        cls, 
        value: Union[int, float, complex], 
        dtype: torch.dtype = torch.float64, 
        label: str = "Scalar",
        device: Optional[Union[str, torch.device]] = None,
        requires_grad: bool = False
    ) -> Tensor:
        """Create a scalar (0D tensor) with a single value.
        
        Parameters
        ----------
        value : int, float, or complex
            Scalar value
        dtype : torch.dtype, optional
            Data type (default: torch.float64)
        label : str, optional
            Label for the scalar (default: "Scalar")
        device : str or torch.device, optional
            Device to place tensor on (default: current default device)
        requires_grad : bool, optional
            If True, enables gradient tracking for this tensor (default: False)
            
        Notes
        -----
        MPS (Apple Silicon) doesn't support float64/complex128. If creating on MPS with
        these dtypes, they will be automatically downgraded to float32/complex64.
        """
        if device is None:
            device = torch.get_default_device()
        device = torch.device(device)
        
        # Normalize dtype for device compatibility (e.g., MPS doesn't support float64)
        dtype = normalize_dtype_for_device(dtype, device)
        
        block = torch.tensor(value, dtype=dtype, device=device, requires_grad=requires_grad)
        data = {(): block}
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
        from .display import tensor_summary
        indices = Tensor._prune_unused_sectors(self.indices, self.data)
        return tensor_summary(indices, self.itags, self.data, self.intw,
            self.dtype, self.label, self.norm(), self.sorted_keys)

    __repr__ = __str__

    def show(self, block_ids: Sequence[int]) -> None:
        """Display selected blocks without max_line limits."""
        from .display import tensor_summary
        
        # Convert single integer to list
        if isinstance(block_ids, int):
            block_ids = [block_ids]
        # Convert block ids to their corresponding keys
        selected_keys = [self.key(i) for i in block_ids]
        
        indices = Tensor._prune_unused_sectors(self.indices, self.data)
        # Call tensor_summary with selected keys, original block ids, and no max_lines limit
        print(tensor_summary(indices, self.itags, self.data, self.intw, self.dtype, self.label, self.norm(),
            sorted_keys=selected_keys, max_lines=None, block_ids=list(block_ids)))
    
    # ------------------------------------------------------------
    #   Device management: cpu, cuda, mps
    # ------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Return the device of the tensor blocks."""
        if not self.data:
            return torch.get_default_device()
        # All blocks must be on same device
        return next(iter(self.data.values())).device
    
    def to(self, device: Union[str, torch.device]) -> Tensor:
        """Move tensor to specified device.
        
        Parameters
        ----------
        device : str or torch.device
            Target device ('cpu', 'cuda', 'mps', etc.)
            
        Returns
        -------
        Tensor
            New tensor on the specified device
        
        Notes
        -----
        MPS (Apple Silicon) doesn't support float64/complex128. If moving a tensor with
        these dtypes to MPS, they will be automatically downgraded to float32/complex64.
        """
        device = torch.device(device)
        if device == self.device:
            return self
        
        # Normalize dtype for device compatibility (e.g., MPS doesn't support float64)
        new_dtype = normalize_dtype_for_device(self.dtype, device)
        
        if new_dtype != self.dtype:
            # Need to convert dtype as well as device
            new_data = {k: v.to(device, dtype=new_dtype) for k, v in self.data.items()}
        else:
            # Just move to new device
            new_data = {k: v.to(device) for k, v in self.data.items()}
        
        # Move intertwiner to new device/dtype
        new_intw: Optional[Dict[BlockKey, dg.Bridge]] = None
        if self.intw is not None:
            new_intw = {k: bridge.to(device, dtype=new_dtype) for k, bridge in self.intw.items()}
        
        return Tensor(
            indices=self.indices, itags=self.itags, data=new_data, intw=new_intw,
            dtype=new_dtype, label=self.label
        )
    
    def cpu(self) -> Tensor:
        """Move tensor to CPU."""
        return self.to('cpu')
    
    def cuda(self) -> Tensor:
        """Move tensor to CUDA device."""
        return self.to('cuda')
    
    # ------------------------------------------------------------
    #   Autograd control: requires_grad
    # ------------------------------------------------------------
    
    @property
    def requires_grad(self) -> bool:
        """Check if this tensor tracks gradients.
        
        Returns True if all underlying blocks have requires_grad=True,
        False otherwise.
        
        Returns
        -------
        bool
            Whether this tensor tracks gradients
            
        Examples
        --------
        >>> t = Tensor.zeros(indices)
        >>> print(t.requires_grad)  # False
        >>> 
        >>> t.requires_grad = True
        >>> print(t.requires_grad)  # True
        """
        if not self.data:
            return False
        return all(block.requires_grad for block in self.data.values())
    
    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Set gradient tracking for this tensor.
        
        Sets requires_grad for all underlying torch.Tensor blocks.
        
        Parameters
        ----------
        value : bool
            Whether to track gradients for this tensor
            
        Examples
        --------
        >>> t = Tensor.random(indices)
        >>> t.requires_grad = True  # Enable gradient tracking
        >>> 
        >>> # Operations will now build computational graphs
        >>> result = t * 2
        >>> 
        >>> # Disable for inference
        >>> t.requires_grad = False
        """
        for block in self.data.values():
            block.requires_grad_(value)
    
    def backward(self) -> None:
        """Compute gradients by backpropagating through the computational graph.
        
        This method can only be called on scalar tensors (0D tensors). It calls
        the backward() method on the underlying PyTorch tensor to compute gradients
        for all tensors in the computational graph that have requires_grad=True.
        
        Raises
        ------
        ValueError
            If the tensor is not a scalar (has more than 0 dimensions)
            
        Examples
        --------
        >>> # Create tensors with gradient tracking
        >>> t = Tensor.random(indices, requires_grad=True)
        >>> 
        >>> # Perform operations
        >>> loss = contract(t, t, ...)  # Some operation resulting in a scalar
        >>> 
        >>> # Compute gradients
        >>> loss.backward()
        >>> 
        >>> # Access gradients from underlying PyTorch tensors
        >>> for block in t.data.values():
        ...     print(block.grad)
        """
        if not self.is_scalar():
            raise ValueError(
                f"backward() can only be called on scalars (0D tensors), got {len(self.indices)} indices"
            )
        # Verify the underlying torch tensor is a scalar (single element)
        block = self.data[()]
        if block.numel() != 1:
            raise ValueError(
                f"backward() requires a scalar tensor with 1 element, got {block.numel()} elements"
            )
        # Call backward on the underlying PyTorch scalar tensor
        block.backward()

    # ------------------------------------------------------------
    #   Utility methods: norm, clone, and sector access
    # ------------------------------------------------------------

    def norm(self) -> float:
        """Compute the Frobenius norm aggregated across all dense blocks."""
        if not self.data:
            return 0.0
        
        # Abelian: direct Frobenius norm
        if not self.indices or self.indices[0].group.is_abelian:
            return float(
                torch.sqrt(sum(torch.sum(torch.abs(block) ** 2) for block in self.data.values()))
            )
        else:
            # Generic: ||T||² = Σ_blocks Tr(W† R† R W)
            total = 0.0
            for key, block in self.data.items():
                r_flat = block.flatten(0, -2)  # (d₁...dₙ, r)
                weights = self.intw[key].weights  # (r, μ)
                gram = r_flat.T.conj() @ r_flat  # (r, r)
                total += torch.sum(weights.conj() * (gram @ weights)).real
            return float(torch.sqrt(total))

    def clone(self) -> Tensor:
        """Create a deep clone of this tensor."""
        new_data = {k: v.clone() for k, v in self.data.items()}
        
        # Deep clone intertwiner
        new_intw: Optional[Dict[BlockKey, dg.Bridge]] = None
        if self.intw is not None:
            new_intw = {k: bridge.clone() for k, bridge in self.intw.items()}
        
        return Tensor(
            indices=self.indices, itags=self.itags, data=new_data, intw=new_intw,
            dtype=self.dtype, label=self.label
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

    def block(self, i: int) -> torch.Tensor:
        """Access the i-th block by integer index (1-indexed, matching display)."""
        return self.data[self.key(i)]

    # ------------------------------------------------------------
    #   Utility methods: rand_fill, insert_index, trim_zeros
    # ------------------------------------------------------------

    @property
    def group(self) -> SymmetryGroup:
        """Fetch the symmetry group of this tensor."""
        if len(self.indices) == 0:
            raise ValueError("Scalar tensor has no symmetry group")
        return self.indices[0].group

    def rand_fill(self, seed: Optional[int] = None) -> None:
        """Fill all data blocks with random values in-place."""
        device = self.device
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)
        else:
            gen = None
        for key in self.data:
            shape = self.data[key].shape
            if self.dtype.is_complex:
                real = torch.randn(shape, generator=gen, device=device, 
                    dtype=torch.float64 if self.dtype == torch.complex128 else torch.float32)
                imag = torch.randn(shape, generator=gen, device=device,
                    dtype=torch.float64 if self.dtype == torch.complex128 else torch.float32)
                self.data[key] = torch.complex(real, imag)
            else:
                self.data[key] = torch.randn(shape, generator=gen, device=device, dtype=self.dtype)

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
        
        For non-Abelian groups (e.g. SU(2)), each intertwiner (Bridge) is updated
        via `Bridge.insert_edge`, which inserts the neutral-charge edge and
        applies the appropriate R-symbol so that the result is consistent with
        a direct permutation of the new index to `position`. The OM dimension
        is preserved exactly since the neutral irrep does not participate in coupling.
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
        new_data: Dict[BlockKey, torch.Tensor] = {}
        new_intw: Optional[Dict[BlockKey, dg.Bridge]] = None
        if self.intw is not None:
            new_intw = {}
        
        for key, arr in self.data.items():
            # Insert neutral charge at the appropriate position in the key
            key_list = list(key)
            key_list.insert(position, neutral_charge)
            new_key = tuple(key_list)
            
            # Add singleton dimension at the appropriate axis
            new_data[new_key] = torch.unsqueeze(arr, dim=position)
            
            if new_intw is not None:
                new_intw[new_key] = self.intw[key].insert_edge(position, direction)
        
        self.data = new_data
        self.intw = new_intw
        self._invalidate_sorted_keys()

    def normalize_sectors(self) -> None:
        """Remove sectors from each index that do not appear in any block.

        This operation modifies the tensor in-place by updating `self.indices`
        so that only sectors whose charges are referenced by at least one block
        key are retained.
        """
        self.indices = Tensor._prune_unused_sectors(self.indices, self.data)

    def trim_zero_blocks(self, eps: Optional[float] = None) -> None:
        """Remove blocks whose data is negligible relative to the tensor's overall scale.

        This operation modifies the tensor in-place by:
        - Removing blocks from self.data where max(abs(values)) < eps * norm
        - For generic groups, also removing blocks where all weights are similarly negligible
        - Updating each index to only include sectors that still have data in remaining blocks

        Parameters
        ----------
        eps : float or None
            Relative tolerance. A block is considered zero when its maximum absolute
            value is less than `eps * self.norm()`. Defaults to
            `torch.finfo(torch.float64).eps` (~2.2e-16) when None.

        Notes
        -----
        Using the Frobenius norm as the scale makes the criterion fully relative: a block
        is trimmed only when it is negligible compared to the tensor as a whole, regardless
        of the absolute magnitude of individual entries.

        If the tensor is identically zero (norm == 0) all blocks are removed.
        """
        if eps is None:
            eps = torch.finfo(torch.float64).eps

        threshold = eps * self.norm()  # == 0.0 when tensor is identically zero

        # Step 1: Identify and remove blocks with all near-zero values
        blocks_to_remove = []
        for key, arr in self.data.items():
            is_data_zero = torch.max(torch.abs(arr)).item() <= threshold
            is_weights_zero = False

            # For generic groups: also check if weights are zero (T = R @ 0 = 0)
            if self.intw is not None and key in self.intw:
                weights = self.intw[key].weights
                is_weights_zero = torch.max(torch.abs(weights)).item() <= threshold
            
            if is_data_zero or is_weights_zero:
                blocks_to_remove.append(key)
        
        for key in blocks_to_remove:
            del self.data[key]
            if self.intw is not None and key in self.intw:
                del self.intw[key]
        
        # Step 2: Determine which charges are still present at each index position
        n_indices = len(self.indices)
        if n_indices == 0 or len(self.data) == 0:
            # Scalar tensor or no data left
            self._invalidate_sorted_keys()
            return
        
        # Collect charges that appear in remaining blocks for each index position
        charges_present = [set() for _ in range(n_indices)]
        for key in self.data.keys():
            for i, charge in enumerate(key):
                charges_present[i].add(charge)
        
        # Step 3: Rebuild each index to only include sectors with present charges
        new_indices = []
        for i, idx in enumerate(self.indices):
            present = charges_present[i]
            # Filter sectors to keep only those with charges still in data
            new_sectors = [s for s in idx.sectors if s.charge in present]
            
            if len(new_sectors) == 0:
                # No sectors remain for this index - this shouldn't happen with valid data
                # but handle gracefully by keeping the original index
                new_indices.append(idx)
            else:
                # Create new index with filtered sectors
                new_index = Index(
                    direction=idx.direction,
                    group=idx.group,
                    sectors=tuple(new_sectors)
                )
                new_indices.append(new_index)
        
        self.indices = tuple(new_indices)
        self._invalidate_sorted_keys()

    # ------------------------------------------------------------
    #   Binary operations: add, sub, mul
    # ------------------------------------------------------------

    def _align_for_binary(self, other: Tensor) -> Tuple[Tensor, Tensor]:
        """Ensure two tensors are compatible for element-wise binary operations."""
        if len(self.indices) != len(other.indices):
            raise ValueError("Cannot add/sub tensors with different order")
        if any((a.group != b.group) or (a.direction != b.direction) for a, b in zip(self.indices, other.indices)):
            raise ValueError("Indices groups and directions must match")
        
        return self, other

    def __add__(self, other: Tensor) -> Tensor:
        """Element-wise addition while preserving symmetry metadata.
        
        For generic groups (SU(2)), handles intertwiner weights:
        - If weights match: adds reduced tensors directly
        - If weights differ: concatenates along reduced multiplicity dimension
        """
        # Special case for scalar + scalar
        if self.is_scalar() and other.is_scalar():
            # Perform operation on torch tensors to preserve computational graph
            scalar_data = self.data[()] + other.data[()]
            return Tensor(
                indices=(), itags=(), data={(): scalar_data},
                dtype=torch.promote_types(self.dtype, other.dtype), label=self.label
            )
        
        self._align_for_binary(other)
        
        # Union indices to include all sectors from both tensors
        new_indices = tuple(
            union_indices(idx_a, idx_b) 
            for idx_a, idx_b in zip(self.indices, other.indices)
        )
        
        # Perform addition on blocks
        keys = set(self.data.keys()) | set(other.data.keys())
        new_data: Dict[BlockKey, torch.Tensor] = {}
        new_intw: Optional[Dict[BlockKey, dg.Bridge]] = None
        
        # Abelian groups: direct block addition
        if not self.indices or self.indices[0].group.is_abelian:
            for k in keys:
                a = self.data.get(k)
                b = other.data.get(k)
                if a is None:
                    new_data[k] = (+b)
                elif b is None:
                    new_data[k] = (+a)
                else:
                    new_data[k] = a + b
        else:
            # Non-Abelian groups: handle intertwiners
            new_intw: Dict[BlockKey, dg.Bridge] = {}
            for k in keys:
                a = self.data.get(k)
                b = other.data.get(k)
                bridge_a = self.intw.get(k) if a is not None else None
                bridge_b = other.intw.get(k) if b is not None else None
                
                new_data[k], new_intw[k] = BlockSchema.block_add(
                    a, bridge_a, b, bridge_b, rtol=1e-12, atol=1e-15
                )
        
        return Tensor(
            indices=new_indices, itags=self.itags, data=new_data, intw=new_intw,
            dtype=torch.promote_types(self.dtype, other.dtype), label=self.label
        )

    def __sub__(self, other: Tensor) -> Tensor:
        """Element-wise subtraction while preserving symmetry metadata.
        
        For generic groups (SU(2)), handles intertwiner weights:
        - If weights match: subtracts reduced tensors directly
        - If weights differ: concatenates along reduced multiplicity dimension
        """
        # Special case for scalar - scalar
        if self.is_scalar() and other.is_scalar():
            # Perform operation on torch tensors to preserve computational graph
            scalar_data = self.data[()] - other.data[()]
            return Tensor(
                indices=(), itags=(), data={(): scalar_data},
                dtype=torch.promote_types(self.dtype, other.dtype), label=self.label
            )
        
        self._align_for_binary(other)
        
        # Union indices to include all sectors from both tensors
        new_indices = tuple(
            union_indices(idx_a, idx_b) 
            for idx_a, idx_b in zip(self.indices, other.indices)
        )
        
        # Perform subtraction on blocks
        keys = set(self.data.keys()) | set(other.data.keys())
        new_data: Dict[BlockKey, torch.Tensor] = {}
        new_intw: Optional[Dict[BlockKey, dg.Bridge]] = None
        
        # Abelian groups: direct block subtraction
        if not self.indices or self.indices[0].group.is_abelian:
            for k in keys:
                a = self.data.get(k)
                b = other.data.get(k)
                if a is None:
                    new_data[k] = -b
                elif b is None:
                    new_data[k] = +a
                else:
                    new_data[k] = a - b
        else:
            # Generic (non-Abelian) groups: handle intertwiners
            new_intw: Dict[BlockKey, dg.Bridge] = {}
            for k in keys:
                a = self.data.get(k)
                b = other.data.get(k)
                bridge_a = self.intw.get(k) if a is not None else None
                bridge_b = other.intw.get(k) if b is not None else None
                
                new_data[k], new_intw[k] = BlockSchema.block_add(
                    a, bridge_a, -b if b is not None else None, bridge_b,
                    rtol=1e-12, atol=1e-15
                )
        
        return Tensor(
            indices=new_indices, itags=self.itags, data=new_data, intw=new_intw,
            dtype=torch.promote_types(self.dtype, other.dtype), label=self.label
        )

    def __mul__(self, scalar: Union[int, float, complex]) -> Tensor:
        """Scale every dense block by a scalar."""
        # Special case for scalar tensor * scalar value
        if self.is_scalar():
            # Perform operation on torch tensor to preserve computational graph
            scalar_data = self.data[()] * scalar
            # Determine scalar dtype for promotion
            if isinstance(scalar, complex):
                scalar_dtype = torch.complex128
            elif isinstance(scalar, float):
                scalar_dtype = torch.float64
            else:  # int
                scalar_dtype = torch.int64
            return Tensor(
                indices=(), itags=(), data={(): scalar_data},
                dtype=torch.promote_types(self.dtype, scalar_dtype), label=self.label
            )
        
        new_data = {k: (v * scalar) for k, v in self.data.items()}
        # Determine scalar dtype for promotion
        if isinstance(scalar, complex):
            scalar_dtype = torch.complex128
        elif isinstance(scalar, float):
            scalar_dtype = torch.float64
        else:  # int
            scalar_dtype = torch.int64
        return Tensor(
            indices=self.indices, itags=self.itags, data=new_data, intw=self.intw,
            dtype=torch.promote_types(self.dtype, scalar_dtype), label=self.label
        )

    __rmul__ = __mul__

    # ------------------------------------------------------------
    #   Compression: reduce redundant components
    # ------------------------------------------------------------

    def compress(
        self, 
        keys: Optional[Sequence[BlockKey]] = None, 
        cutoff: float = 1e-14
    ) -> None:
        """Compress intertwiner weights by removing linearly dependent components (in-place).
        
        For generic groups, performs SVD on weight matrices and truncates
        singular values below the cutoff threshold. This reduces the reduced
        multiplicity dimension when weight rows are linearly dependent.
        
        The compression preserves the physical tensor: T = R @ W is decomposed as
        R @ (U @ S @ Vh) ≈ (R @ U @ S) @ Vh, where small singular values are removed.
        
        This operation modifies the tensor in place.
        
        Parameters
        ----------
        keys : Sequence[BlockKey], optional
            Block keys to compress. If None, compresses all blocks with num_components >= 2.
        cutoff : float, optional
            Singular value threshold for truncation. Default: 1e-14.
        
        Examples
        --------
        >>> # After adding tensors with different weights, compress redundancy
        >>> C = A + B  # May have redundant components
        >>> C.compress(cutoff=1e-12)  # Modifies C in place
        """
        # Abelian groups: no compression needed
        if not self.indices or self.indices[0].group.is_abelian:
            return
        
        # Determine which keys to compress
        if keys is None:
            # Compress all blocks with num_components >= 2
            keys_to_compress = [k for k, bridge in self.intw.items() if bridge.num_components >= 2]
        else:
            keys_to_compress = list(keys)
        
        # If no blocks to compress, nothing to do
        if not keys_to_compress:
            return
        
        # Perform compression on specified keys
        for key in keys_to_compress:
            block = self.data[key]
            bridge = self.intw[key]
            
            # Perform SVD compression
            U, S, Vh = torch.linalg.svd(bridge.weights, full_matrices=False)
            
            # Truncate small singular values (keep at least 1)
            k = max(1, (S >= cutoff).sum().item())
            
            if k < bridge.num_components:
                # Compression: absorb U[:, :k] @ diag(S[:k]) into data, keep Vh[:k, :]
                r_flat = block.flatten(0, -2)
                r_new = (r_flat @ (U[:, :k] * S[:k])).reshape(block.shape[:-1] + (k,))
                
                # Update in place
                self.data[key] = r_new
                self.intw[key] = dg.Bridge(cgspec=bridge.cgspec, weights=Vh[:k, :])

    # ------------------------------------------------------------
    #   Weights canonicalisation or regularisation
    # ------------------------------------------------------------

    def regularize(self) -> None:
        """Canonicalize (2nd order) or regularize (higher order) Bridge weights.

        For an 2nd order non-Abelian tensor (SU(2) matrix), the reduced data `R`
        and the Bridge weight `W` satisfy:

            physical block  ≈  R  ×  W

        The method absorbs the deviation of each block's weight from the
        canonical value `sqrt(irrep_dim(q))` into `R`, so that after the
        call the tensor uses the same Bridge-weight convention as
        `identity`:

            physical block  ≈  R_new  ×  sqrt(irrep_dim(q))

        Both branches use a row-normalisation strategy, differing only in target:

        - **2nd-order**: By Schur's lemma `om = 1`, so each weight row is a
          single scalar `W[i, 0]`. The factor is absorbed into the
          corresponding data component so that the canonical positive value
          `sqrt(irrep_dim(q))` is enforced:

                factor[i] = W[i, 0] / sqrt(irrep_dim(q))
                W_new[i, 0] = sqrt(irrep_dim(q))
                R_new[..., i] = R[..., i] * factor[i]

        - **Higher-order**: each row is normalised to unit norm, with the
          norm absorbed into the data:

                norms[i] = ‖W[i, :]‖
                W_new[i, :] = W[i, :] / norms[i]
                R_new[..., i] = R[..., i] * norms[i]

        Has no effect on Abelian tensors or tensors without an intertwiner.
        """
        if self.intw is None:
            return

        group = self.indices[0].group
        _sp_eps = torch.finfo(torch.float32).eps

        if len(self.indices) == 2:
            # 2nd order (matrix): normalise each weight to sqrt(irrep_dim(q))
            for key, arr in self.data.items():
                bridge = self.intw.get(key)
                if bridge is None:
                    continue

                q = key[0]
                target = torch.sqrt(
                    torch.tensor(group.irrep_dim(q), dtype=self.dtype, device=self.device)
                )
                factors = bridge.weights[:, 0] / target         # (k,) signed ratio
                bridge.weights[:] = target                      # all rows → +target
                self.data[key] = arr * factors                  # (..., k) * (k,)
        else:
            # Higher-order: row-normalise Bridge weight matrix.
            # norms[i] = ‖W[i,:]‖; absorbed into the trailing component axis of R.
            for key, arr in self.data.items():
                bridge = self.intw.get(key)
                if bridge is None:
                    continue

                norms = bridge.weights.norm(dim=1)          # (k,)
                safe_norms = norms.clamp(min=1e-12)
                bridge.weights[:] = bridge.weights / safe_norms[:, None]
                self.data[key] = arr * norms                # (..., k) * (k,)

    # ------------------------------------------------------------
    #   Tensor operations: conj, permute, transpose
    # ------------------------------------------------------------

    def conj(self, in_place: bool = False) -> Tensor:
        """Complex conjugate every dense block if dtype is complex, and revert all index directions.
        
        Parameters
        ----------
        in_place : bool, optional
            If True, modifies this tensor in-place and returns self.
            If False (default), returns a new Tensor instance with conjugated data
            (as views for complex dtype) and flipped directions. The underlying torch
            tensors are not cloned - torch.conj() returns a view for complex dtypes,
            and real dtypes share the same tensors.
        
        Returns
        -------
        Tensor
            Self if in_place=True, new Tensor instance if in_place=False.
        
        Examples
        --------
        >>> # Functional style (default, efficient with sharing)
        >>> t2 = t1.conj()
        >>> t2 is not t1  # Different Tensor instances
        >>> # But for complex dtype, t2.data shares storage with t1.data (as conjugate views)
        >>> 
        >>> # In-place style (allows chaining)
        >>> result = t1.conj(in_place=True)
        >>> result is t1  # Returns self for chaining
        """
        # Prepare conjugated data
        if self.dtype.is_complex:
            new_data = {k: torch.conj(v) for k, v in self.data.items()}
        else:
            new_data = dict(self.data)  # Shallow copy: share tensors
        
        # Flip all index directions
        new_indices = tuple(idx.flip() for idx in self.indices)
        
        # Update intw with flipped directions
        new_intw: Optional[Dict[BlockKey, dg.Bridge]] = None
        if self.intw is not None:
            new_intw: Dict[BlockKey, dg.Bridge] = {}
            for key, bridge in self.intw.items():
                new_intw[key] = bridge.conj()
        
        if in_place:
            # Modify in-place and return self for chaining
            self.data = new_data
            self.indices = new_indices
            self.intw = new_intw
            return self
        else:
            # Return new instance
            return Tensor(
                indices=new_indices, itags=self.itags, data=new_data, intw=new_intw,
                dtype=self.dtype, label=self.label
            )

    def permute(self, order: Sequence[int], in_place: bool = False) -> Tensor:
        """Permute tensor axes according to the provided reordering.
        
        Parameters
        ----------
        order : Sequence[int]
            Sequence of integer axes specifying the new ordering. Must be a
            permutation of range(len(self.indices)).
        in_place : bool, optional
            If False (default), returns a new Tensor instance with permuted axes.
            The data blocks share the same underlying storage (torch.permute
            creates views). If True, modifies this tensor in-place and returns
            self.
        
        Returns
        -------
        Tensor
            Self if in_place=True, new Tensor instance if in_place=False.
        
        Notes
        -----
        For non-Abelian (SU2) tensors, permutation involves R-symbols that transform
        the outer multiplicity (OM) indices. The weights are updated by matrix
        multiplication with the R-symbol: new_weights = old_weights @ R.
        
        Examples
        --------
        >>> # Functional style (default, efficient with sharing)
        >>> t2 = t.permute([2, 0, 1])
        >>> t2 is not t  # Different Tensor instances
        >>> # But t2.data blocks share storage with t.data (as permuted views)
        >>> 
        >>> # In-place style (allows chaining)
        >>> result = t.permute([2, 0, 1], in_place=True)
        >>> result is t  # Returns self for chaining
        """
        if sorted(order) != list(range(len(self.indices))):
            raise ValueError("Invalid permutation order")
        
        # Create new indices and itags
        new_indices = tuple(self.indices[i] for i in order)
        new_itags = tuple(self.itags[i] for i in order)
        
        # Permute data blocks (always use trailing OM axis if present)
        new_data: Dict[BlockKey, torch.Tensor] = {}
        if self.intw is not None:
            # Non-Abelian: data has trailing OM axis, permute all but last
            order_with_om = tuple(order) + (len(order),)
            for key, arr in self.data.items():
                new_key = tuple(key[i] for i in order)
                new_data[new_key] = torch.permute(arr, order_with_om)
        else:
            # Abelian: standard permutation
            for key, arr in self.data.items():
                new_key = tuple(key[i] for i in order)
                new_data[new_key] = torch.permute(arr, order)
        
        # Update intw with R-symbols for non-Abelian case
        new_intw: Optional[Dict[BlockKey, dg.Bridge]] = None
        if self.intw is not None:
            new_intw: Dict[BlockKey, dg.Bridge] = {}
            for key, bridge in self.intw.items():
                # Compute R-symbol for this permutation
                r_symbol, spec_permuted = dg.compute_rsymbol(bridge, order)
                
                # Update weights: new_weights = old_weights @ R
                # R has shape (om_original, om_permuted)
                # weights has shape (num_components, om_original)
                # Result: (num_components, om_permuted)
                new_weights = bridge.weights @ r_symbol
                
                # Create new Bridge with permuted spec and updated weights
                new_key = tuple(key[i] for i in order)
                new_intw[new_key] = dg.Bridge(cgspec=spec_permuted, weights=new_weights)
        
        if in_place:
            # Modify in-place and return self for chaining
            self.indices = new_indices
            self.itags = new_itags
            self.data = new_data
            self.intw = new_intw
            self._invalidate_sorted_keys()
            return self
        else:
            # Return new instance
            return Tensor(
                indices=new_indices, itags=new_itags, data=new_data, intw=new_intw,
                dtype=self.dtype, label=self.label
            )

    def transpose(self, in_place: bool = False) -> Tensor:
        """Transpose tensor axes by reversing the index order.
        
        Parameters
        ----------
        in_place : bool, optional
            If False (default), returns a new Tensor instance with reversed axes.
            If True, modifies this tensor in-place and returns self.
        
        Returns
        -------
        Tensor
            New Tensor instance if in_place=False, self if in_place=True.
        """
        return self.permute(tuple(reversed(range(len(self.indices)))), in_place=in_place)

    def invert(self, positions: Union[int, Sequence[int]]) -> None:
        """Invert the direction of specified index/indices while maintaining charge conservation.
        
        This operation inverts the direction(s) and conjugates the charge(s)
        using Index.dual(), effectively inverting the tensor's index structure
        at the specified positions.
        
        Parameters
        ----------
        positions:
            Index position(s) to invert. Can be a single int or a sequence of ints.
            Positions are 0-indexed.

        Warnings
        --------
        Use with extreme caution! This method is supposed to work in isolation.
        For inverting a bond between two tensors, use `capcup` instead, which
        applies the necessary Frobenius–Schur phase for SU(2).
        
        Notes
        -----
        This operation uses Index.dual() to invert both the direction and conjugate
        the charges, ensuring charge conservation is maintained. Both the index
        metadata and the block keys are updated to reflect the conjugated charges.
        The tensor data arrays themselves remain unchanged.
        
        This differs from Index.flip() which only reverses direction without
        conjugating charges. The tensor invert operation performs a complete
        inversion of the index structure (direction + charge conjugation).
        
        For non-Abelian groups (e.g. SU(2)), the intertwiner (Bridge) at each
        affected block has its edge directions inverted at the corresponding
        positions without any additional phase factor, ensuring that two
        successive calls to invert() with the same positions restore the
        original tensor exactly.
        
        Examples
        --------
        >>> # Invert a single index at position 0
        >>> tensor.invert(0)
        >>> 
        >>> # Invert multiple indices at positions 0 and 2
        >>> tensor.invert([0, 2])
        """
        # Normalize to a sequence
        if isinstance(positions, int):
            positions = [positions]
        
        # Validate positions
        n = len(self.indices)
        for pos in positions:
            if pos < 0 or pos >= n:
                raise IndexError(f"Index position {pos} out of range [0, {n})")
        
        # Get the symmetry group
        if n == 0:
            return  # Scalar tensor, nothing to invert
        group = self.indices[0].group
        
        # Create new indices with dual (inverted direction + conjugated charges)
        # at the specified positions
        indices_list = list(self.indices)
        for pos in positions:
            indices_list[pos] = indices_list[pos].dual()
        self.indices = tuple(indices_list)
        
        # Update block keys: conjugate charges at inverted positions
        new_data: Dict[BlockKey, torch.Tensor] = {}
        for key, arr in self.data.items():
            key_list = list(key)
            for pos in positions:
                key_list[pos] = group.dual(key_list[pos])
            new_key = tuple(key_list)
            new_data[new_key] = arr
        self.data = new_data
        
        # Update intw: invert edge directions at specified positions, update keys
        if self.intw is not None:
            new_intw: Dict[BlockKey, dg.Bridge] = {}
            for key, bridge in self.intw.items():
                key_list = list(key)
                for pos in positions:
                    key_list[pos] = group.dual(key_list[pos])
                new_key = tuple(key_list)
                new_intw[new_key] = bridge.invert_edges(positions)
            self.intw = new_intw
        
        self._invalidate_sorted_keys()

    # ------------------------------------------------------------
    #   itag manipulations: multiple modes of retagging
    # ------------------------------------------------------------

    def retag(self, mapping_or_axes: Union[Mapping[str, str], Sequence[str], int, Sequence[int]], 
              new_tags: Optional[Union[str, Sequence[str]]] = None) -> None:
        """Retag indices using one of three modes.
        
        Parameters
        ----------
        mapping_or_axes:
            Can be one of:
            - Mapping[str, str]: Dictionary mapping old tags to new tags
            - Sequence[str]: Complete list of new tags (must match number of indices)
            - Sequence[int] or int: Index position(s) to update (requires new_tags)
        new_tags:
            New tag(s) to use when mapping_or_axes is an integer or sequence of integers.
            Can be a single string or sequence of strings. Must match the length of mapping_or_axes.
        
        Examples
        --------
        >>> # Mode 1: Mapping (update specific tags by name)
        >>> tensor.retag({"a": "left", "b": "right"})
        >>> 
        >>> # Mode 2: Full replacement (replace all tags)
        >>> tensor.retag(["left", "middle", "right"])
        >>> 
        >>> # Mode 3: Selective update by position
        >>> tensor.retag([0, 2], ["left", "right"])
        >>> tensor.retag(0, "left")  # Single index and tag
        """
        # Parse and normalize input arguments depending on the mode
        if isinstance(mapping_or_axes, Mapping):
            # Mode 1: Mapping dictionary
            self.itags = tuple(mapping_or_axes.get(tag, tag) for tag in self.itags)
        elif new_tags is not None:
            # Mode 3: Update specific indices
            # Convert single int to list
            if isinstance(mapping_or_axes, int):
                axes = [mapping_or_axes]
            elif isinstance(mapping_or_axes, Sequence):
                axes = list(mapping_or_axes)
            else: # mapping_or_axes is not a sequence of integers
                raise TypeError("When new_tags is provided, first argument must be an integer "
                    "or sequence of integers")
            
            # Convert single str to list
            if isinstance(new_tags, str):
                tags = [new_tags]
            else:
                tags = list(new_tags)
            
            if len(axes) != len(tags):
                raise ValueError("Number of axes must match number of new tags")
            if not all(isinstance(i, int) for i in axes):
                raise TypeError("Index positions (axes) must be integers")
            if any(i < 0 or i >= len(self.itags) for i in axes):
                raise IndexError("Index position (axis) out of range")
            
            # Convert to list for mutation, then back to tuple
            new_itags = list(self.itags)
            for idx, tag in zip(axes, tags):
                new_itags[idx] = tag
            self.itags = tuple(new_itags)
        else:
            # Mode 2: Full replacement
            if not isinstance(mapping_or_axes, Sequence):
                raise TypeError("Expected a sequence of strings for full replacement")
            if len(mapping_or_axes) != len(self.itags):
                raise ValueError(f"Number of new tags ({len(mapping_or_axes)}) must match number of indices ({len(self.itags)})")
            self.itags = tuple(mapping_or_axes)
