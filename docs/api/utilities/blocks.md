# Block Utilities

Low-level utilities for managing tensor block structure. A symmetric tensor is a dictionary from charge keys to dense arrays, and this module holds the routines that decide which keys are legal, the shape of the array behind each one, and how two arrays sharing a key combine. `Tensor` leans on them throughout, so this page is less a set of calls to make than a look at what the tensor class is doing on your behalf.

## Description

The `blocks` module provides internal utilities for reasoning about tensor block dictionaries. Most users won't need these directly. The module is the layer where symmetry stops being a property of the indices and starts being a property of the stored data: an `Index` knows its charges and directions, a `Tensor` knows which blocks it holds, and `BlockSchema` translates between the two. Every routine here takes the indices as an argument, so the same helpers serve tensor construction, validation, arithmetic and compression alike.

## Key Concepts

### BlockKey

A `BlockKey` is a tuple of charges, one per tensor index:

```python
from nicole.blocks import BlockKey

# For a 2-index U(1) tensor
key1 = (0, 0)      # Both indices have charge 0
key2 = (1, 1)      # Both indices have charge 1
```

It is a type alias for `Tuple[Charge, ...]`, not a class, so any tuple of the right length serves as a key and the charge type follows the group — an integer for U(1) and Z(2), a tuple for a `ProductGroup`. Position in the key matches position of the index, so a key reads as the coordinates of a block.

### BlockSchema

Static utility class for block operations:

- `iter_admissible_keys()`: Generate all possible charge combinations
- `shape_for_key()`: Get block shape from key
- `validate_blocks()`: Validate block data
- `charges_conserved()`: Check conservation for a key
- `block_add()`: Merging or concatenating weights from a pair of blocks
- `block_compress()`: Remove linearly dependent components from a block

Every method is a `staticmethod`, so the class is a namespace and never an object to construct: the indices are passed in on each call and nothing about a tensor is cached between them. Two details are worth knowing before calling any of them:

1. `iter_admissible_keys` yields the full cartesian product of the charges available on each index, candidates instead of survivors, so charge conservation has to be imposed afterwards with `charges_conserved`.
2. Both `charges_conserved` and the charge totals behind it are direction-aware: an OUT index contributes its charge as it stands and an IN index contributes the dual, and admissibility is the test that those contributions fuse to the neutral charge.

For a generic symmetry the dense array behind a key carries one extra trailing axis for the outer multiplicity components, and `shape_for_key` appends it when given `num_components`. This is also where `block_add` earns its complexity: when two blocks share a key, their intertwiner weights are collinear often enough to simply scale and add, and when they are not, the data and the weights are concatenated along that trailing axis instead. `block_compress` is the inverse pressure, dropping components that an SVD of the weight matrix finds linearly dependent.

## Usage

```python
from nicole.blocks import BlockSchema

# Check if block conserves charge
conserved = BlockSchema.charges_conserved(indices, block_key)
```

## See Also

- [Tensor](../core/tensor.md): Main tensor class
- [Index](../core/index-class.md): Index structure

## Notes

- These are internal utilities. Most functionality is accessed through the `Tensor` class interface.
- `shape_for_key` raises `ValueError` when the key length disagrees with the number of indices and `KeyError` when a charge is absent from the index at that position, so a mistyped key fails at the schema without producing an erroneously shaped block.
