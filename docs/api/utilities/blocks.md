# Block Utilities

Low-level utilities for managing tensor block structure.

## Description

The `blocks` module provides internal utilities for reasoning about tensor block dictionaries. Most users won't need these directly.

## Key Concepts

### BlockKey

A `BlockKey` is a tuple of charges, one per tensor index:

```python
from nicole.blocks import BlockKey

# For a 2-index U(1) tensor
key1 = (0, 0)      # Both indices have charge 0
key2 = (1, 1)      # Both indices have charge 1
```

### BlockSchema

Static utility class for block operations:

- `iter_admissible_keys()`: Generate all possible charge combinations
- `shape_for_key()`: Get block shape from key
- `validate_blocks()`: Validate block data
- `charges_conserved()`: Check conservation for a key

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

These are internal utilities. Most functionality is accessed through the `Tensor` class interface.
