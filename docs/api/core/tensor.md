# Tensor

Block-sparse tensor with symmetry-aware indices.

::: nicole.Tensor
    options:
      show_source: false
      heading_level: 2
      members:
        - __init__
        - zeros
        - random
        - norm
        - copy
        - rand_fill
        - insert_index
        - sorted_keys
        - key
        - block
        - show
        - conj
        - permute
        - transpose
        - retag

## Description

The `Tensor` class is the core data structure in Nicole, representing block-sparse tensors backed by symmetry-aware indices. Each tensor stores a collection of dense NumPy blocks, where each block corresponds to a specific combination of charges that satisfies charge conservation rules.

### Key Features

- **Block-sparse storage**: Only admissible blocks are stored
- **Automatic charge conservation**: Selection rules enforced by structure
- **NumPy-backed blocks**: Dense operations within each symmetry sector
- **Immutable indices**: Index structure fixed at creation

## See Also

- [Index](index-class.md): Define tensor leg structure
- [zeros](../creation/zeros.md): Create zero tensor
- [random](../creation/random.md): Create random tensor
- [Examples: Creating Tensors](../../examples/basic/first-tensor.md)
- [Examples: Arithmetic](../../examples/basic/arithmetic.md)

## Notes

Tensors are mutable objects. Use `copy()` when independence is needed. For functional (non-mutating) operations, see the [operators](../manipulation/conjugate.md) module.

Charge conservation is enforced: `∑(OUT charges) - ∑(IN charges) = neutral element`.
