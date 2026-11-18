# filter_blocks

Extract a chosen subset of a tensor's blocks as a new tensor, pruning the sectors the survivors no longer use. Blocks are addressed by their 1-based display number, the same numbering `print(tensor)` shows, so the selection is made in the terms the output already presents. The usual purpose is to select an appropriate subspace: keep only the charge sectors that matter for the calculation, and drop the rest.

::: nicole.filter_blocks
    options:
      show_source: false
      heading_level: 2

## Description

Each requested block is cloned into the new tensor, leaving the result independent of the original. The itags, dtype, and label carry over, and for an SU(2) tensor the `Bridge` of every kept block comes along with it. A number outside the range of existing blocks raises `IndexError`, and `block_indices` may be a single integer as readily as a sequence.

The indices are then pruned: a sector whose charge appears in none of the kept blocks is dropped. The tensor handed back therefore carries narrower indices than its parent, keeping its structure minimal and consistent with the blocks it actually holds. Block numbering follows the sorted charge keys of whatever blocks are present, so the numbers on the result will generally differ from the numbers used to select them.

## See Also

- [Tensor](../core/tensor.md): `key` and `block` for inspecting a block before selecting it
- [Display Utilities](../utilities/display.md): The numbering that `block_indices` refers to
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

- `Tensor.key(i)` returns the charge key of block `i` and `Tensor.block(i)` its data, both 1-based like the display. Use them to confirm what a number refers to before filtering on it.
- Pruning means the result is not merely a sub-dictionary of blocks. Its indices describe a smaller space, so it may not contract against a partner built for the parent's sectors.
- To drop blocks by content instead of by number, `Tensor.trim_zero_blocks()` removes those that are numerically zero.
