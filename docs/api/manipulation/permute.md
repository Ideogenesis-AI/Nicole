# permute

Reorder the axes of a tensor, returning a fully independent copy. The reordering applies to the indices, the itags, and every data block at once, so the content is unchanged and only the axis order differs. This is the functional form; `Tensor.permute()` is the cheaper method form.

::: nicole.permute
    options:
      show_source: false
      heading_level: 2

## Description

`order` must be a permutation of `range(len(tensor.indices))`, and anything else raises `ValueError`. An index travels with its itag, so an axis keeps its direction, its symmetry sectors, and its tag as it moves. The charge keys are rewritten alongside, leaving each key still listing its charges in axis order, and each dense block is permuted along to match.

For a non-Abelian tensor a permutation is more than a relabeling. Reordering the external edges changes the fusion tree, so the outer multiplicity basis has to be rotated: the R-symbols of the move are applied to the intertwiner weights as `new_weights = old_weights @ R`, while the reduced data block undergoes the plain permutation.

## Copies and Views

The functional form clones every data block, so the tensor it returns is isolated from the input and survives any later mutation of it. The method form `Tensor.permute(order)` instead hands back a new tensor whose blocks are views sharing storage with the original, which is cheaper and the better default. Pass `Tensor.permute(order, in_place=True)` to reorder the tensor itself and receive it back for chaining.

## See Also

- [Tensor.permute](../core/tensor.md): Method form, sharing storage by default or acting in place
- [transpose](transpose.md): Reverse axis order
- [merge_axes](merge_axes.md): Combine axes instead of reordering them
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

- Axes may be addressed by itag elsewhere in the API, but `permute` takes integer positions only.
- Permuting by an order and then by its inverse restores the original axis order exactly, including the intertwiner weights of a non-Abelian tensor.
- [transpose](transpose.md) is this function with the order fixed to the reversal of the axes.
