# transpose

Reverse the axis order of a tensor, returning a fully independent copy. It is [permute](permute.md) with the order fixed to `(n-1, …, 1, 0)`, supplied for the common case of reflecting a matrix-like tensor end to end. This is the functional form; `Tensor.transpose()` is the cheaper method form.

::: nicole.transpose
    options:
      show_source: false
      heading_level: 2

## Description

The semantics are those of [permute](permute.md): indices, itags, and charge keys move together, each dense block is transposed, and for a non-Abelian tensor the intertwiner weights are rotated by the R-symbols of the move. The one difference is that the order is derived from the tensor's order instead of being supplied, so the call takes no argument.

For a 2nd-order tensor this is the ordinary matrix transpose applied block by block. It does not conjugate the data and leaves every index direction as it stands, so on its own it is not the adjoint. Compose it with [conj](conjugate.md) when the adjoint is what you need.

## Copies and Views

The functional form clones every data block, so the tensor it returns is isolated from the input and survives any later mutation of it. The method form `Tensor.transpose()` instead hands back a new tensor whose blocks are views sharing storage with the original, which is cheaper and the better default. Pass `Tensor.transpose(in_place=True)` to reverse the axes of the tensor itself and receive it back for chaining.

## See Also

- [Tensor.transpose](../core/tensor.md): Method form, sharing storage by default or acting in place
- [permute](permute.md): General axis reordering
- [conj](conjugate.md): Conjugate the data and flip directions
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

- Transposing twice returns the original axis order, whatever the tensor's order.
- For anything other than a full reversal, use [permute](permute.md) with an explicit order.
