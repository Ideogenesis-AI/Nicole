# conj

Return the conjugate of a tensor, with complex-conjugated blocks and every index direction flipped. The two effects together let the conjugate contract against the original, making `conj` the building block of norms, overlaps, and adjoints. This is the functional form; `Tensor.conj()` is the cheaper method form.

::: nicole.conj
    options:
      show_source: false
      heading_level: 2

## Description

Under a complex dtype every dense block is conjugated elementwise. Under a real dtype the data passes through untouched and only the directions change, so a real tensor and its conjugate hold the same numbers with opposite orientation. Flipping reverses direction while leaving charges alone, so the charge keys are the same on both tensors. The operation that conjugates the charges as well is [Tensor.invert()](../core/tensor.md), which is a different operation and needs more care.

For a non-Abelian tensor, each block's `Bridge` is conjugated in step: its Clebsch–Gordan edge directions flip, and the cumulated Frobenius–Schur phase of ±1 is multiplied into the weights. The weight values themselves are not conjugated, as they are real at any dtype — each descends from real arithmetic in `yuzuha`, so a complex tensor's weights carry a complex dtype with a zero imaginary part.

## Copies and Views

The functional form clones every data block, so the tensor it returns is isolated from the input and survives any later mutation of it. The method form `Tensor.conj()` instead hands back a new tensor that shares storage with the original, as `torch.conj` returns a view for a complex dtype and a real dtype shares its blocks outright. Pass `Tensor.conj(in_place=True)` to conjugate the tensor itself and receive it back for chaining.

## See Also

- [Tensor.conj](../core/tensor.md): Method form, sharing storage by default or acting in place
- [transpose](transpose.md): Reverse axis order, the other half of an adjoint
- [capcup](capcup.md): Invert a bond between two tensors, with the SU(2) phase correction
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

- The itags, dtype, and label all carry over unchanged, so a conjugate pairs tag for tag with its original under automatic contraction.
- To invert the direction of a bond shared by two tensors, use [capcup](capcup.md). It applies the Frobenius–Schur phase to both sides consistently, which `conj` on one tensor cannot do.
