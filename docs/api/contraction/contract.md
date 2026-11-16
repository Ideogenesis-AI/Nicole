# contract

Contract two tensors over one or more index pairs, computing only the blocks that charge conservation admits. It generalizes matrix multiplication to tensors of any order, and in the default mode, it works out the pairs itself by matching itags with opposing directions. The output carries the free axes of `A` followed by those of `B`.

::: nicole.contract.contract
    options:
      show_source: false
      heading_level: 2

## Description

A pair is contractible when its two indices share an itag and carry opposite directions. On a contracted axis the charges must agree for a block pair to contribute, and the block that comes out must still conserve the total charge. Only those blocks are computed, which is where the saving over a dense contraction comes from: the forbidden combinations are never visited.

The output index order is fixed. The axes of `A` that were not contracted come first, in their original order, followed by those of `B`. Pass `perm` to reorder them within the same call, which saves a separate [permute](../manipulation/permute.md) afterwards.

## Selecting the Pairs

Three modes are available, and `axes` and `excl` are mutually exclusive: supplying both raises `ValueError`.

**Automatic**, with neither argument given. Every pair of indices sharing an itag with opposing directions is contracted. This is the usual mode, and the reason itags are worth setting deliberately.

**Explicit**, through `axes`. One pair reads `(axis_in_A, axis_in_B)` and several read `([axes_in_A], [axes_in_B])`, following the `np.tensordot` convention. The itags and directions are still validated, so a pair that is not contractible raises before proceeding.

**Exclusion**, through `excl`. Pairing proceeds automatically but skips the axes listed as `([excl_in_A], [excl_in_B])`. This is the mode for two tensors that share a tag on an axis which has to stay free.

## See Also

- [einsum](einsum.md): Contraction patterns named by subscript
- [trace](trace.md): Trace over index pairs within a tensor
- [capcup](../manipulation/capcup.md): Reverse a bond's direction before contracting
- [Examples: Contraction](../../examples/operations/contraction-examples.md)

## Notes

- Automatic pairing raises when it cannot resolve a tag unambiguously, so a tag repeated across several contractible axes needs `axes` or `excl` to disambiguate.
- A contraction yielding no free axes gives a scalar tensor, whose value comes out through `Tensor.item()`.
- For a pattern that is awkward to express as pairs, such as one that also permutes or traces, [einsum](einsum.md) takes the whole thing as a subscript string.
