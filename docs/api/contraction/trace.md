# trace

Sum a tensor over one or more of its own index pairs, dropping two axes for each pair traced. A pair qualifies on the same terms as a contraction pair, sharing an itag and carrying opposite directions, which makes `trace` the self-contraction counterpart of [contract](contract.md). Trace over every pair and a scalar comes back.

::: nicole.contract.trace
    options:
      show_source: false
      heading_level: 2

## Description

Each traced pair must share an itag, carry opposite directions, and agree in charge and dimension within every symmetry block. The sum then runs over the matching diagonal entries, block by block, so \(\mathrm{Tr}(A) = \sum_i A_{ii}\) holds within each charge sector. The axes that remain keep their original relative order.

Several pairs are traced one after another instead of all at once, so `trace(T, [(a, b), (c, d)])` is exactly `trace(trace(T, (a, b)), (c, d))`. The outcome therefore does not depend on how the pairs are grouped, nor on the order they are listed in.

## Selecting the Pairs

The three modes mirror those of [contract](contract.md), and `axes` and `excl` are mutually exclusive.

**Automatic**, with no argument. Every pair of indices sharing an itag with opposing directions is traced.

**Explicit**, through `axes`. One pair reads `(axis_a, axis_b)` and several read `[(a1, b1), (a2, b2)]`. Integer positions only here, not itags.

**Exclusion**, through `excl`. Tracing proceeds automatically but skips what is listed, given as a single position or itag, or a sequence of either. Unlike `axes`, this argument does accept itags.

## See Also

- [contract](contract.md): General contraction between two tensors
- [einsum](einsum.md): Trace and contract together in one subscript string
- [Examples: Contraction](../../examples/operations/contraction-examples.md)

## Notes

- Automatic mode raises when the pairing is ambiguous, so a tag carried by more than two axes needs `axes` or `excl`.
- Tracing a 2nd-order tensor over its single pair gives a scalar tensor; read the number out with `Tensor.item()`.
- A trace that would leave exactly one free axis raises `ValueError`, since a 1st-order tensor cannot enforce charge conservation. Tracing a 3rd-order tensor down to one axis is therefore refused, while 0 or 2 or more remaining axes are fine.
- A pair whose dimensions disagree within a block raises `ValueError`, as the diagonal it would sum over is not square.
