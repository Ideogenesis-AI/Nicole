# einsum

Evaluate an Einstein summation equation on symmetry-aware tensors.

::: nicole.einsum.einsum
    options:
      show_source: false
      heading_level: 2

## Description

`einsum` parses a subscript equation string and dispatches to
[`contract`](contract.md), [`trace`](trace.md), and
[`permute`](../manipulation/permute.md) to carry out the requested operation.
Three core equation types are supported — permutation, trace, and contraction
— and they can be freely combined. Some representative patterns:

1. **Permutation** — a single input tensor whose output subscript is a
   reordering of the input subscript:

    ```python
    einsum('ij->ji', A)           # transpose
    einsum('abcd->dcba', A)       # reverse all axes
    ```

2. **Trace** — a repeated letter within one input subscript causes those two
   axes to be traced out. Remaining axes are kept in the order given by the
   output subscript:

    ```python
    einsum('ii->', A)             # full trace → scalar
    einsum('iijk->jk', A)         # partial trace, axes 0 and 1
    einsum('abiicd->abcd', A)     # trace middle axes 2 and 3
    ```

3. **Contraction** — a letter shared between two input subscripts but absent
   from the output subscript is summed over at that step. Tensors are
   contracted strictly from left to right:

    ```python
    einsum('ij,jk->ik', A, B)         # matrix multiply
    einsum('ij,jk->ki', A, B)         # matrix multiply then transpose
    einsum('ij,ji->', A, B)           # full contraction to scalar
    einsum('ijk,kjl->il', A, B)       # multi-index, non-trivial axis order
    einsum('ij,jk,kl->il', A, B, C)  # chain contraction
    ```

4. **Outer product** — no shared letters between two inputs means no summation;
   the result carries all axes from both tensors:

    ```python
    einsum('ij,kl->ijkl', A, B)       # outer product, natural order
    einsum('ij,kl->klij', A, B)       # outer product with output permutation
    ```

5. **Mixed trace and contraction** — within-tensor repeated letters are traced
   first; the result then participates in the contraction chain as normal:

    ```python
    einsum('iijk,kl->jl', A, B)       # trace A on axes 0,1 then contract on k
    einsum('ijk,kjl,lmn->imn', A, B, C)  # non-trivial chain with trace step
    ```

## Equation syntax

An equation has the form `'<lhs>-><rhs>'` where:

- `<lhs>` is a comma-separated list of subscript strings, one per input tensor.
- `<rhs>` is the output subscript.
- The `->` separator is **required**; implicit output subscripts are not
  supported.
- Each subscript character must be a single ASCII letter.
- A letter may appear at most **twice** within a single input subscript
  (forming one trace pair).  Three or more occurrences raise `ValueError`.

## Contraction order

**No contraction-order optimisation is performed.** For multi-tensor equations
the tensors are contracted strictly from left to right. This may be
suboptimal for networks where a different pairing would reduce intermediate
tensor sizes. If performance matters, determine the optimal contraction order
manually (e.g. by estimating intermediate tensor sizes or using a dedicated
contraction-order tool), then rearrange the tensors and equation string
accordingly.

## Unsupported: Hadamard (batch) indices

A letter that appears in two or more input subscripts **and** in the output
subscript is a Hadamard (batch) index.  In NumPy/PyTorch einsum this means
element-wise multiplication along that axis with no summation.
[`contract`](contract.md) has no such mode, so equations of this kind are not
supported (e.g. `'ij,ij->ij'`).

## See Also

- [contract](contract.md): General contraction between two tensors
- [trace](trace.md): Trace over index pairs within a tensor
- [permute](../manipulation/permute.md): Permute tensor axes
- [Examples: Contraction](../../examples/operations/contraction-examples.md)
