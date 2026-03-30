# capcup

Invert a bond direction between two tensors with phase corrections.

::: nicole.capcup
    options:
      show_source: false
      heading_level: 2

## Description

`capcup` inverts the directions of a contraction pair (bond) between two tensors — equivalent to inserting a cap-cup metric on the bond. For SU(2) tensors, it also applies the **Frobenius–Schur (FS) phase** \((-1)^{2j}\) to each block of `B` at the relevant bond position, keeping all subsequent contractions numerically correct after the direction reversal. For Abelian groups, only the directions are flipped and no phase is applied.

The FS phase is absorbed into the intertwiner weights of `B` (shape `(num_components, om_dimension)`), which is much smaller than the full data blocks.

!!! warning "Use `capcup` for bond inversion, not `Tensor.invert()`"
    Whenever you need to invert a bond — a pair of compatible indices eligible for contraction — always use `capcup`. `Tensor.invert()` acts on a single tensor in isolation and is unaware of the paired index on the other tensor; it cannot apply the necessary corrections consistently across both sides. `Tensor.invert()` should only be used alone, on a free index, and with extreme caution.

## See Also

- [conj](conjugate.md): Conjugate all indices of a tensor
- [Yuzuha Protocol](../../getting-started/yuzuha-protocol.md): SU(2) intertwiner and FS phase background
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

- Both `axis_a` and `axis_b` must refer to the shared bond: same itag and opposite directions.
- The FS phase \((-1)^{2j}\) is applied only when both bond axes are of the same type (both leading or both terminal) in their respective CG fusion trees.
- `capcup` modifies both `A` and `B` in place and returns `None`.
