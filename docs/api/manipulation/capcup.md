# capcup

Invert the directions of a to-be-contracted bond between two tensors, leaving every subsequent contraction numerically unchanged. The operation is the insertion of a cap-cup metric on the bond: both ends reverse direction so the pair still contracts, and for an SU(2) tensor the Frobenius–Schur phase that the reversal would otherwise leave behind is absorbed into `B`. Reach for it whenever a bond has to flip orientation, never for double `Tensor.invert()`.

::: nicole.capcup
    options:
      show_source: false
      heading_level: 2

## Description

A to-be-contracted pair is a bond where the two tensors carry the same itag with opposite directions. `capcup` inverts both ends of that pair. For an Abelian group the directions alone change. For SU(2) the Frobenius–Schur phase \((-1)^{2j}\) is additionally multiplied into each block of `B` at the bond position, so that any follow-up contractions involving the bond still yields the same number after the reversal.

The phase lands in the intertwiner weights of `B` (shape `(num_components, om_dimension)`), which is much smaller than the full data blocks. It is applied only when both bond axes sit at the same type in their respective CG fusion trees — both leading or both terminal — because only then does the combined X-symbol transformation require a non-trivial correction.

!!! warning "Use `capcup` for bond inversion, not `Tensor.invert()`"
    Whenever you need to invert a bond — a pair of compatible indices eligible for contraction — always use `capcup`. `Tensor.invert()` acts on a single tensor in isolation and is unaware of the paired index on the other tensor; it cannot apply the necessary corrections consistently across both sides. `Tensor.invert()` should only be used alone, on a free index, and with extreme caution.

## See Also

- [conj](conjugate.md): Conjugate all indices of a tensor
- [Yuzuha Protocol](../../getting-started/yuzuha-protocol.md): SU(2) intertwiner and FS phase background
- [Examples: Manipulation](../../examples/operations/manipulation-examples.md)

## Notes

- Both `axis_a` and `axis_b` must refer to the shared bond: same itag and opposite directions. A mismatch raises `ValueError`.
- `capcup` modifies both `A` and `B` in place and returns `None`.
- `axis_a` and `axis_b` are integer positions. Tags identify the bond; positions select which occurrence of that tag each tensor contributes.
