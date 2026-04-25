# allclose

::: nicole.maneuver.allclose
    options:
      show_source: false
      heading_level: 2

## Notes

- For Abelian tensors, each dense block is compared directly with `torch.allclose`.
- For non-Abelian (SU(2)) tensors, the physical tensor `R @ W` is compared
  block-by-block, making the check invariant to the gauge freedom in the reduced
  representation. Two tensors that represent the same physical content but differ
  in their internal `(R, W)` factorization will still compare as equal.
- Structural mismatches (different tensor order, incompatible groups or directions)
  raise `ValueError`. Differing block keys return `False` without raising.

## See Also

- [Basic Operations](addition.md): `+`, `-`, `*`, `/`
- [Tensor](../core/tensor.md): `Tensor.__eq__` for exact equality
- [Bridge](../symmetry/bridge.md): SU(2) intertwiner container
