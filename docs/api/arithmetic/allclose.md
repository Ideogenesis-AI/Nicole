# allclose

Compare two tensors numerically within tolerances and return a `bool`. Where `==` demands exact agreement of values and dtype, `allclose` asks only that the numbers match to `rtol` and `atol`, which makes it the check to use after any computation that accumulates rounding error. For a generic symmetry group it compares physical content, so it is blind to the gauge freedom of the reduced representation.

::: nicole.maneuver.allclose
    options:
      show_source: false
      heading_level: 2

## Description

Both arguments must be tensors, or `TypeError` follows, and they must share an order and compatible indices, or `ValueError` does. Differing block keys are not an error: they simply make the two tensors unequal, and the call returns `False`. The same holds when one tensor carries intertwiners and the other does not.

For an Abelian tensor each dense block is handed to `torch.allclose`. For a generic tensor the comparison is made on the physical tensor `R @ W` instead of the stored reduced block only. That makes the outcome gauge-invariant: two tensors holding the same physical content in different `(R, W)` factorizations still compare as equal, where a naive block-by-block check on `R` alone would call them different.

## See Also

- [Basic Operations](addition.md): `+`, `-`, `*`, `/`
- [Tensor](../core/tensor.md): `Tensor.__eq__` for exact equality
- [Bridge](../symmetry/bridge.md): SU(2) intertwiner container

## Notes

- The defaults are `rtol=1e-5` and `atol=1e-8`, passed through to `torch.allclose` unchanged. Tighten them when comparing quantities that should agree to machine precision.
- Structural mismatches raise instead of returning `False`, so a failed comparison between tensors of different order is a bug report, not an answer.
- `==` compares dtype as well as values, so the same numbers at two dtypes are unequal under `==` and equal under `allclose`.
