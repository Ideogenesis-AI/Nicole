# Tensor-Train States and Operators

Matrix-product state (MPS) and matrix-product operator (MPO) containers built
from Nicole cores, together with their constructors.

## TensorTrain

::: nicole.bug.TensorTrain
    options:
      show_source: false
      heading_level: 3

## TensorTrainOperator

::: nicole.bug.TensorTrainOperator
    options:
      show_source: false
      heading_level: 3

## Constructors

::: nicole.bug.product_tt
    options:
      show_source: false
      heading_level: 3

::: nicole.bug.ttutils.random_tt
    options:
      show_source: false
      heading_level: 3

::: nicole.bug.siteinds
    options:
      show_source: false
      heading_level: 3

::: nicole.bug.ttutils.mpo_from_opsum
    options:
      show_source: false
      heading_level: 3

## Description

MPS core legs follow the convention `b{k-1}` (left link), `s{k}` (site), `b{k}`
(right link); MPO cores add a bra leg `s{k}*` and operator links `w{k-1}`,
`w{k}`. Dense factorizations (QR/LQ/SVD and matrix exponentials) run on the
per-block `torch` tensors, so symmetry sectors are never mixed.

## See Also

- [Tensor Networks (BUG)](index.md): subpackage overview
- [BUG Integrator](bug.md): two-site time evolution
- [decomp](../decomposition/decomp.md): the block-wise decomposition used internally
