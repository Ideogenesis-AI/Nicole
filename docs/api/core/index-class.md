# Index

The symmetry-aware tensor index: a direction, a symmetry group, and the charge sectors available alongside. An `Index` carries symmetry into a tensor, since its sectors decide which blocks may exist and how large each one can be, while its direction decides how their charges enter conservation. Indices are frozen, so the structure of a tensor cannot drift after it is built.

::: nicole.Index
    options:
      show_source: false
      heading_level: 2
      members:
        - __init__
        - dim
        - num_states
        - flip
        - dual
        - charges
        - sector_dim_map

## Description

Construction validates the sectors it is given. Each charge is checked against the group, each dimension must be positive, and a charge may not appear twice on one index, so an index cannot be made to describe an inconsistent space. Being a frozen dataclass, it also cannot be edited afterwards: `flip` and `dual` return new indices to avoid mutation.

Two notions of size are worth keeping apart. `dim` sums the sector dimensions and counts multiplets, which is the space the reduced blocks are stored in. `num_states` weights each sector by `irrep_dim(charge)` and counts physical states. Under an Abelian group every irrep is 1-dimensional and the two agree. Under SU(2) they diverge: an index with sectors `(0, 1)`, `(1, 2)`, `(2, 1)` in the 2j convention has `dim` 4 and `num_states` 8, since the two spin-1/2 multiplets stand for four states.

## Direction Rules

- **`Direction.OUT`**: the charge contributes as it stands.
- **`Direction.IN`**: the charge contributes through its dual.

A block key is admissible when those contributions fuse to the group's neutral charge.

## Methods

### flip()

Returns a new instance with the direction reversed and the charges untouched. This is the partner an index contracts against, and the form [identity](../creation/identity.md) uses for its second index.

### dual()

Returns a new instance with the direction reversed and every charge replaced by `group.dual(charge)`, a negation under U(1) and the identity under the self-dual Z(2).

## See Also

- [Sector](sector.md): Charge-dimension pairs
- [Direction](direction.md): Index direction enum
- [Tensor](tensor.md): Main tensor class
- [Examples: Indexing](../../examples/basic/indexing.md)

## Notes

For contraction, indices must have:

1. Opposite directions (OUT ↔ IN)
2. Same symmetry group
3. Compatible charge sectors

`charges()` returns the charges in sector order and `sector_dim_map()` returns the charge-to-dimension mapping, both convenient when inspecting a block key against the index.
