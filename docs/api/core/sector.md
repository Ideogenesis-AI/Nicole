# Sector

The atom of an index's structure: one conserved charge paired with the dimension of the space carrying it. An [Index](index-class.md) is a tuple of sectors, and the sectors make a block's extent known, since a block's shape is read off from the sector dimensions that its charge key selects. Sectors are frozen, so they can be shared freely between indices.

::: nicole.Sector
    options:
      show_source: false
      heading_level: 2
      members:
        - charge
        - dim

## Description

The `charge` is the conserved quantum number, and its type follows the symmetry group, as tabulated below. The `dim` counts multiplets instead of physical states: under SU(2) a sector of charge 1 and dimension 2 holds two spin-1/2 multiplets, which is four states. Under an Abelian group the distinction collapses, every irrep being 1-dimensional.

A sector says nothing about direction. The same `Sector` can sit on an incoming and an outgoing index alike, and it is the index that decides how the charge enters conservation.

## Charge Types by Group

| Group | Charge Type | Examples |
|-------|-------------|----------|
| `U1Group` | `int` | `-2, -1, 0, 1, 2` |
| `Z2Group` | `int` (0 or 1) | `0, 1` |
| `SU2Group` | `int` ≥ 0 (2j convention) | `0, 1, 2, 3` (spins 0, 1&#x2F;2, 1, 3&#x2F;2) |
| `ProductGroup` | `tuple` | `(2, 1), (0, 0)` |

## Validation

Validation is split between the sector and the index that holds it. A `Sector` checks only what it can judge alone: the dimension must be a positive integer, and a non-positive one raises `ValueError` at construction.

The remaining checks need the group, so they happen when the sector is placed on an [Index](index-class.md): the charge must be valid for that group, and no charge may appear twice among the index's sectors. A sector built in isolation with a charge that its intended group would reject is therefore accepted on its own and refused later.

## See Also

- [Index](index-class.md): Uses sectors to define structure
- [Direction](direction.md): Index orientation
- [U1Group](../symmetry/u1-group.md): Integer charges
- [Z2Group](../symmetry/z2-group.md): Binary charges
- [SU2Group](../symmetry/su2-group.md): Spin charges (2j convention)
- [Examples: First Tensor](../../examples/basic/first-tensor.md)

## Notes

- SU(2) charges use the 2j convention, so they stay integers: 0 for spin 0, 1 for spin 1/2, 2 for spin 1.
- A `ProductGroup` charge is a tuple with one entry per component group, in the order the groups were given.
- Sector order is preserved as supplied, but block numbering and iteration go through `Tensor.sorted_keys`, so two indices holding the same sectors in different orders present their blocks identically.
