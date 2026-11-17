# Direction

The enum that orients a tensor index. A direction records whether an index is incoming or outgoing, and that distinction enables charge conservation: an outgoing index contributes its charge as it stands, an incoming one contributes the dual. Contraction pairs an outgoing against an incoming, so direction also decides which indices are allowed to connect.

::: nicole.Direction
    options:
      show_source: false
      heading_level: 2
      members:
        - IN
        - OUT
        - reverse

## Description

Every [Index](index-class.md) carries one direction, and every operation that reasons about charges consults it. When a block key is tested for admissibility, each charge is taken with its index's direction applied: an outgoing index contributes the charge itself, an incoming index contributes the dual. The block exists when the combination reaches the group's neutral charge.

## Values

- **`Direction.IN`**: an incoming index. Integer value `+1`.
- **`Direction.OUT`**: an outgoing index. Integer value `-1`.

`Direction` is an `IntEnum`, so the values can be used in arithmetic directly. The particular signs are an internal convention with no physical content; what matters is that the two differ, and that an incoming charge enters conservation through its dual.

## Charge Conservation

```
∑(OUT charges) - ∑(IN charges) = neutral element
```

### Example with U(1)

```python
# Index 1: OUT, charge = 2
# Index 2: IN, charge = 1  
# Index 3: OUT, charge = -1
# Total: 2 - 1 + (-1) = 0 ✓ (conserved)
```

For a group whose dual is not a negation the subtraction above is the group's own conjugation. Z(2) is self-dual, so direction makes no difference to its conservation sum, while SU(2) conservation asks whether the neutral charge is among the channels the key can couple to.

## Methods

### reverse()

Returns the opposite direction:
- `Direction.OUT.reverse()` → `Direction.IN`
- `Direction.IN.reverse()` → `Direction.OUT`

## Physical Interpretation

In quantum physics:

| Direction | Represents |
|-----------|------------|
| **OUT** | Ket vectors \|ψ⟩, creation operators, outgoing states |
| **IN** | Bra vectors ⟨ψ\|, annihilation operators, incoming states |

## See Also

- [Index](index-class.md): Uses direction
- [Sector](sector.md): Charge-dimension pairs
- [contract](../contraction/contract.md): Requires opposite directions on the axes it pairs
- [Examples: First Tensor](../../examples/basic/first-tensor.md)

## Notes

- Contraction requires paired indices to carry opposite directions. `Index.flip()` reverses direction and leaves the charges alone; `Index.dual()` reverses direction and conjugates the charges as well. Use `flip` to prepare a contraction partner and `dual` when the conjugate space is intended.
- To reverse a bond already shared by two tensors, use [capcup](../manipulation/capcup.md), which applies the appropriate phase for the bond.
