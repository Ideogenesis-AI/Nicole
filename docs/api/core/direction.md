# Direction

Enum specifying tensor index orientation.

::: nicole.Direction
    options:
      show_source: false
      heading_level: 2
      members:
        - IN
        - OUT
        - reverse

## Description

The `Direction` enum specifies whether an index is incoming or outgoing, which determines how charges contribute to conservation laws.

## Values

- **`Direction.IN = -1`**: Incoming index
- **`Direction.OUT = 1`**: Outgoing index

## Charge Conservation

Direction affects charge conservation:

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
- [Examples: First Tensor](../../examples/basic/first-tensor.md)

## Notes

For contraction, paired indices must have opposite directions. Use `Index.flip()` to reverse direction without conjugating charges.
