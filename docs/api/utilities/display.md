# Display Utilities

Pretty-printing for symmetry-aware tensors.

## Description

Nicole tensors have built-in display functionality accessed via `print(tensor)` or `tensor.show()`.

### Display Format

```
  info:  2x { 3 x 1 }  having 'A'    Tensor,  { i*, j }
  data:  2-D float64 (48 B)    4 x 4 => 4 x 4  @ norm = 2.85035

     1.  1x1     |  1x1     [ -1 ; -1 ] -1.30218.
     2.  2x2     |  1x1     [  0 ;  0 ]    32 B
     3.  1x1     |  1x1     [  1 ;  1 ] -1.95104.
```

### Components

- **info**: 
  - `2x` - tensor order (2 indices)
  - `{ 3 x 1 }` - 3 blocks with 1 charge component each
  - `'A'` - Abelian symmetry group (e.g., U(1), Z(2))
  - `{ i*, j }` - index tags (* marks OUT direction)
- **data**: Dimensionality, dtype, memory, multiplets and states, norm
- **blocks**: Per-block dimensions, multiplet info, charges, value or memory

## Methods

### print(tensor)

Standard print with line limit (default 10 blocks).

### tensor.show(block_indices=None)

Show specific blocks or all blocks without limit.

## See Also

- [Tensor](../core/tensor.md): Main tensor class
- [Examples: First Tensor](../../examples/basic/first-tensor.md)

## Notes

- Block indices are 1-based in display
- Memory usage shown per block and total
- `*` marks outgoing (OUT) directions
