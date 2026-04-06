# Display Utilities

Pretty-printing for symmetry-aware tensors.

## Description

Nicole tensors have built-in display functionality accessed via `print(tensor)` or `tensor.show(block_ids)`.

### Display Format

```
  info:  2x { 3 x 1 }  having 'A'    Tensor,  { i*, j }
  data:  2-D float64 (48 B)    4 x 4 => 4 x 4  @ norm = 2.85035

     1.  1x1     |  1x1     [ -1 ; -1 ]  -1.30218
     2.  2x2     |  1x1     [  0 ;  0 ]     32 B
     3.  1x1     |  1x1     [  1 ;  1 ]  -1.95104
```

### Components

- **info**: `2x { 3 x 1 }  having 'A'    Tensor,  { i*, j }`
    - `2x` — tensor order (number of indices)
    - `{ 3 x 1 }` — 3 blocks; 1 charge component per index (increases for product groups)
    - `'A'` — symmetry group label (e.g. `'SU2'`, `'U1×Z2'`)
    - `Tensor` — label identifying the tensor kind; defaults to `Tensor`
    - `{ i*, j }` — index tags; `*` marks OUT direction
- **data**: `2-D float64 (48 B)    4 x 4 => 4 x 4  @ norm = 2.85035`
    - `2-D float64` — order of underlying dense arrays (torch.Tensor) and data type
    - `(48 B)` — total memory across all blocks
    - `4 x 4 => 4 x 4` — multiplets per index (reduced space) → physical states per index (full space); equal for Abelian, diverge for non-Abelian
    - `@ norm = 2.85035` — Frobenius norm
- **blocks**: `1.  1x1     |  1x1     [ -1 ; -1 ]  -1.30218`
    - `1.` — block id (1-indexed, matching the argument to `show()`)
    - `1x1` — shape of the stored reduced tensor (multiplet counts per index)
    - `| 1x1` — irrep dimension per index (always `1×1` for Abelian; equals `2j+1` for SU(2))
    - `[ -1 ; -1 ]` — charges per index; semicolons separate indices, spaces separate tuple components
    - `-1.30218` or `32 B` — scalar value if the block has a single element, otherwise memory footprint

### SU(2) Tensor Example

For an SU(2) tensor with a spin-½ sector (2j=1) and a spin-1 sector (2j=2):

```
  info:  2x { 2 x 1 }  having 'SU2',   Tensor,  { i*, j }
  data:  2-D float64 (16 B)    2 x 2 => 5 x 5  @ norm = 0.360489

     1.  1x1     |  2x2     [ 1 ; 1 ]   0.3367  {+}
     2.  1x1     |  3x3     [ 2 ; 2 ]   0.1288  {+}
```

Key differences from Abelian:

- `'SU2'` instead of `'A'` marks the non-Abelian group
- **data** line shows `2 x 2 => 5 x 5`: 2 multiplets per index (reduced space), expanding to 5 physical states (2+3) per index
- **blocks** show two dimensions per entry: `1x1` (shape of the stored reduced tensor R per block, i.e. multiplet count per index) and `| 2x2` or `| 3x3` (irrep dimension `2j+1` per index — magnetic substates per multiplet); the full physical block size is their elementwise product
- `{+}` or `{-}` is the sign of the Bridge weight entry `W`; shown only when the weight matrix has a single element (`num_components = om_dimension = 1`)

### Product Group Examples

The second number in `{ N x K }` increases to `K=2` (or more) when charges are tuples.

**Abelian: U(1) × Z(2)** — 3 blocks, each charge is a 2-component tuple:

```
  info:  2x { 3 x 2 }  having 'U1×Z2',   Tensor,  { i*, j }
  data:  2-D float64 (48 B)    4 x 4 => 4 x 4  @ norm = 1.23835

     1.  1x1     |  1x1     [ -1  1 ; -1  1 ]  -0.1863
     2.  2x2     |  1x1     [  0  0 ;  0  0 ]    32 B
     3.  1x1     |  1x1     [  1  1 ;  1  1 ]  -1.123
```

- `{ 3 x 2 }` — 3 blocks, 2 charge components per index (U(1) + Z(2))
- `'U1×Z2'` — symmetry group label
- Charge columns: `[ -1  1 ; -1  1 ]` — first value is the U(1) charge, second is the Z(2) charge; semicolons separate indices

**Non-Abelian: U(1) × SU(2)** — combines a U(1) particle-number charge with an SU(2) spin charge:

```
  info:  2x { 3 x 2 }  having 'U1×SU2',   Tensor,  { i*, j }
  data:  2-D float64 (24 B)    3 x 3 => 5 x 5  @ norm = 0.430029

     1.  1x1     |  2x2     [ -1  1 ; -1  1 ]   0.2345  {+}
     2.  1x1     |  1x1     [  0  0 ;  0  0 ]   0.3367  {+}
     3.  1x1     |  2x2     [  1  1 ;  1  1 ]   0.1288  {+}
```

- `'U1×SU2'` — non-Abelian product group
- `3 x 3 => 5 x 5` — 3 multiplets per index; 5 physical states per index (1 singlet + 2 doublets = 1×1 + 2×2 = 5)
- Block 2 has `| 1x1` (singlet, irrep_dim=1) and blocks 1, 3 have `| 2x2` (spin-½ doublet, irrep_dim=2)
- `{+}` sign indicators appear as in the pure SU(2) case

## Methods

### print(tensor)

Standard print with line limit (default 10 blocks).

### tensor.show(block_ids)

Show specific blocks or all blocks without limit.

## See Also

- [Tensor](../core/tensor.md): Main tensor class
- [Examples: First Tensor](../../examples/basic/first-tensor.md)

## Notes

- Block ids are 1-based in display
- Memory usage shown per block and total
- `*` marks outgoing (OUT) directions
- The tensor label is set automatically by construction routines: `Diagonal` for diagonal tensors, `Operator` for space operators; generic tensors default to `Tensor`
