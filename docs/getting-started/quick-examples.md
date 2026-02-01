# Quick Examples

This page provides quick examples to get you started with Nicole. After understanding the [Core Concepts](core-concepts.md), these examples will help you create and manipulate symmetry-aware tensors.

```python exec="1" session="quick" result=""
from nicole import Tensor, Index, Sector, Direction, U1Group, Z2Group, ProductGroup
from nicole import identity, contract, trace, permute, transpose, conj, decomp
import numpy as np
```

## Basic U(1) Tensor

Create a simple tensor with U(1) symmetry (e.g., particle number conservation):

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
# Create a U(1) symmetric index
group = U1Group()
index = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector(charge=0, dim=2),   # neutral sector, 2 states
        Sector(charge=1, dim=1),   # charge +1, 1 state
        Sector(charge=-1, dim=1),  # charge -1, 1 state
    )
)

# Create a random tensor with this symmetry
tensor = Tensor.random([index, index.flip()], itags=["i", "j"], seed=42)
print(tensor)
```

### Understanding the Output

When you print a tensor, Nicole displays:

```
  info:  2x { 3 x 1 }  having 'A'    Tensor,  { i*, j }
  data:  2-D float64 (48 B)    4 x 4 => 4 x 4  @ norm = 2.85035

     1.  1x1     |  1x1     [ -1 ; -1 ] -1.30218.
     2.  2x2     |  1x1     [  0 ;  0 ]    32 B
     3.  1x1     |  1x1     [  1 ;  1 ] -1.95104.
```

- **info**: Tensor shape, symmetry type, and index tags (* marks OUT direction)
- **data**: Data type, total bytes, multiplets and states, Frobenius norm
- **Blocks**: Each line shows block dimensions, multiplet info, charges, and value or memory

## Creating Different Tensors

### Zero and Random Tensors

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
index_out = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector(charge=0, dim=2),
        Sector(charge=1, dim=1),
        Sector(charge=-1, dim=1),
    )
)
index_in = Index(
    Direction.IN,
    group,
    sectors=(
        Sector(charge=0, dim=3),
        Sector(charge=1, dim=2),
        Sector(charge=-1, dim=1),
    )
)

# Create a zero tensor
tensor_zero = Tensor.zeros([index_out, index_in], itags=["i", "j"])
print(f"Zero tensor:\n{tensor_zero}\n")

# Create a random tensor
tensor_random = Tensor.random([index_out, index_in], itags=["i", "j"], seed=42)
print(f"Random tensor:\n{tensor_random}")
```

### Identity Tensors

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))

# Create identity tensor
I = identity(idx, itags=("i", "j"))
print(I)
```

## Basic Operations

### Arithmetic Operations

Tensors with the same index structure support element-wise arithmetic:

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))

A = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=1)
B = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=2)

# Addition and subtraction
C = A + B
D = A - B

# Scalar multiplication
E = 2.5 * A

# Norm
print(f"Norm of A: {A.norm()}")
```

### Tensor Contraction

Contract two tensors along matching indices:

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
idx_out = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
idx_in = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 1)))

# Create tensors
A = Tensor.random([idx_out, idx_out], itags=["i", "mid"], seed=10)
B = Tensor.random([idx_in, idx_out], itags=["mid", "j"], seed=11)

# Automatic contraction on matching tags with opposite directions
result = contract(A, B)
print(f"Automatic contraction result:\n{result}\n")

# Or specify contraction pairs explicitly
result2 = contract(A, B, axes=(1, 0))
print(f"Explicit contraction result:\n{result2}")
```

### Trace

Take the trace of a tensor:

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
left = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))
right = Index(Direction.IN, group, sectors=(Sector(0, 2), Sector(1, 1)))

T = Tensor.random([left, right], itags=["i", "j"], seed=99)

# Trace over both indices
scalar = trace(T, axes=(0, 1))
print(f"Trace result: {scalar.norm()}")
```

### Permutation and Transpose

Reorder tensor axes:

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))

T = Tensor.random([idx, idx.flip(), idx], itags=["i", "j", "k"], seed=5)

# Permute axes
T_perm = permute(T, [2, 0, 1])  # k, i, j
print(f"Permuted (k, i, j):\n{T_perm}\n")

# Transpose (reverses axis order by default)
T_trans = transpose(T)  # k, j, i
print(f"Transposed (k, j, i):\n{T_trans}")
```

### Conjugation

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
T = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=43)

# Conjugate tensor (flips all index directions)
T_conj = conj(T)
print(f"Conjugated tensor (directions flipped: OUT→IN, IN→OUT):\n{T_conj}")
```

## Tensor Decomposition (SVD)

Decompose a tensor using SVD:

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
idx = Index(Direction.OUT, group, sectors=(Sector(0, 3), Sector(1, 2)))

T = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=7)

# Perform SVD: T ≈ U @ S @ Vh
U, S, Vh = decomp(T, axes=0, mode="SVD")

print(f"U = \n{U}\n")
print(f"S = \n{S}\n")
print(f"Vh = \n{Vh}")
```

## Working with Multiple Symmetries

Use `ProductGroup` to combine multiple symmetries:

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
# Create U(1) × Z(2) group (particle number and parity)
group = ProductGroup([U1Group(), Z2Group()])

# Create index with composite charges (particle, parity)
index = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector((0, 0), 1),  # vacuum: no particles, even parity
        Sector((1, 1), 2),  # one particle, odd parity
        Sector((2, 0), 1),  # two particles, even parity
    )
)

# Create and manipulate tensors as before
T = Tensor.random([index, index.flip()], itags=["i", "j"], seed=42)
print(T)
```

## Advanced: Accessing Blocks

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
# Access blocks by integer index (1-indexed)
first_block = tensor.block(1)

# Get block key (charges)
key = tensor.key(1)
print(f"Block 1 has charges: {key}")

# Iterate over all blocks
for key, block_array in tensor.data.items():
    print(f"Block {key}: shape {block_array.shape}")
```

## Tips for Getting Started

1. **Index Tags**: Use descriptive tags like "left", "right", "phys" to make code more readable
2. **Seeds**: Always use seeds with `Tensor.random()` for reproducible results in development
3. **Display**: Call `print(tensor)` frequently to visualize block structure and verify charge conservation
4. **Direction**: Remember that OUT and IN directions must be opposite for valid contractions
5. **Performance**: Nicole only stores and operates on admissible blocks, automatically saving memory

## Common Patterns

### Matrix-Matrix Multiplication

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
# Create compatible tensors (2-index each)
M1 = Tensor.random([idx_out, idx_in], itags=["i", "j"], seed=20)
M2 = Tensor.random([idx_out, idx_in], itags=["j", "k"], seed=21)

# Contract (like matrix multiplication)
result = contract(M1, M2)  # Contracts on "j"
print(f"Result of M1 @ M2:\n{result}")
```

### Building Tensor Networks

```python exec="1" session="quick" result="console" idprefix="" source="material-block"
# Three tensors forming a chain
A = Tensor.random([idx_out, idx_out], itags=["i", "bond1"], seed=30)
B = Tensor.random([idx_in, idx_out], itags=["bond1", "bond2"], seed=31)
C = Tensor.random([idx_in, idx_out], itags=["bond2", "j"], seed=32)

# Contract step by step
AB = contract(A, B)
ABC = contract(AB, C)
print(f"Final contracted tensor:\n{ABC}")
```

## Next Steps

- **[API Reference](../api/index.md)**: Detailed documentation of all functions and classes
- **[Examples](../examples/index.md)**: More complex use cases organized by topic

## Troubleshooting

### ValueError: Sector dim must be positive
Ensure all sector dimensions are positive integers when creating indices.

### ValueError: Duplicate sector charge
Each charge can appear only once per index. Check your sector definitions for duplicates.

### Contraction fails
Verify that:
- Tags match between the two tensors
- Directions are opposite (OUT ↔ IN)
- Symmetry groups are compatible

For more help, check the [API Reference](../api/index.md) or open an issue on [GitHub](https://github.com/Ideogenesis-AI/Nicole/issues).
