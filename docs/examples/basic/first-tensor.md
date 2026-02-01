# Creating Your First Tensor

Tensors are the fundamental objects in Nicole. Unlike standard numerical arrays, Nicole tensors are **symmetry-aware**: they automatically organize data into blocks according to quantum numbers (charges) and enforce conservation laws during operations.

This page introduces the basic concepts:

- **Indices**: Each tensor leg has an associated `Index` that defines its symmetry sectors, dimensions, and direction
- **Sectors**: Subspaces labeled by quantum numbers (e.g., particle number, spin)
- **Blocks**: Dense arrays corresponding to specific charge combinations
- **Charge conservation**: Only blocks with compatible charges are non-zero

By leveraging symmetry, Nicole tensors can be orders of magnitude more efficient than dense arrays for quantum many-body problems.

```python exec="1" session="first-tensor" result=""
from nicole import Tensor, Index, Sector, Direction, U1Group
```

## Basic Tensor Creation

```python exec="1" session="first-tensor" result="console" idprefix="" source="material-block"
# 1. Create a U(1) symmetry group (particle number conservation)
group = U1Group()

# 2. Define an outgoing index with three sectors
index_out = Index(
    Direction.OUT,
    group,
    sectors=(
        Sector(charge=0, dim=2),   # neutral sector with 2 states
        Sector(charge=1, dim=1),   # charge +1 with 1 state
        Sector(charge=-1, dim=1),  # charge -1 with 1 state
    )
)

# 3. Create the conjugate index (flipped direction)
index_in = index_out.flip()

# 4. Create a random tensor
tensor = Tensor.random([index_out, index_in], itags=["i", "j"], seed=42)

print(tensor)
```

## Understanding the Output

- **`2x`**: 2 indices (tensor order)
- **`{ 3 x 1 }`**: 3 blocks, 1 charge component each (single symmetry)
- **`'A'`**: Abelian symmetry group (U(1) in this case)
- **`{ i*, j }`**: Index tags (* marks OUT direction, i is OUT, j is IN)
- **`4 x 4 => 4 x 4`**: Total multiplets and states for each index
- Each line shows: block dimensions | multiplet info | charges | value or memory

## Creating Different Tensors

### Zero Tensor

```python exec="1" session="first-tensor" result="console" idprefix="" source="material-block"
# Create with zeros instead of random values
T_zero = Tensor.zeros([index_out, index_in], itags=["i", "j"])

print(T_zero)
```

### Multi-Index Tensor

```python exec="1" session="first-tensor" result="console" idprefix="" source="material-block"
# Create a 3-index tensor
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1)))

T_3d = Tensor.random(
    [idx, idx.flip(), idx],
    itags=["i", "j", "k"],
    seed=99
)

print(T_3d)
```

### Symmetric Sectors

```python exec="1" session="first-tensor" result="console" idprefix="" source="material-block"
# Create an index with sectors symmetric around zero
sectors_symmetric = tuple(
    Sector(charge=q, dim=2)
    for q in range(-2, 3)  # -2, -1, 0, 1, 2
)

idx_sym = Index(Direction.OUT, group, sectors=sectors_symmetric)
T_sym = Tensor.random([idx_sym, idx_sym.flip()], itags=["a", "b"], seed=7)

print(T_sym)
```

## Accessing Tensor Information

```python exec="1" session="first-tensor" result="console" idprefix="" source="material-block"
# Get tensor properties
print(f"Data type: {tensor.dtype}")
print(f"Index tags: {tensor.itags}")
print(f"Number of indices: {len(tensor.indices)}\n")

# Get specific blocks (1-indexed to match display)
block_1 = tensor.block(1)
print(f"First block shape: {block_1.shape}")

# Get block key (charges)
key_1 = tensor.key(1)
print(f"First block charges: {key_1}\n")

# Iterate over all blocks
for i in range(1, len(tensor.data) + 1):
    key = tensor.key(i)
    block = tensor.block(i)
    print(f"Block {i}: charges {key}, shape {block.shape}")
```

## Charge Conservation in Action

```python exec="1" session="first-tensor" result="console" idprefix="" source="material-block"
# Only charge-conserving blocks exist
# For (OUT, IN) tensor: OUT_charge - IN_charge = 0

# Valid blocks:
# (0, 0): 0 - 0 = 0 ✓
# (1, 1): 1 - 1 = 0 ✓
# (-1, -1): -1 - (-1) = 0 ✓

# Invalid blocks (not created):
# (1, 0): 1 - 0 = 1 ≠ 0 ✗
# (0, 1): 0 - 1 = -1 ≠ 0 ✗

# Verify: all blocks satisfy conservation
from nicole.blocks import BlockSchema

for key in tensor.data.keys():
    conserved = BlockSchema.charges_conserved(tensor.indices, key)
    print(f"Block {key}: conserved = {conserved}")  # All True
```

## Copying Tensors

```python exec="1" session="first-tensor" result="console" idprefix="" source="material-block"
# Deep copy (independent data)
T_copy = tensor.copy()
T_copy.data[tensor.key(1)][0, 0] = 999.0

# Original unchanged
print(f"Original unchanged: {tensor.data[tensor.key(1)][0, 0] != 999.0}")
```

## See Also

- API Reference: [Tensor](../../api/core/tensor.md), [Index](../../api/core/index-class.md)
- API Reference: [zeros](../../api/creation/zeros.md), [random](../../api/creation/random.md)
- Next: [Arithmetic Operations](arithmetic.md)
- Next: [Indexing](indexing.md)
