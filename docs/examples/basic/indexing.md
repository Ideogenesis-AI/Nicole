# Indexing and Block Access

Symmetry-aware tensors store data in multiple blocks, each labeled by quantum numbers. This page shows how to access and manipulate these blocks and their associated indices.

**Key operations:**

- **Block access**: Retrieve individual blocks by integer index or charge key
- **Iteration**: Loop over all blocks in a tensor
- **Subsector extraction**: Create new tensors containing only specific blocks
- **Index manipulation**: Inspect properties, modify tags, flip directions
- **Trivial indices**: Insert auxiliary indices for tensor network operations

Understanding block structure is essential for debugging, analyzing tensor properties, and performing advanced manipulations.

```python exec="1" session="indexing" result=""
from nicole import Tensor, Index, Sector, Direction, U1Group, subsector
import numpy as np
```

## Accessing Blocks

```python exec="1" session="indexing" result="console" idprefix="" source="material-block"
group = U1Group()
idx = Index(Direction.OUT, group, sectors=(Sector(0, 2), Sector(1, 1), Sector(-1, 1)))
T = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=42)

# Display tensor to see block structure
print(f"T = \n{T}\n")

# Access blocks by integer index (1-based, matching display)
first_block = T.block(1)
print(f"First block shape: {first_block.shape}")

# Get block key (charges)
key_1 = T.key(1)
print(f"First block charges: {key_1}\n")

# Access blocks directly via data dictionary
if (0, 0) in T.data:
    block_00 = T.data[(0, 0)]
    print(f"Block (0,0) shape: {block_00.shape}")
```

## Iterating Over Blocks

```python exec="1" session="indexing" result="console" idprefix="" source="material-block"
# Method 1: Using sorted_keys property
for key in T.sorted_keys:
    block = T.data[key]
    print(f"Block {key}: shape {block.shape}, norm {np.linalg.norm(block):.4f}")

print()

# Method 2: Using integer indices
for i in range(1, len(T.data) + 1):
    key = T.key(i)
    block = T.block(i)
    print(f"Block {i} has charges {key}")

print()

# Method 3: Direct dictionary iteration
for key, block in T.data.items():
    print(f"Charges {key}: {block.shape}")
```

## Extracting Block Subsets

```python exec="1" session="indexing" result="console" idprefix="" source="material-block"
# Get a single block (using integer)
T_single = subsector(T, 1)
print(f"Single block: {len(T_single.data)}\n")

# Get multiple blocks (using list)
T_sub = subsector(T, [1, 2])
print(f"Original blocks: {len(T.data)}")
print(f"Subset blocks: {len(T_sub.data)}\n")

# Note: subsector automatically removes unused sectors from indices

# Extract specific charge sectors
# Find blocks with positive charges only
positive_blocks = []
for i in range(1, len(T.data) + 1):
    key = T.key(i)
    if all(q >= 0 for q in key):
        positive_blocks.append(i)

if positive_blocks:
    T_positive = subsector(T, positive_blocks)
    print(f"Positive charge blocks: {len(T_positive.data)}")
```

## Displaying Index Information

```python exec="1" session="indexing" result="console" idprefix="" source="material-block"
# The Index class has a __str__ method for display
idx_display = T.indices[0]
print(f"First index:\n{idx_display}\n")

# Second index
idx_display2 = T.indices[1]
print(f"Second index:\n{idx_display2}")
```

## Index Properties

```python exec="1" session="indexing" result="console" idprefix="" source="material-block"
# Get index information
idx_inspect = T.indices[0]
print(f"Direction: {idx_inspect.direction}")
print(f"Group: {idx_inspect.group.name}")
print(f"Number of sectors: {len(idx_inspect.sectors)}")
print(f"Total dimension: {idx_inspect.dim}\n")

# List all sectors
for sector in idx_inspect.sectors:
    print(f"Charge {sector.charge}: dimension {sector.dim}")

print()

# Get charges and dimensions
charges = idx_inspect.charges()
dim_map = idx_inspect.sector_dim_map()
print(f"Charges: {charges}")
print(f"Dimension map: {dim_map}")
```

## Modifying Index Tags

```python exec="1" session="indexing" result="console" idprefix="" source="material-block"
# Change tags to make contractions clearer
T_tag = Tensor.random([idx, idx.flip(), idx], itags=["i", "j", "k"], seed=99)
print(f"Original tags: {T_tag.itags}")

# Mode 1: Retag by name using dictionary mapping
T_tag.retag({"i": "left", "k": "right"})
print(f"After retag by name: {T_tag.itags}")

# Mode 2: Retag all at once with full list
T_tag.retag(["a", "b", "c"])
print(f"After retag all: {T_tag.itags}")

# Mode 3: Retag by position
T_tag.retag([0, 2], ["x", "z"])
print(f"After retag by position: {T_tag.itags}")
```

## Flipping and Dualizing Indices

```python exec="1" session="indexing" result="console" idprefix="" source="material-block"
# Flip: reverse direction only
idx_out = Index(Direction.OUT, group, sectors=(Sector(1, 2),))
idx_in = idx_out.flip()

print(f"Original: {idx_out.direction}")  # OUT
print(f"Flipped: {idx_in.direction}")    # IN
print(f"Charges unchanged: {idx_out.sectors == idx_in.sectors}\n")

# Dual: reverse direction AND conjugate charges
idx_dual = idx_out.dual()
print(f"Dual direction: {idx_dual.direction}")  # IN
# For U(1), charges are negated
print()

# Check sectors
print("Original sectors:")
for s in idx_out.sectors:
    print(f"  Charge {s.charge}, dim {s.dim}")

print("\nDual sectors:")
for s in idx_dual.sectors:
    print(f"  Charge {s.charge}, dim {s.dim}")
```

## Inserting Trivial Indices

```python exec="1" session="indexing" result="console" idprefix="" source="material-block"
# Create a tensor to demonstrate inserting a trivial index
T_insert = Tensor.random([idx, idx.flip()], itags=["i", "j"], seed=7)
print(f"Before: {len(T_insert.indices)} indices")
print(f"Tags: {T_insert.itags}")
print(T_insert)
```

```python exec="1" session="indexing" result="console" idprefix="" source="material-block"
# Insert a trivial index (neutral charge, dimension 1) at position 1
T_insert.insert_index(position=1, direction=Direction.OUT, itag="trivial")
print(f"After: {len(T_insert.indices)} indices")
print(f"Tags: {T_insert.itags}")
print(T_insert)
```

## Block Memory Usage

```python exec="1" session="indexing" result="console" idprefix="" source="material-block"
# Check memory usage per block
print("Block memory usage:")
for i in range(1, len(T.data) + 1):
    key = T.key(i)
    block = T.block(i)
    mem_bytes = block.nbytes
    mem_kb = mem_bytes / 1024
    print(f"Block {i} {key}: {block.shape} -> {mem_bytes} B ({mem_kb:.2f} KB)")

# Total memory
total_bytes = sum(block.nbytes for block in T.data.values())
print(f"\nTotal block memory: {total_bytes} B ({total_bytes/1024:.2f} KB)")
```

## See Also

- API Reference: [Tensor](../../api/core/tensor.md)
- API Reference: [Index](../../api/core/index-class.md)
- API Reference: [subsector](../../api/manipulation/subsector.md)
- Previous: [Arithmetic](arithmetic.md)
- Next: [Symmetry Examples](../symmetries/u1-examples.md)
