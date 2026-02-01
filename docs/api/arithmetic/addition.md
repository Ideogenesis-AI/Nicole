# Arithmetic Operations

Basic arithmetic operations on tensors.

## Supported Operations

Nicole tensors support standard arithmetic operations:

### Addition and Subtraction

```python
C = A + B  # Element-wise addition
D = A - B  # Element-wise subtraction
E = A + scalar  # Add scalar to all blocks
```

**Requirements**: Tensors must have identical index structure (same indices, same tags).

### Scalar Multiplication

```python
F = 2.0 * A  # Multiply by scalar
G = A * 3.0  # Multiply by scalar
H = A / 2.0  # Divide by scalar
```

### Negation

```python
I = -A  # Negate all elements
```

## Description

Arithmetic operations are performed block-wise. Each operation processes only the blocks that exist in both tensors (for binary operations) or in the single tensor (for unary operations).

## See Also

- [oplus](oplus.md): Direct sum operation
- [Tensor](../core/tensor.md): Main tensor class
- [Examples: Arithmetic](../../examples/basic/arithmetic.md)

## Notes

- Operations preserve block structure
- Results have same dtype unless promotion occurs
- Blocks must have matching shapes for addition/subtraction
