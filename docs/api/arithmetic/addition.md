# Arithmetic Operations

The operators `+`, `-`, `*`, `/`, and unary minus, defined block by block on symmetry-aware tensors. Each acts on the blocks that a tensor actually stores, so symmetry is carried through with no bookkeeping on the caller's part. A binary operation asks the two tensors to agree on their frame; a scalar operation asks nothing.

## Description

Addition and subtraction work over the union of the two tensors' block keys. A key held by both sides is combined entry by entry, and a key held by one side only passes into the output as it stands. The indices are unioned too, so the operands need not carry the same sectors: each output index offers every charge either side had.

What must agree is the frame. Both tensors need the same order, every axis must match in symmetry group and direction, and a charge present on both sides of an axis must carry the same dimension there. A violation raises `ValueError`. The itags are not part of this check: the left operand's tags are the ones that survive, so adding tensors whose axes are tagged differently succeeds quietly.

Scalar multiplication and division scale every stored block and leave the structure untouched.

## Supported Operations

Nicole tensors support standard arithmetic operations:

### Addition and Subtraction

```python
C = A + B  # Element-wise addition
D = A - B  # Element-wise subtraction
```

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

## See Also

- [allclose](allclose.md): Numerical equality within tolerance
- [oplus](oplus.md): Direct sum operation
- [Tensor](../core/tensor.md): Main tensor class, including its dtype rules
- [Examples: Arithmetic](../../examples/basic/arithmetic.md)

## Notes

- Operands are promoted to a common dtype first, so a real tensor combines with a complex one directly. See [Tensor](../core/tensor.md) for the full dtype surface.
- Scalar operations preserve precision, so a `float32` tensor times `1j` gives `complex64`.
- For a generic symmetry group, intertwiner weights are merged when they are collinear and concatenated when they are not, after which linearly dependent components are compressed away.
- Two scalar tensors are a special case: they combine directly, preserving the autograd graph.
