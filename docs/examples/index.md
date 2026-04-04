# Examples

This section provides comprehensive, executable examples demonstrating Nicole's capabilities for symmetry-aware tensor computations. Each example includes **live code execution** with real outputs, allowing you to see exactly what Nicole produces.

The examples are organized by complexity and topic:

- **Basic**: Learn tensor creation, arithmetic, and block access
- **Symmetries**: Work with U(1), Z(2), and product groups
- **Operations**: Master contractions, decompositions, and manipulations  
- **Advanced**: Build operators and optimize performance

All code examples use `markdown-exec` to show both the source code and its actual output, making it easy to understand what each operation does and verify the results.

## Getting Started

New to Nicole? Start here:

- **[First Tensor](basic/first-tensor.md)**: Create your first symmetry-aware tensor
- **[Arithmetic Operations](basic/arithmetic.md)**: Add, subtract, and scale tensors
- **[Indexing and Blocks](basic/indexing.md)**: Access and manipulate tensor blocks

## Symmetries

Learn to work with different symmetry groups:

- **[U1 Examples](symmetries/u1-examples.md)**: Particle number conservation
- **[Z2 Examples](symmetries/z2-examples.md)**: Parity and binary symmetries  
- **[SU(2) Examples](symmetries/su2-examples.md)**: Full spin-rotation symmetry
- **[Product Groups](symmetries/product-examples.md)**: Multiple simultaneous symmetries

## Operations

Master tensor operations:

- **[Contraction](operations/contraction-examples.md)**: Contract, trace, and multiply tensors
- **[Decomposition](operations/decomposition-examples.md)**: SVD and tensor decomposition
- **[Manipulation](operations/manipulation-examples.md)**: Permute, transpose, conjugate
- **[Serialization](advanced/serialization-examples.md)**: Save and load tensors to disk

## Advanced

For experienced users:

- **[Build Operators](advanced/build-operators.md)**: Build physical operators with symmetries
- **[Load Space](advanced/load-space-examples.md)**: Use `load_space` for quantum systems
- **[Performance Tips](advanced/performance.md)**: Optimize your code

## Example Format

Each example page includes:

1. **Complete runnable code**
2. **Step-by-step explanations**
3. **Expected output**
4. **Links to API reference**

## Running Examples

All examples assume Nicole is installed:

```bash
pip install nicole
```

Then copy-paste code directly into Python or Jupyter notebooks.

## Contributing Examples

Have an interesting use case? Contributions welcome! See the [GitHub repository](https://github.com/Ideogenesis-AI/Nicole) to submit examples.

## See Also

- [API Reference](../api/index.md): Complete function documentation
- [Core Concepts](../getting-started/core-concepts.md): Understanding symmetry-aware tensors
- [Quick Examples](../getting-started/quick-examples.md): Hands-on introduction
