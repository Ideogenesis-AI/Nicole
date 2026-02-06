# Nicole: A Symmetry-Aware Tensor Library

Welcome to the documentation for **Nicole**, a Python library for symmetry-aware tensor computations, specifically designed for quantum many-body physics and tensor network algorithms.

## What is Nicole?

Nicole provides efficient block-sparse tensor operations that respect Abelian symmetries (U(1), Z₂, etc.), enabling memory-efficient and computationally optimized tensor network calculations. Created as part of the Ideogenesis-AI effort in studying quantum many-body systems, Nicole reimagines the block-symmetric tensor approach with a Python-native API built on PyTorch, making it accessible to the broader scientific Python ecosystem while maintaining the mathematical rigor needed for quantum physics applications.

## Key Features

- **Block-Sparse Tensors**: Efficient representation of tensors with conserved quantum numbers
- **Abelian Symmetries**: Built-in support for U(1), Z₂, and their product groups
- **Charge Conservation**: Enforcement of selection rules through symmetry-aware indices
- **PyTorch Backend**: Implementation using PyTorch for high-performance dense operations
- **GPU Acceleration**: Optional GPU support (CUDA/MPS) for accelerated computations on large tensors
- **Autograd Control**: Optional gradient tracking for efficient optimization tasks
- **Tensor Network Operations**: Essential operations including contraction, trace, SVD, and more
- **Type-Safe API**: Modern Python with type hints for better IDE support
- **Extensible Design**: Clean abstractions for adding custom symmetry groups

## Getting Started

New to Nicole? Follow these steps:

1. **[Installation](installation.md)**: Install Nicole via pip or from source
2. **[Core Concepts](core-concepts.md)**: Understand symmetries, charges, sectors, and indices
3. **[Quick Examples](quick-examples.md)**: Create your first tensors and learn basic operations

## Documentation Structure

This documentation is organized into four main sections:

- **[Getting Started](what-is-nicole.md)** (this section): Learn the basics and concepts of Nicole library
- **[API Reference](../api/index.md)**: Comprehensive documentation for all classes and functions
- **[Examples](../examples/index.md)**: Practical examples and use cases organized by topic

## Use Cases

Nicole is designed for researchers and developers working on:

- **Tensor Network Algorithms**: DMRG, TEBD, PEPS, and other tensor network methods
- **Quantum Many-Body Physics**: Systems with conserved quantum numbers
- **Condensed Matter Theory**: Strongly correlated electron systems
- **Quantum Chemistry**: Fermionic and bosonic systems with symmetries

## Acknowledgments

Nicole is inspired by the [QSpace](https://bitbucket.org/qspace4u/) tensor library developed for MATLAB. While QSpace excels in complex symmetries (e.g. SU(N), Sp(N), SO(N)) with a C++ backend, Nicole focuses on providing a Python implementation for Abelian symmetries (SU(2) or more will come as a plugin) with an emphasis on clarity, extensibility, and integration with the scientific Python ecosystem.

## License

Nicole is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This means you are free to use, modify, and distribute this software under the terms of the GPL-3.0 license. We encourage you to share any improvements you make back to the community, helping Nicole grow and benefit all users. For more information about GPL-3.0, visit [https://www.gnu.org/licenses/gpl-3.0.html](https://www.gnu.org/licenses/gpl-3.0.html)
