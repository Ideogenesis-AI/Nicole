# Nicole: A Symmetry-Aware Tensor Library

Welcome to the documentation for **Nicole**, a Python library for symmetry-aware tensor computations, specifically designed for quantum many-body physics and tensor network algorithms.

## What is Nicole?

Nicole provides efficient block-sparse tensor operations that respect Abelian symmetries (U(1), Z₂, etc.), enabling memory-efficient and computationally optimized tensor network calculations. Created as part of the Ideogenesis-AI effort in studying quantum many-body systems, Nicole reimagines the block-symmetric tensor approach with a Python-native API built on NumPy, making it accessible to the broader scientific Python ecosystem while maintaining the mathematical rigor needed for quantum physics applications.

## Key Features

- **Block-Sparse Tensors**: Efficient representation of tensors with conserved quantum numbers
- **Abelian Symmetries**: Built-in support for U(1), Z₂, and their product groups
- **Charge Conservation**: Enforcement of selection rules through symmetry-aware indices
- **NumPy Backend**: Implementation using NumPy for high-performance dense operations
- **Tensor Network Operations**: Essential operations including contraction, trace, SVD, and more
- **Type-Safe API**: Modern Python with type hints for better IDE support
- **Extensible Design**: Clean abstractions for adding custom symmetry groups

## Getting Started

New to Nicole? Follow these steps:

1. **[What is Nicole?](getting-started/what-is-nicole.md)**: Understand what Nicole is and how it works
2. **[Installation](getting-started/installation.md)**: Install Nicole via pip or from source
3. **[Core Concepts](getting-started/core-concepts.md)**: Learn about symmetries, charges, sectors, and indices
4. **[Quick Examples](getting-started/quick-examples.md)**: Create your first tensors and learn basic operations

## Documentation Structure

This documentation is organized into three main sections:

- **[What is Nicole?](getting-started/what-is-nicole.md)**: Overview and introduction
- **[Installation](getting-started/installation.md)**: Install Nicole on your system
- **[Core Concepts](getting-started/core-concepts.md)**: Understand the fundamentals of symmetry-aware tensors
- **[Quick Examples](getting-started/quick-examples.md)**: Hands-on examples to get started quickly
- **[API Reference](api/index.md)**: Comprehensive documentation for all classes and functions
- **[Examples](examples/index.md)**: Practical examples and use cases organized by topic

## Use Cases

Nicole is designed for researchers and developers working on:

- **Tensor Network Algorithms**: DMRG, TEBD, PEPS, and other tensor network methods
- **Quantum Many-Body Physics**: Systems with conserved quantum numbers
- **Condensed Matter Theory**: Strongly correlated electron systems
- **Quantum Chemistry**: Fermionic and bosonic systems with symmetries

## Acknowledgments

Nicole is inspired by the [QSpace](https://bitbucket.org/qspace4u/) tensor library developed for MATLAB. While QSpace excels in complex symmetries (e.g. SU(N), Sp(N), SO(N)) with a C++ backend, Nicole focuses on providing a Python implementation for Abelian symmetries (SU(2) or more will come as a plugin) with an emphasis on clarity, extensibility, and integration with the scientific Python ecosystem.

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, implementing new symmetry groups, or improving documentation, your help is appreciated. You can also contribute by requesting new features or reporting performance bottlenecks.

**Ways to contribute:**
- Report issues and request features via [GitHub Issues](https://github.com/Ideogenesis-AI/Nicole/issues)
- Submit pull requests with bug fixes or enhancements
- Improve documentation and add examples
- Share your use cases and provide constructive feedback

**Development guidelines:**
- Ensure all contributions include appropriate tests
- Follow the existing code style (enforced by `ruff`)
- Add type hints for new functions and classes
- Update documentation for user-facing changes

**Authors and Maintainers:**

Nicole is created and maintained by [Changkai Zhang](https://chx-zh.cc) as part of the Ideogenesis-AI effort in studying quantum many-body systems. If you have questions about contributing to the project or are interested in collaboration opportunities, please feel free to open an issue on GitHub or contact the maintainer directly.

## License

Nicole is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This means you are free to use, modify, and distribute this software under the terms of the GPL-3.0 license. We encourage you to share any improvements you make back to the community, helping Nicole grow and benefit all users. For more information about GPL-3.0, visit [https://www.gnu.org/licenses/gpl-3.0.html](https://www.gnu.org/licenses/gpl-3.0.html)
