# Nicole: A Symmetry-Aware Tensor Library for Quantum Many-Body Simulations

Nicole is a Python library for symmetry-aware tensor computations, specifically designed for quantum many-body physics and tensor network algorithms. It provides efficient block-sparse tensor operations that respect Abelian symmetries (U(1), Z₂, etc.), enabling memory-efficient and computationally optimized tensor network calculations.

Inspired by [QSpace](https://bitbucket.org/qspace4u/), Nicole reimagines the block-symmetric tensor approach with a Python-native API built on NumPy, making it accessible to the broader scientific Python ecosystem while maintaining the mathematical rigor needed for quantum physics applications.


## Key Features

- **Block-Sparse Tensors**: Memory-efficient representation of tensors with conserved quantum numbers
- **Abelian Symmetries**: Built-in support for U(1) (particle number, magnetization) and Z₂ (parity) symmetries
- **Charge Conservation**: Automatic enforcement of selection rules through symmetry-aware indices
- **NumPy Backend**: Pure Python implementation using NumPy for high-performance dense block operations
- **Tensor Network Operations**: Essential operations including contraction, trace, SVD, and more
- **Type-Safe API**: Modern Python with type hints for better IDE support and fewer runtime errors
- **Extensible Design**: Clean abstractions for adding custom symmetry groups


## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, implementing new symmetry groups, or improving documentation, your help is appreciated.

**Ways to contribute:**
- Report bugs and request features via [GitHub Issues](https://github.com/Ideogenesis-AI/Nicole/issues)
- Submit pull requests with bug fixes or enhancements
- Improve documentation and add examples
- Share your use cases and feedback

Please ensure all contributions include appropriate tests and follow the existing code style.


## Acknowledgments

Nicole is inspired by the [QSpace](https://bitbucket.org/qspace4u/) tensor library developed for MATLAB. While QSpace excels in both Abelian and non-Abelian symmetries with a C++ backend, Nicole focuses on providing a pure Python implementation for Abelian symmetries with an emphasis on clarity, extensibility, and integration with the scientific Python ecosystem.


## License

Nicole is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This means you are free to use, modify, and distribute this software under the terms of the GPL-3.0 license.

See the [LICENSE](LICENSE) file for the full license text. For more information about GPL-3.0, visit https://www.gnu.org/licenses/gpl-3.0.html
