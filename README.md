<h1 align="center">
  <img src="docs/images/nicole.png" alt="Nicole Tensor Library" width="300">
</h1>

<!-- ## Nicole: A Symmetry-Aware Tensor Library -->

Nicole is a Python library for symmetry-aware tensor computations, specifically designed for quantum many-body physics and tensor network algorithms. It provides efficient block-sparse tensor operations that respect Abelian symmetries (U(1), Z₂, etc.), enabling memory-efficient and computationally optimized tensor network calculations.

With the assistance of various AI coding agent, Nicole reimagines the block-symmetric tensor approach with a Python-native API built on NumPy, making it accessible to the broader scientific Python ecosystem while maintaining the mathematical rigor needed for quantum physics applications.


## Key Features

- **Block-Sparse Tensors**: Memory-efficient representation of tensors with conserved quantum numbers
- **Abelian Symmetries**: Built-in support for U(1) (particle number, magnetization) and Z₂ (parity), etc.
- **Charge Conservation**: Automatic enforcement of selection rules through symmetry-aware indices
- **NumPy Backend**: Pure Python implementation using NumPy for high-performance dense block operations
- **Tensor Operations**: Essential operations including contraction, trace, SVD decompositions, and more
- **Type-Safe API**: Modern Python with type hints for better IDE support and fewer runtime errors
- **Extensible Design**: Clean abstractions for adding custom symmetry groups


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


## Acknowledgments

Nicole is inspired by the [QSpace](https://bitbucket.org/qspace4u/) tensor library developed for MATLAB. While QSpace excels in complex symmetries (e.g. SU(N), Sp(N), SO(N)) with a C++ backend, Nicole focuses on providing a Python implementation for Abelian symmetries (SU(2) or more will come as a plugin) with an emphasis on clarity, extensibility, and integration with the scientific Python ecosystem.


## License

Nicole is licensed under the **GNU General Public License v3.0 (GPL-3.0)**. This means you are free to use, modify, and distribute this software under the terms of the GPL-3.0 license. We encourage you to share any improvements you make back to the community, helping Nicole grow and benefit all users. See the [LICENSE](LICENSE) file for the full license text. For more information about GPL-3.0, visit https://www.gnu.org/licenses/gpl-3.0.html
