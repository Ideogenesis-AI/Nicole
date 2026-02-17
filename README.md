<h1 align="center">
  <img src="docs/images/nicole-high.png" alt="Nicole Tensor Library" width="300">
</h1>

<!-- ## Nicole: A Symmetry-Aware Tensor Library -->

<p align="center">
  <a href="https://pypi.org/project/nicole/"><img src="https://img.shields.io/pypi/v/nicole?color=red" alt="PyPI Version"></a>
  <a href="https://github.com/Ideogenesis-AI/Nicole/blob/stable/LICENSE"><img src="https://img.shields.io/github/license/Ideogenesis-AI/Nicole?color=orange" alt="License"></a>
  <a href="https://ideogenesis-ai.github.io/Nicole"><img src="https://img.shields.io/badge/docs-github.io-c9a400" alt="Documentation"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/pypi/pyversions/nicole?color=228b22" alt="Python Version"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.5+-blue?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/coverage-%3E95%25-9400d3" alt="Coverage"></a>
  <a href="https://pypi.org/project/nicole/"><img src="https://img.shields.io/pypi/status/nicole?color=4b0082" alt="Status"></a>
</p>

Nicole is a Python library for symmetry-aware tensor computations, specifically designed for quantum many-body physics and tensor network algorithms. It provides efficient block-sparse tensor operations that respect Abelian symmetries (U(1), Z₂, etc.), enabling memory-efficient and computationally optimized tensor network calculations.

With the assistance of various AI coding agents, Nicole reimagines the block-symmetric tensor approach with a Python-native API built upon PyTorch, making it accessible to the broader scientific Python ecosystem while maintaining the mathematical rigor needed for quantum physics applications.


## Key Features

- **Block-Sparse Tensors**: Memory-efficient representation of tensors with conserved quantum numbers
- **Abelian Symmetries**: Built-in support for U(1) (particle number, magnetization) and Z₂ (parity), etc.
- **Charge Conservation**: Automatic enforcement of selection rules through symmetry-aware indices
- **PyTorch Backend**: Python implementation using PyTorch for high-performance dense block operations
- **GPU Acceleration**: Optional GPU support (CUDA/MPS) for accelerated computations on large tensors
- **Autograd Control**: Optional gradient tracking for efficient optimization tasks
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
