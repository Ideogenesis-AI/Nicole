# Changelog

All notable changes to Nicole will be documented in this file.

## [0.2.1] - 2026-02-08

**QR Decomposition and Documentation Enhancement Release**

Release version 0.2.1 of Nicole, introducing QR decomposition for orthogonal tensor factorization, API improvements for more flexible tensor operations, and comprehensive documentation enhancements covering the PyTorch backend transition.

### QR Decomposition

#### Core Functionality
- Added `qr()` function in `decomp` module for symmetry-preserving QR decomposition
- Separates tensors into orthogonal (Q) and upper triangular (R) components
- Block-wise decomposition preserving symmetry structure
- No truncation applied, useful for canonical forms without compression

#### Integration with decomp()
- New **"QR"** mode in `decomp()` function alongside SVD, UR, and LV modes
- Consistent API with existing decomposition modes
- Custom bond index tags and flow direction control via `itag` and `flow` parameters
- Full support for axis specification (integer positions or string itags)

#### Testing
- Comprehensive test coverage in `test_factorize.py` (new)
- Tests for standard 2-index and multi-index tensors
- Validation of orthogonality (Q†Q = I) and reconstruction accuracy
- Stress tests for high-order tensors with complex symmetry structures
- Refactored test organization: `test_decomp.py` for high-level API, `test_factorize.py` for low-level functions

### API Enhancements

#### oplus Function Improvements
- Enhanced `oplus()` to accept single integer or string for `axes` parameter
- Previously required sequences: now `axes=0` works alongside `axes=[0]`
- Single string itag support: `axes='i'` alongside `axes=['i']`
- Improved ergonomics for common single-axis direct sum operations
- Added validation and error messages for non-existent itags

#### Operator Labeling
- Added `label="Operator"` parameter to all operator tensors in `load_space()`
- Operators created for spin systems, fermions, and band structures now properly labeled
- Improved tensor identification and debugging capabilities
- Enhances clarity in tensor network diagrams and summaries

#### Display Improvements
- Refined tensor summary output formatting for consistent alignment
- Enhanced readability of charge and value representations
- Improved spacing and comma placement in info lines

### Documentation Infrastructure

#### Comprehensive PyTorch Transition Documentation
- Updated all code examples from NumPy to PyTorch syntax
- Added GPU acceleration guide with CUDA and MPS (Apple Silicon) support
- Added autograd documentation for gradient tracking in tensor networks
- Updated installation guide with PyTorch dependency requirements
- Revised API reference to reflect device management and autograd features

#### QR Decomposition Documentation
- Complete API reference page for `qr()` function with examples
- Updated `decomp()` documentation to include QR mode
- Added QR section to decomposition examples with working code
- Cross-references between SVD, QR, and eigenvalue decomposition docs
- Usage patterns for orthogonal factorization without truncation

#### MkDocs Hooks for Enhanced Rendering
- Implemented custom post-processing hook for bullet list conversion
- Automatically converts markdown bullets to proper HTML `<ul>/<li>` tags in tables
- Handles multi-line bullet items with continuation line detection
- Supports inline HTML tags (code, emphasis, links) within bullet lists
- Debug logging mode (`NICOLE_HOOKS_DEBUG=1`) for troubleshooting
- Logs stored in `.logging/hooks.log` with detailed transformation tracking

#### Documentation Quality Improvements
- Fixed bullet list rendering in parameter tables across API docs
- Enhanced navigation structure with clear decomposition method organization
- Added "linalg" labels for low-level functions (svd, qr, eig)
- Improved cross-referencing between related functions
- Updated performance recommendations for CPU vs GPU usage in tensor networks

### Code Quality and Maintenance

#### Test Organization
- Refactored decomposition tests into focused modules:
  - `test_decomp.py`: High-level `decomp()` API tests
  - `test_factorize.py`: Low-level SVD, QR, EIG function tests
- Enhanced test clarity with descriptive names and documentation
- Renamed "flip" tests to "invert" for consistency with API changes

#### Project Hygiene
- Cleaned up `.gitignore` for better project management
- Added `.logging/` directory to gitignore for Nicole logging
- Improved code formatting consistency in tensor summary outputs

### Statistics and Scope

#### Code Changes
- 27 commits focused on QR decomposition, API enhancements, and documentation
- New function: `qr()` in `decomp.py` (~200 lines)
- Enhanced: `decomp()` with QR mode support
- Enhanced: `oplus()` with flexible axis specification
- New test file: `test_factorize.py` with comprehensive factorization tests

#### Documentation Expansion
- 3 new documentation pages: `qr.md`, `eig.md`, hooks implementation
- Updated 15+ existing documentation pages for PyTorch transition
- Added 2 comprehensive guides: GPU acceleration and autograd
- Enhanced API index with clearer categorization

#### Test Coverage
- Added 10+ new tests for QR decomposition
- Enhanced 5+ tests for oplus functionality
- Maintained 100% pass rate across CPU, CUDA, and MPS devices
- Stress tests validate correctness for high-order tensors

### Rationale

Version 0.2.1 completes the tensor decomposition toolkit with QR factorization, providing researchers with orthogonal decomposition capabilities essential for canonical forms in tensor network algorithms. The flexible axis specification in `oplus()` improves code ergonomics for direct sum operations common in symmetry-adapted basis constructions. The comprehensive documentation updates ensure users can effectively leverage the new PyTorch backend, GPU acceleration, and autograd features introduced in v0.2.0.

The MkDocs hooks enhancement addresses a long-standing rendering issue with bullet lists in parameter tables, ensuring professional-quality documentation that properly displays multi-line parameter descriptions with inline code formatting.

**Breaking Changes**: None - fully backward compatible with v0.2.0 API.

**Target Users**: Researchers in quantum many-body physics and tensor network methods requiring orthogonal decompositions, improved API ergonomics, and comprehensive documentation for PyTorch-based workflows.

---

## [0.2.0] - 2026-02-06

**PyTorch Backend Migration with Autograd and Device Management**

Release version 0.2 of Nicole, introducing a major backend migration from NumPy to PyTorch, enabling automatic differentiation, GPU acceleration, and enhanced device management for tensor network computations with Abelian symmetries.

### Backend Migration - NumPy to PyTorch

#### Core Infrastructure Changes
- Complete migration from NumPy to PyTorch as the tensor backend
- All tensor operations now leverage PyTorch's optimized kernels
- Backward compatibility maintained for existing user code
- Updated dependencies: `torch>=2.5` replaces `numpy>=2.0` as primary backend
- Preserved block-sparse semantics with PyTorch tensors

### Autograd Support

#### Gradient Tracking
- Added `requires_grad` property for gradient computation control
- Automatic differentiation through all tensor operations
- `Tensor.backward()` method for scalar tensors (0D)
- Full computational graph support for optimization workflows
- Element-wise operations (add, sub, mul) preserve gradient flow
- Contraction and decomposition operations support autograd

#### Gradient Management
- `requires_grad` parameter in constructors (`zeros`, `random`, `from_scalar`)
- Setter for `requires_grad` to enable/disable gradient tracking
- Integration with PyTorch's autograd engine
- Access to gradients via underlying `torch.Tensor` blocks
- Default: gradients disabled (`torch.set_grad_enabled(False)`) for performance

### Device Management

#### Multi-Device Support
- **CPU**: Full dtype support (`float32`, `float64`, `complex64`, `complex128`)
- **CUDA (NVIDIA)**: Full dtype support with optimal GPU performance
- **MPS (Apple Silicon)**: `float32`/`complex64` with automatic dtype normalization

#### Device Operations
- `Tensor.device` property for querying tensor placement
- `Tensor.to(device)` method for device transfer
- `Tensor.cpu()` convenience method
- `Tensor.cuda()` convenience method
- `device` parameter in constructors (`zeros`, `random`, `from_scalar`)
- Automatic device consistency validation in operations

#### MPS Dtype Normalization
- `normalize_dtype_for_device()` utility function in typing module
- Automatic `float64` → `float32` conversion on MPS
- Automatic `complex128` → `complex64` conversion on MPS
- Transparent handling in constructors and `.to()` method
- Comprehensive test coverage for MPS compatibility

### Testing Infrastructure

#### Comprehensive Test Coverage
- **708 tests** covering all functionality (up from 662 in v0.1)
- New test modules: `test_autograd.py`, `test_device.py`
- MPS dtype normalization tests integrated into `test_device.py`
- Device management tests for CPU, CUDA, MPS
- Autograd tests for gradient computation and `backward()`
- Gradient flow tests for operations (add, sub, mul, contract)
- All existing tests updated for PyTorch backend

#### Test Organization
- Device tests in `tests/support/test_device.py` (new)
- Autograd tests in `tests/support/test_autograd.py` (new)
- GPU tests skip gracefully when hardware unavailable
- MPS-specific tests for dtype normalization

### Implementation Highlights

#### Backend Changes
- Replaced numpy arrays with torch tensors throughout codebase
- Updated `torch.randn()` for random generation with generator support
- `torch.eye()` for identity matrices
- `torch.zeros()` for zero initialization
- `torch.complex()` for complex number construction
- Maintained block-sparse structure with PyTorch tensors

#### Performance Optimizations
- Disabled autograd by default (`torch.set_grad_enabled(False)`)
- Set default device to CPU (`torch.set_default_device('cpu')`)
- Efficient device transfers with minimal overhead
- GPU acceleration for large-scale computations
- Block-sparse algorithms unchanged, now with PyTorch backend

### API Surface Updates

- **Core (enhanced)**: `Tensor.requires_grad`, `Tensor.backward()`, `Tensor.device`, `Tensor.to()`, `Tensor.cpu()`, `Tensor.cuda()`
- **Utilities (new)**: `normalize_dtype_for_device()`
- **Constructors (enhanced)**: `device` and `requires_grad` parameters
- **Operations**: All operations now support autograd and device management

### Statistics and Scope

#### Code Changes
- 161 commits across develop branch
- 17 files changed: 195 insertions, 56 deletions
- Major refactors: `tensor.py`, test suite updates
- New helper functions: `normalize_dtype_for_device()` in typing module

#### Test Coverage
- 708 comprehensive tests (46 new tests since v0.1)
- 16 device management tests (including MPS)
- 21 autograd tests for gradient computation
- All tests pass on CPU, CUDA, and MPS devices

### Development Workflow

Version 0.2 represents a major evolution of Nicole, transitioning from a pure NumPy library to a PyTorch-powered framework. This migration unlocks critical capabilities for modern tensor network research:

- Automatic differentiation for variational algorithms (variational MPS, PEPS optimization)
- GPU acceleration for large-scale simulations
- Seamless integration with the broader PyTorch ecosystem
- Apple Silicon (MPS) support for Mac users

Despite the significant backend change, the migration maintains full API compatibility with v0.1.x, ensuring existing user code continues to work without modification. The enhanced testing suite validates correctness across all devices and operations.

**Breaking Changes**: None - fully backward compatible with v0.1.x API. Internal backend changed from NumPy to PyTorch, but user-facing API unchanged.

**Target Users**: Researchers in quantum many-body physics, machine learning, and quantum information who require GPU acceleration, automatic differentiation, or modern optimization workflows for tensor network methods.

---

## [0.1.1] - 2026-02-01

**Documentation and Developer Experience Release**

Release version 0.1.1 of Nicole, introducing complete documentation infrastructure with live code execution, comprehensive API references, extensive examples, and enhanced developer experience for tensor network computations with Abelian symmetries.

### Documentation Infrastructure

#### MkDocs Configuration with Material Theme
- Professional documentation site with modern Material Design
- Configured navigation with hierarchical structure (Getting Started, Examples, API Reference)
- Custom branding with Nicole logo (nicole-font-awesome.png, 1.6 MB)
- Responsive design for desktop and mobile viewing
- Search functionality with indexed content
- Dark/light theme switching with system preference support

#### Live Code Execution (markdown-exec)
- All code examples execute automatically during documentation build
- Real-time output generation ensures documentation accuracy
- Session-based execution for shared imports across code blocks
- Source code and console output displayed side-by-side using material-block format
- ANSI color support for enhanced terminal output visualization
- Pyodide integration for future web-based interactive examples

#### Documentation Plugins and Extensions
- mkdocstrings: Automatic API documentation from Python docstrings with NumPy style
- git-revision-date-localized: Last modified timestamps on each page
- git-committers: Contributor tracking and author information
- pymdownx.arithmatex: LaTeX math rendering via MathJax
- pymdownx.superfences: Enhanced code blocks with syntax highlighting
- pymdownx.tabbed: Tabbed content for alternative implementations
- pymdownx.emoji: Icon support with Material Design and FontAwesome

### API Reference Documentation

**45+ documentation pages covering:**

- **Core Concepts**: Tensor, Index, Sector, Direction with detailed explanations
- **Symmetry Groups**: U1Group, Z2Group, ProductGroup with mathematical foundations
- **Operations**: contract, trace, decomp, svd with comprehensive examples
- **Arithmetic**: Addition, subtraction, oplus, diag, inv with sector handling
- **Creation Functions**: identity, isometry, isometry_n, random, zeros
- **Manipulation**: retag, subsector, merge_axes, flip, permute, transpose, conjugate
- **Utilities**: load_space, blocks, display with usage patterns

### Example Documentation with Live Execution

**20+ pages of executable examples:**

#### Basic Examples
- Creating Your First Tensor: Introduction to Index, Sector, block structure
- Arithmetic Operations: Addition, subtraction, norms with symmetries
- Indexing: Block access, sector filtering, index properties, trivial indices

#### Symmetry Examples
- U(1) Examples: Particle number conservation, multi-particle states, Fock space
- Z(2) Examples: Fermion parity, parity operators, Jordan-Wigner strings
- Product Group Examples: Charge-spin systems, SU(2) via U(1)⊗U(1), multi-quantum numbers

#### Operations Examples
- Contraction Examples: Matrix multiplication, multi-index contractions, MPS-like patterns, trace operations
- Decomposition Examples: SVD for entanglement, truncation strategies, UR/LV decompositions
- Manipulation Examples: Index reordering, axis merging, conjugation, retag workflows

#### Advanced Examples
- Build Operators: Custom operator construction (identity, number, ladder, spin)
- Load Space: Physical system presets with spherical tensor conventions
- Performance Tips: Memory optimization, computational efficiency, profiling strategies

### Getting Started Guide

**7 pages of comprehensive introduction:**

- **What is Nicole**: Philosophy, design principles, target audience
- **Installation**: pip installation, development setup, dependency management
- **Core Concepts**: Symmetries, sectors, blocks, charge conservation explained
- **Quick Examples**: Complete workflows from tensor creation to decomposition
- **Contributing**: Guidelines for community contributions and development practices
- **Changelog**: Version history and release notes

### Configuration and Build System

#### Build System Migration
- Migrated from setuptools to Hatchling for modern Python packaging
- Simplified build configuration with cleaner pyproject.toml structure
- Added project URLs: homepage, documentation, repository, issues
- Removed setuptools-specific configuration sections

#### Documentation Dependencies
- mkdocs>=1.5, mkdocs-material>=9.5: Core documentation framework
- mkdocstrings[python]>=0.24: API documentation generator
- markdown-exec[ansi]>=1.12: Live code execution with ANSI support
- mkdocs-git-revision-date-localized-plugin>=1.2: Date tracking
- mkdocs-git-committers-plugin-2>=2.0: Contributor information

### Visual Branding

- nicole-font-awesome.png (1.6 MB): Navigation header logo with FontAwesome styling
- Consistent branding across documentation site
- Professional visual identity for the library
- Custom CSS styling (extra.css) for enhanced presentation

### Statistics

- **57 files changed**, 4,663 lines added
- **45+ documentation pages** across Getting Started, Examples, API Reference
- **100+ code examples** with live execution
- **20+ API reference pages** with function signatures and descriptions
- Complete coverage of all core classes, operations, and utilities

### Rationale

Version 0.1.1 focuses on documentation and developer experience, addressing the critical need for comprehensive, accessible documentation as Nicole gains users. The live code execution via markdown-exec ensures all examples are accurate, executable, and up-to-date, eliminating documentation drift that plagues many scientific libraries. The complete API references, extensive practical examples, and clear getting-started guides significantly lower the barrier to entry for new users while providing depth for advanced use cases. The professional documentation infrastructure with modern tooling (MkDocs Material, mkdocstrings, live execution) establishes Nicole as a mature, well-maintained library suitable for research and production use in the quantum physics and tensor network communities.

### Breaking Changes

None - fully backward compatible with v0.1.0

---

## [0.1.0] - 2026-01-26

**Initial stable release of Nicole Tensor Library**

Release the first stable version of Nicole, a Python library for block-sparse tensor computations with Abelian symmetries, designed for tensor network algorithms in quantum many-body physics.

### Core Features

#### Symmetry-Aware Tensor Framework
- Tensor class with automatic block-sparse structure for Abelian symmetries
- Support for U(1) and Z(2) symmetry groups with automatic charge conservation
- ProductGroup implementation for direct products of multiple Abelian symmetries
- Efficient memory usage through block-sparse representation
- Index class with directional quantum number flow (IN/OUT)
- Flexible index tagging system (itags) for intuitive tensor operations

#### Tensor Operations
- **Contraction**: np.tensordot-style interface with automatic index pairing
    - Flexible axes specification with exclusion support
    - Automatic ambiguity detection and validation
    - Full support for scalar tensors (0D)
- **Trace**: Automatic pairing with exclusion options
- **Arithmetic**: Element-wise addition, subtraction with sector union
- **Manipulation**: permute, transpose, conjugate, retag, merge_axes, flip
- **Block access**: Efficient block extraction with getsub/subsector

#### Tensor Decomposition
- SVD with symmetry preservation and truncation support
    - Bond dimension (chi) and singular value (tol) truncation
    - Automatic index direction handling
- Eigenvalue decomposition for symmetric tensors
- High-level decomp function with customizable truncation
- Multi-axis decomposition support

#### Tensor Construction
- Identity tensors with automatic sector matching
- Isometry for index fusion (isometry, isometry_n for multi-index)
- Random tensor generation with symmetry constraints
- Zero tensors for initialization
- Direct sum operation (oplus) for selective axis merging

#### Specialized Operators
- `diag`: Diagonal matrix tensor construction
- `inv`: Tensor inversion with automatic index flipping
- Operator arithmetic for quantum systems

#### Quantum Many-Body Systems (load_space)
- Preset-based Hilbert space construction with physical operators:
    - **Ferm**: Spinless fermions with U(1) charge conservation
    - **FermU1U1/FermZ2U1**: Spinful fermions with spin and charge symmetries
    - **Band**: Hardcore bosons with U(1)⊗U(1) or Z(2)⊗U(1) symmetries
    - **Spin**: Spin-1/2 chains with U(1) or Z(2) symmetries
- Automatic operator generation (creation, annihilation, number, spin)
- Charge conventions optimized for half-filling calculations
- Spherical tensor convention for spin operators
- Vacuum sector support for open boundary conditions

#### Utilities and Display
- Comprehensive tensor summary with customizable block display
- Index summary for debugging and inspection
- Sector pruning for zero-dimension removal
- Block enumeration and manipulation tools
- Trim zero sectors functionality

### Documentation and Testing
- **662 comprehensive tests** covering all functionality
- Test suite organized by feature area
- Documentation infrastructure using MkDocs with Material theme
- Enhanced README with project overview and contributing guidelines
- Visual branding with Nicole logo

### Implementation Highlights
- Pure Python implementation with NumPy backend
- Type hints throughout for better IDE support
- Modular architecture: symmetry, tensor, operators, decomposition, contraction
- Efficient block-sparse algorithms with automatic charge validation
- Sector pruning to maintain minimal representation
- Direction-aware charge contributions for index operations

### API Surface

**Core Classes:**
- `Tensor`, `Index`, `Direction`

**Symmetry Groups:**
- `U1Group`, `Z2Group`, `ProductGroup`, `AbelianGroup` (base)

**Operations:**
- `contract`, `trace`, `decomp`, `svd`, `eig`

**Manipulation:**
- `permute`, `transpose`, `conjugate`, `retag`, `merge_axes`, `flip`

**Construction:**
- `identity`, `isometry`, `isometry_n`, `random`, `zeros`

**Operators:**
- `diag`, `inv`, `oplus`

**Utilities:**
- `load_space`, `blocks`, `tensor_summary`, `index_summary`, `subsector`

**Types:**
- `Sector`, `SectorPair`, `GroupElem` (type aliases)

### Statistics

- **38 files**: 19 source modules, 18 test modules, 1 utility module
- **~18,000 lines of code** added
- **Core modules**: `tensor.py` (766 lines), `decomp.py` (784 lines), `space.py` (802 lines), `operators.py` (830 lines), `contract.py` (639 lines)
- **Comprehensive test coverage**: `test_decomp.py` (2193 lines), `test_manipulation.py` (1638 lines), `test_contract.py` (1197 lines), `test_construction.py` (1030 lines), `test_space.py` (970 lines)

### Development History

- **119 commits** across 8 feature branches
- **Major features**: product groups, scalar tensors, oplus, easy decomposition, isometry_n, contract syntax, load_space, documentation
- Extensive stress testing and edge case coverage
- Continuous refinement of API and conventions

### Rationale

Version 0.1 represents the first stable release of Nicole, providing a solid foundation for tensor network computations with Abelian symmetries. The library has been thoroughly tested and is ready for use in quantum many-body physics applications, including DMRG, TEBD, PEPS, and related algorithms. The combination of intuitive API design, comprehensive symmetry support, and efficient block-sparse implementation makes Nicole a powerful tool for the tensor network community. This release establishes the core functionality and conventions that will guide future development.

### Target Users

Researchers and students in quantum many-body physics, condensed matter theory, and quantum information who work with tensor network methods and require efficient handling of Abelian symmetries.

---

[0.2.1]: https://github.com/Ideogenesis-AI/Nicole/releases/tag/v0.2.1
[0.2.0]: https://github.com/Ideogenesis-AI/Nicole/releases/tag/v0.2.0
[0.1.1]: https://github.com/Ideogenesis-AI/Nicole/releases/tag/v0.1.1
[0.1.0]: https://github.com/Ideogenesis-AI/Nicole/releases/tag/v0.1.0
