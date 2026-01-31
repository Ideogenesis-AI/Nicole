# Changelog

All notable changes to Nicole will be documented in this file.

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

[0.1.0]: https://github.com/Ideogenesis-AI/Nicole/releases/tag/v0.1.0
