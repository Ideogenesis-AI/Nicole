# Changelog

All notable changes to Nicole will be documented in this file.

## [0.3.2] - 2026-04-07

**Einstein Summation and Tensor Serialization**

Version 0.3.2 introduces two significant new capabilities: `einsum` brings Einstein summation notation to symmetry-aware tensors, and `serialize`/`deserialize` enable portable, `torch.save`-compatible persistence of Tensor instances including their SU(2) intertwiner weights. The release also cleans up several API inconsistencies — most notably the simplification of `transpose`, the standardization of block-identifier terminology, the refactoring of compression logic into `BlockSchema.block_compress`, and the removal of the now-redundant `Tensor.compress`.

### New Features

#### `einsum` — Einstein Summation Notation

A new `einsum(equation, *tensors)` function parses a subscript equation string and dispatches to `contract`, `trace`, and `permute` to carry out the requested operation.

Three equation forms are supported:

- **Permutation** — single input tensor with a reordered output subscript, e.g. `'ij->ji'`
- **Trace** — single input tensor with one repeated subscript letter, e.g. `'ii->'`; surviving axes are permuted as needed
- **Sequential contraction** — two or more tensors contracted left-to-right, e.g. `'ij,jk->ik'`; tensors are contracted pairwise in order (no contraction-order optimization)

`einsum` supports both Abelian and SU(2) symmetry groups and is exposed as a top-level public API function.

#### `serialize` / `deserialize` — Tensor Persistence

A new `serialize.py` module provides `serialize(tensor)` and `deserialize(payload, device=None)` for lossless round-trip conversion of Tensor instances to and from plain Python dicts.

The serialized payload:

- Uses only Python primitives (`str`, `int`, `tuple`, `dict`, `None`) and `torch.Tensor` values
- Is directly compatible with `torch.save` / `torch.load(..., weights_only=True)`
- Encodes the full block-sparse data structure, index metadata (direction, symmetry group, sectors), and SU(2) intertwiner weights
- Carries a `"version"` key for forward-compatibility

`Bridge` serialization is also supported internally. Both `serialize` and `deserialize` are exposed as top-level public API functions.

#### `BlockSchema.block_compress` — Automatic Component Compression

A new static method `BlockSchema.block_compress(data, bridge, cutoff=1e-14)` removes linearly dependent components from a single non-Abelian tensor block via thin SVD on the Bridge weight matrix. Singular vectors whose singular values fall below `cutoff` are discarded; the retained factors are absorbed back into the reduced data so that the physical block `R @ W` is exactly preserved.

`block_compress` is now called automatically by:

- `Tensor.regularize` — after the normalization pass (cutoff forwarded via a new `cutoff` kwarg)
- `Tensor.__add__` and `Tensor.__sub__` — per block after `block_add`, to reclaim rank deficiencies that arise when operands carry identical or collinear components
- `oplus` — same treatment as addition and subtraction
- `contract` and `trace` — via the now unconditional `regularize()` call that replaces the previous conditional `compress()`

### API Changes

- **`transpose` simplified**: `transpose` no longer accepts a positional `*order` argument; it now unconditionally reverses all tensor axes. Callers that previously passed an explicit axis ordering should use `permute` instead. Applies to both `maneuver.transpose(tensor)` and `Tensor.transpose(in_place=False)`.
- **`Tensor.compress` removed**: `Tensor.compress()` has been removed from the public API. Its logic now lives in `BlockSchema.block_compress` and is invoked automatically through `regularize`. Callers should use `Tensor.regularize()` instead, which now covers both normalization and compression.
- **`block_ids` terminology**: Block-identifier parameter names standardized across the display layer — `tensor_summary`: `block_numbers` renamed to `block_ids`; `Tensor.show`: `block_indices` renamed to `block_ids`.

### Documentation

#### New API Pages

- **`einsum`**: Full reference with equation syntax, supported forms, and worked examples for Abelian and SU(2) tensors
- **`serialize`**: Reference documentation with the full serialized dict schema and `torch.save` / `torch.load` usage examples
- **`deserialize`**: Reference documentation including the `device` argument and version-error behavior

#### New and Extended Examples

- **Serialization examples**: New advanced examples page demonstrating round-trip persistence with `torch.save` and `torch.load`
- **Contraction examples**: Extended with `einsum` equation examples covering permutation, trace, and multi-tensor contractions
- **Examples index**: New entries for Autograd, GPU Acceleration, and Serialization

#### Other Updates

- **`transpose` documentation**: Updated to reflect the removed `order` argument and the new always-reverse semantics
- **`identity` documentation**: Corrected namespace reference and removed outdated notes
- **Display documentation**: Enhanced with additional details on Nicole's tensor summary format
- **SU(2) Protocol page**: Added a references section

### Test Suite (1521 tests)

- **1521 tests pass**, 10 skipped (accelerator-only tests on CPU-only CI)
- New test module: `tests/operations/test_einsum.py` (39 tests) covering permutation, trace, and sequential contraction for Abelian and SU(2) tensors, including higher-order multi-tensor equations
- Serialization tests added to `tests/support/test_helpers.py` (11 new tests): round-trip for U(1), Z2, product-Abelian, SU(2), and product-SU(2) tensors; complex dtype; scalar tensors; `torch.save` / `torch.load` integration; `device` forwarding; unknown-version error
- Bridge serialization tests added to `tests/symmetry/test_delegate.py` (7 new tests)
- Outer product consistency tests added to `tests/integration/test_consistency.py` (8 new tests) for U(1) and SU(2) groups
- `tests/operations/test_maneuver.py`: transpose tests rewritten to reflect the simplified signature
- `tests/primary/test_blocks.py`: five new tests for `block_compress` covering single-component no-op, independent rows unchanged, dependent rows reduced, cutoff sensitivity, and input immutability
- `tests/support/test_helpers.py`: four new tests for the compression aspect of `Tensor.regularize`; existing tests updated to use physical-tensor comparison where SVD sign ambiguity would break strict weight equality
- `tests/operations/test_arithmetic.py` and `test_oplus.py`: assertions updated to compare physical content `R @ W` rather than raw component counts; oplus weight tests upgraded to 4-index spin-1 tensors where component count assertions remain meaningful

### Statistics

- **65 commits** since v0.3.1
- **46 files changed**: 3,187 insertions, 705 deletions
- New source modules: `src/nicole/einsum.py`, `src/nicole/serialize.py`
- Source modules touched: `tensor.py`, `blocks.py`, `contract.py`, `display.py`, `maneuver.py`, `symmetry/delegate.py`, `__init__.py`
- New test module: `tests/operations/test_einsum.py`

### Compatibility

**Breaking Changes:**

- `transpose(tensor, *order)` / `Tensor.transpose(*order, in_place)` — the `order` argument is removed; use `permute` for custom axis orderings
- `tensor_summary(..., block_numbers=...)` → `tensor_summary(..., block_ids=...)`
- `Tensor.show(block_indices=...)` → `Tensor.show(block_ids=...)`
- `Tensor.compress()` — removed; use `Tensor.regularize()` instead

All other changes are backward compatible with v0.3.1.

**Requirements:** Python ≥ 3.11, PyTorch ≥ 2.5, Yuzuha ≥ 0.1.5

---

## [0.3.1] - 2026-03-31

**Documentation and GPU Enhancements**

Patch release delivering comprehensive documentation for the SU(2) features introduced in v0.3.0, new Getting Started content, and end-to-end GPU device propagation across all Abelian and SU(2) tensor operations.

### Documentation

#### New Getting Started Pages

- **Landing page** with hero section for the documentation site
- **"Why Nicole?"** page comparing Nicole against TensorKit, ITensor, and QSpace; includes a "Why Python over Julia?" section on the PyTorch ecosystem and AI coding-agent compatibility
- **"Terminology"** page defining Axis / Index / Edge, Sector / Block, and reverse / flip / invert / fuse / combine / merge

#### New SU(2) API Reference Pages

- `SU2Group` — full reference including Wigner-Eckart and R-W-C decomposition details
- `Bridge` — complete reference for intertwiner manipulation methods
- `capcup` — bond inversion documentation with warnings on the distinction from `Tensor.invert()`
- `filter_blocks` — replaces the old `subsector` page
- `ProductGroup` — extended with non-Abelian examples for `fuse_channels` and `irrep_dim`
- `Tensor` members extended: `normalize_sectors`, `compress`, `regularize`
- Symmetry overview split into Abelian and non-Abelian subsections
- `load_space` examples extended with U(1)×SU(2) and Z2×SU(2) walkthroughs and state-convention admonitions

#### Accuracy and Terminology Fixes (25+ pages)

- "n-leg tensor" → "nth-order tensor"; "tensor leg" → "tensor index / tensor axis"
- "matrix elements" → "tensor elements" in non-second-order contexts
- `fuse()` → `fuse_unique()` across all Abelian examples and API pages
- Manipulation examples rewritten to demonstrate the chainable method-based API (`T.conj()`, `T.permute()`, `T.transpose()`)

### GPU Device Propagation

Explicit `device=` arguments are now forwarded through every layer of the computation graph, so no intermediate tensor silently falls back to CPU:

- **`Bridge` and CG symbols**: `Bridge.from_block`, `compute_xsymbol`, `compute_rsymbol`, and `Bridge.permute` all propagate device and dtype
- **`identity`, `isometry`, `isometry_n`**: Accept and forward a `device` keyword argument
- **`oplus`, `diag`, `merge_axes`**: Forward device to all internal tensor allocations
- **`svd`, `qr`, `eig`, `decomp`**: Forward `device=T.device` to `Bridge.from_block`
- **`trace`**: Forwards `device=T.device` to `torch.full`
- **`Tensor.zeros`, `Tensor.random`, `regularize`**: Device forwarded to `Bridge.from_block` and scalar allocations
- **`load_space`**: New `_get_device(option)` helper; all 8 operator construction functions call `.to(device)` on output tensors
- **`BlockSchema`**: Collinearity check pins intermediate tensors to the correct device
- **31 new integration tests** in `tests/integration/test_propagation.py` covering all operations and all 10 `load_space` presets for both Abelian and SU(2) groups; 27 pass unconditionally, 4 skipped without MPS

### API Changes

- **`filter_blocks`**: `subsector` renamed to `filter_blocks` in `maneuver.py` and removed from public exports; update call sites accordingly
- **`Tensor.normalize_sectors()`**: New public method to prune unused sectors from tensor indices; `__str__` and `print` now use pruned indices for cleaner summaries
- `load_space` fermionic operators migrated from internal `_prune_unused_sectors` to `normalize_sectors`

### Display Improvements

- SU(2) tensor blocks with a single-value weight matrix now display a sign indicator (`+`/`-`) in the block summary for quick inspection of Wigner-Eckart reduced matrix elements

### Code Quality

- `Tensor.to()`, `Tensor.cpu()`, `Tensor.cuda()`, and `Tensor._align_for_binary()` now use concrete `Tensor` return type annotations instead of forward-reference strings

---

## [0.3.0] - 2026-02-15

**SU(2) Non-Abelian Symmetry Release**

Major release introducing full SU(2) non-Abelian symmetry support through an intertwiner-based reduced tensor algebra, backed by the [yuzuha](https://github.com/Ideogenesis-AI/yuzuha) Clebsch–Gordan engine.

### New: SU(2) Symmetry Group

#### `SU2Group`
- New `SU2Group` class in `nicole.symmetry` (and re-exported at top level)
- Charges are non-negative integers using the **2j convention**: `0, 1, 2, 3, ...` for spins `0, 1/2, 1, 3/2, ...`
- `irrep_dim(2j)` returns `2j + 1` (dimension of the spin-j multiplet)
- `fuse_channels(*two_js)` returns all achievable total spin channels via the triangular inequality
- All representations are self-dual: `dual(2j) = 2j`
- `is_abelian` property: `False` for `SU2Group`

#### `ProductGroup` extended
- `ProductGroup` now accepts `SU2Group` as the **last** component, e.g. `ProductGroup([U1Group(), SU2Group()])`
- `is_abelian` returns `False` for any `ProductGroup` containing `SU2Group`
- Tuple charges: `(n, 2j)` for U(1) × SU(2); `(p, 2j)` for Z(2) × SU(2)

### New: Intertwiner Engine

#### `Bridge` class (yuzuha integration)
- `Bridge` in `nicole.symmetry.delegate` stores Clebsch–Gordan tensors (intertwiners) for each data block in an SU(2) tensor
- Handles outer-multiplicity dimensions, Frobenius–Schur phases, and direction conventions
- `Bridge.conj()`, `Bridge.clone()`, `Bridge.to(device)`, `Bridge.insert_edge()`, `Bridge.invert_edges()`

#### `Tensor.intw` field
- Non-Abelian tensors carry a per-block intertwiner dictionary `{BlockKey: Bridge}`
- Intertwiners are automatically managed by all operations; users rarely need to access them directly

#### `capcup` function (new top-level export)
- `capcup(A, axis_a, B, axis_b)` inverts a bond direction between two tensors, inserting Frobenius–Schur phase corrections required for SU(2)

### SU(2) Support in All Operations

All existing operations now handle SU(2) and SU(2)-containing `ProductGroup` transparently:

| Operation | SU(2) behaviour |
|-----------|----------------|
| `contract` | Uses X/R symbols for intertwiner algebra |
| `trace` | Enforces SU(2) selection rules for vanishing contributions |
| `identity` | CG-weighted 2-index delta with Bridge intertwiners |
| `isometry`, `isometry_n` | Fusion isometries with multi-channel Bridge |
| `conj` | Flips intertwiner edge directions via `Bridge.conj()` |
| `permute`, `transpose` | Updates intertwiners with R-symbol corrections |
| `merge_axes` | Non-Abelian axis fusion via CG structure |
| `svd`, `qr`, `eig`, `decomp` | Intertwiner-aware factorizations; OM trailing dimension |
| `inv` | Preserves intertwiner field through block inversion |
| `oplus`, `diag` | Non-Abelian direct sum and diagonal construction |
| `filter_blocks` | Clones intertwiner weights for extracted blocks |

### New Tensor Methods

- **`Tensor.regularize()`**: Canonicalizes intertwiner weights to a standard form (identity-like bridges)
- **`Tensor.compress()`**: Removes redundant components from SU(2) tensor blocks, reducing outer multiplicity
- **`Tensor.trim_zero_blocks(epsilon=...)`**: Removes near-zero blocks; works for both Abelian and SU(2) tensors
- **`Tensor.normalize_sectors()`**: Canonicalizes sector ordering across Abelian and non-Abelian tensors

### `Index.num_states`

- New `Index.num_states` property returns the total number of physical states, accounting for irrep dimensions: `sum(irrep_dim(q) × dim for each sector)`
- For Abelian indices, `num_states == dim`; for SU(2) indices, `num_states > dim` whenever spin > 0

### `load_space` SU(2) Presets

New symmetry options for `load_space`:

| `preset` | `preserv` | Operators |
|----------|-----------|-----------|
| `"Spin"` | `"SU2"` | `S`, `vac` — rank-1 spherical tensor (reduced tensor element) |
| `"Band"` | `"U1,SU2"` | `F`, `Z`, `S`, `vac` — fermionic annihilation + spin tensor |
| `"Band"` | `"Z2,SU2"` | `F`, `Z`, `S`, `vac` — same with Z2 parity instead of U1 |

### API Changes

- **`permute` default changed**: `in_place` parameter now defaults to `False` (functional style); pass `in_place=True` for in-place behaviour
- **`eig` new parameter**: `is_hermitian` flag (default `False`); set to `True` to use the Hermitian eigensolver for improved numerical stability
- **`is_abelian` property**: Added to all symmetry group classes (`U1Group`, `Z2Group`, `SU2Group`, `ProductGroup`)
- **`irrep_dim` method**: Added to all symmetry group classes; returns 1 for Abelian groups, `2j+1` for SU(2)

### Statistics

- 150+ commits across the `feature/su2-group` and related branches
- SU(2) test coverage added in `test_su2_*` modules, integration tests for Heisenberg, Hubbard, and band models
- All existing Abelian tests continue to pass unchanged

---

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

[0.3.2]: https://github.com/Ideogenesis-AI/Nicole/releases/tag/v0.3.2
[0.3.1]: https://github.com/Ideogenesis-AI/Nicole/releases/tag/v0.3.1
[0.3.0]: https://github.com/Ideogenesis-AI/Nicole/releases/tag/v0.3.0
[0.2.1]: https://github.com/Ideogenesis-AI/Nicole/releases/tag/v0.2.1
[0.2.0]: https://github.com/Ideogenesis-AI/Nicole/releases/tag/v0.2.0
[0.1.1]: https://github.com/Ideogenesis-AI/Nicole/releases/tag/v0.1.1
[0.1.0]: https://github.com/Ideogenesis-AI/Nicole/releases/tag/v0.1.0
