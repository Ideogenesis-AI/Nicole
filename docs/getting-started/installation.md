# Installation

Nicole is a Python library for symmetry-aware tensor computations. This guide covers installation options for both users and developers.

## Installing from PyPI

The recommended way to install Nicole is via pip:

```bash
pip install nicole
```

This will install Nicole and its required dependencies (NumPy 2.0+).

### Optional Dependencies

Install with additional dependencies for specific use cases:

```bash
# For testing
pip install nicole[test]

# For documentation
pip install nicole[docs]

# For linting and development tools
pip install nicole[lint]

# Install all optional dependencies
pip install nicole[test,docs,lint]
```

## Requirements

- **Python**: 3.11 or higher
- **NumPy**: 2.0 or higher

## Installing from Source (Development)

If you want to contribute to Nicole or need the latest development version:

```bash
# Clone the repository
git clone https://github.com/Ideogenesis-AI/Nicole.git
cd Nicole

# Install in development mode
pip install -e .

# Optional: Install with all development dependencies
pip install -e ".[test,docs,lint]"
```

### Development Setup

For active development, install the development dependencies:

```bash
# Install in editable mode with all dependencies
pip install -e ".[test,docs,lint]"

# Run tests
pytest

# Build documentation
cd docs
mkdocs serve

# Run linter
ruff check src/
```

## Verifying Installation

To verify that Nicole is installed correctly:

```python
import nicole
print(f"Nicole version: {nicole.__version__}")

# Create a simple tensor to test
from nicole import Tensor, Index, Sector, Direction, U1Group

group = U1Group()
index = Index(Direction.OUT, group, sectors=(Sector(0, 2),))
tensor = Tensor.random([index, index.flip()], itags=["i", "j"], seed=42)
print(f"Successfully created tensor with {len(tensor.data)} blocks")
```

## Troubleshooting

### NumPy Version Issues

Nicole requires NumPy 2.0 or higher. If you encounter issues:

```bash
# Upgrade NumPy
pip install --upgrade numpy

# Verify NumPy version
python -c "import numpy; print(numpy.__version__)"
```

### Python Version

Ensure you're using Python 3.11 or higher:

```bash
python --version
```

If your system has multiple Python versions, you may need to use `python3.11` or `python3.12` explicitly.

## Next Steps

Once installed, continue with:

- **[Core Concepts](core-concepts.md)**: Understand the fundamentals of symmetry-aware tensors
- **[Quick Examples](quick-examples.md)**: Create your first tensors and learn basic operations
