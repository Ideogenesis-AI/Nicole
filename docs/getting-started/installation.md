# Installation

Nicole is a pure-Python package distributed on PyPI. It installs like any other scientific Python library: no compiler, no CUDA toolkit, and no build step are required, because all heavy numerical work is delegated to PyTorch. This guide covers the user install, the editable install used for development, GPU setup, and the problems that come up most often.

## Requirements

| Requirement | Version | Notes |
| --- | --- | --- |
| Python | 3.11 – 3.15 | Tested on CPython; 3.11 is the minimum supported release |
| [PyTorch](https://pytorch.org) | 2.5 or higher | Provides the dense block backend and GPU support |
| [NumPy](https://numpy.org) | 2.0 or higher | Used for interoperability and array conversion |
| [Yuzuha](https://ideogenesis-ai.github.io/Yuzuha/) | 0.1.5 or higher | Supplies Clebsch-Gordan data for non-Abelian groups |

All three dependencies are installed automatically with Nicole, so the commands below are all you need.

## Installing from PyPI

=== "pip"

    ```bash
    pip install nicole
    ```

=== "uv"

    ```bash
    uv add nicole
    ```

To upgrade an existing installation, run `pip install --upgrade nicole` (or `uv lock --upgrade-package nicole`).

### Optional Dependencies

Nicole ships three optional dependency groups, none of which are needed to use the library:

| Group | Installs | Use for |
| --- | --- | --- |
| `test` | `pytest`, `pytest-cov`, `pytest-benchmark`, `hypothesis` | Running the test suite and benchmarks |
| `lint` | `ruff`, `mypy` | Linting and static type checking |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings` and plugins | Building this documentation site |

Select them with the usual extras syntax, either one at a time or combined:

```bash
pip install "nicole[test]"
pip install "nicole[test,docs,lint]"
```

## Verifying Installation

Run the following to confirm that Nicole, its backend, and symmetry machinery all work together:

```python
import torch
import nicole
from nicole import Tensor, Index, Sector, Direction, U1Group

print(f"Nicole version:  {nicole.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Default device:  {torch.get_default_device()}")

group = U1Group()
index = Index(Direction.OUT, group, sectors=(Sector(charge=0, dim=2),))
tensor = Tensor.random([index, index.flip()], itags=["i", "j"], seed=42)
print(tensor)
```

The last line prints a block-sparse tensor summary that looks like this:

```
  info:  2x { 1 x 1 }  having 'A',   Tensor,  { i*, j }
  data:  2-D float64 (32 B)    2 x 2 => 2 x 2  @ norm = 0.48783

     1.  2x2     |  1x1     [ 0 ; 0 ]    32 B
```

If you see a tensor summary rather than a traceback, the installation is complete. The [Quick Examples](quick-examples.md) page explains how to read this output in detail.

## Installing from Source

Install from source if you want the latest development version or intend to contribute:

=== "pip"

    ```bash
    git clone https://github.com/Ideogenesis-AI/Nicole.git
    cd Nicole
    pip install -e ".[test,docs,lint]"
    ```

=== "uv"

    ```bash
    git clone https://github.com/Ideogenesis-AI/Nicole.git
    cd Nicole
    uv sync --all-extras
    ```

The editable install means edits under `src/nicole/` take effect immediately, without reinstalling. Day-to-day commands once the environment is ready:

```bash
pytest                  # run the test suite
ruff check src/         # lint
mypy src/               # type check
mkdocs serve            # preview the documentation at http://127.0.0.1:8000
```

Development on Nicole starts from the `develop` branch; see [Contributing](contributing.md) for the branch workflow, code style, and pull request guidelines.

## GPU Support

Nicole runs on any device PyTorch supports, including NVIDIA GPUs through CUDA and Apple Silicon through MPS. Nicole itself needs no extra configuration — it follows whichever device PyTorch is set up for.

**NVIDIA (CUDA)**: the default PyPI wheels of PyTorch already bundle CUDA on Linux. If you need a build for a specific CUDA version, install it from the PyTorch index and pick the tag matching your driver (see [PyTorch's install selector](https://pytorch.org/get-started/locally/) for the current list):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Apple Silicon (MPS)**: MPS support is part of the standard macOS wheel, so nothing extra is required:

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

For moving tensors between devices and the performance trade-offs involved, see the [GPU Acceleration](../examples/advanced/gpu-acceleration.md) guide.

## Troubleshooting

### Wrong Python Version

Nicole supports Python 3.11 through 3.15. Check which version you are running:

```bash
python --version
```

If your system has several versions installed, `python` may not point at the one you expect, so name the interpreter explicitly, for example `python3.11` or `python3.15`.

### Outdated PyTorch

An `ImportError` or an unexpected failure in a tensor operation often means PyTorch is older than 2.5:

```bash
python -c "import torch; print(torch.__version__)"
pip install --upgrade torch
```

### Outdated Yuzuha

SU(2) and other non-Abelian groups rely on Clebsch-Gordan data from Yuzuha. If constructing a non-Abelian `Index` or contracting such tensors raises an unexpected error, upgrade Yuzuha to 0.1.5 or higher:

```bash
pip install --upgrade yuzuha
```

Yuzuha stores computed Clebsch-Gordan coefficients in a local cache directory, so the first non-Abelian computation in a fresh environment is slower than subsequent ones. See the [SU(2) Protocol](yuzuha-protocol.md) page for background.

## Next Steps

Once installed, continue with:

- **[Core Concepts](core-concepts.md)**: Understand the fundamentals of symmetry-aware tensors
- **[Quick Examples](quick-examples.md)**: Create your first tensors and learn basic operations
