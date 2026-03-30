# Why Nicole?

Several mature tensor libraries exist for quantum many-body physics. Nicole occupies a distinct position in this landscape: a **Python-native**, **PyTorch-backed** library with **first-class SU(2) support**, designed for researchers who want the rigor of a symmetry-aware tensor framework without leaving the scientific Python ecosystem.

## vs. TensorKit (Julia)

[TensorKit.jl](https://github.com/Jutho/TensorKit.jl) is a mathematically sophisticated library built on symmetric monoidal category theory. Nicole offers a compelling alternative for SU(2) physics:

- **Zero-friction onboarding**: Python's simplicity and ubiquity mean that any researcher familiar with NumPy or PyTorch can start using Nicole immediately — no new language, no unfamiliar toolchain, no steep learning curve.
- **Full Python and ML/AI ecosystem access**: Nicole integrates naturally with NumPy, SciPy, Matplotlib, Hugging Face, and the full ML/AI tech stack. Your tensor network code lives alongside data pipelines, neural network models, and visualization tools — no language boundary, no FFI overhead.
- **GPU acceleration**: Nicole's PyTorch backend runs dense block operations on CUDA and Apple MPS out of the box, making large-scale tensor network computations significantly faster without any code changes.
- **Autograd throughout**: PyTorch's autograd is available end-to-end, enabling gradient-based optimization over tensor network parameters — something not possible in TensorKit.
- **Accessible abstraction**: Nicole's SU(2) support is grounded in the concrete Wigner–Eckart theorem and R-W-C decomposition rather than abstract category theory, making it easier to reason about for physicists.

## vs. ITensor (Julia / C++)

[ITensor](https://itensor.org/) is one of the most widely used tensor network libraries, particularly for DMRG. Nicole brings several key advantages:

- **A first-class Python library**: Nicole is written in pure Python and designed around Python idioms from the ground up. You get full IDE support, clean type hints, and a composable API. ITensor is a C++/Julia library at its core; its Python bindings are a thin secondary interface, not the primary development target.
- **Agile integration with ML/AI tools**: Nicole's PyTorch foundation makes it trivial to connect tensor network computations with neural networks, optimizers, and the broader ML/AI tech stack. ITensor has no comparable integration path.
- **First-class SU(2) support**: SU(2) remains ongoing in ITensor. Nicole provides a complete, production-ready SU(2) implementation with full R-W-C decomposition, giving you direct access to reduced tensor elements, weight matrices, and CG structures.
- **GPU and autograd**: Nicole brings PyTorch-powered GPU execution and gradient support to tensor network computations — capabilities absent from ITensor.
- **Composable by design**: Rather than providing fixed algorithms, Nicole is a composable tensor layer. Any tensor network algorithm — DMRG, TEBD, PEPS updates, or novel methods — can be built directly from Nicole's primitives.

## vs. QSpace (MATLAB)

[QSpace](https://bitbucket.org/qspace4u/) is the direct inspiration for Nicole and remains the gold standard for non-Abelian symmetry support in tensor networks. Nicole brings its rigour into the modern open-source Python world:

- **Open and freely available**: Nicole is released under GPL-3.0. Anyone can use, inspect, and contribute to it — no MATLAB license required, no access request needed.
- **Simple, flexible, and immediately accessible**: Python's readability and minimal boilerplate lower the barrier to entry dramatically compared to MATLAB. Researchers can prototype new ideas, iterate quickly, and share reproducible code with a global community — all within a language they likely already know.
- **The Python and ML/AI ecosystem**: With Nicole, your SU(2) tensor code has immediate access to PyTorch, NumPy, SciPy, and the full ML/AI tech stack. Connecting tensor networks to neural network architectures, generative models, or large-scale optimization is a natural extension, not an engineering project.
- **GPU acceleration**: Nicole runs on CUDA and MPS, enabling large tensor networks that would be impractical on CPU alone.
- **Consistent CG basis via Yuzuha**: QSpace generates its CG basis on the fly, resulting in a history-dependent convention that complicates data sharing across runs and creates obstacles for concurrent computation. Nicole delegates all CG computations to the dedicated [Yuzuha](https://ideogenesis-ai.github.io/Yuzuha/) engine, which enforces a single, deterministic CG convention regardless of evaluation order. This makes tensor data reproducible, portable, and safe to use in parallel workflows.

## Why Python over Julia?

TensorKit and ITensor are Julia libraries, and Julia was designed with a clear goal: combine Python-level expressiveness with C-level performance. That was a reasonable ambition for its time. But the landscape has shifted in ways that erode Julia's core value proposition.

**PyTorch and the ML/AI ecosystem.** Nicole is built on PyTorch — the dominant framework for deep learning and scientific computing, with an enormous global community actively pushing the frontier of hardware support, numerical methods, and algorithmic innovation. Every advance in mixed-precision training, distributed computing, compiler optimisation (via `torch.compile`), or new accelerator support flows directly into Nicole's dense block operations. Julia libraries cannot tap into this engine.

**AI coding agent compatibility.** Modern AI coding agents — including the one you may be using right now — are dramatically more effective with Python than with Julia. Python's vast training corpus, mature tooling, and clear idioms mean that agents can write, review, and debug Nicole code with high reliability. In practice, this translates to faster iteration, fewer errors, and lower barrier to entry for new contributors. Julia's smaller ecosystem and more idiosyncratic compilation model make it significantly harder for coding agents to assist effectively.

**The post-AI era of performance engineering.** Julia's original promise was to resolve the "two-language problem": write fast *and* readable code in one language, avoiding the Python/C++ split. This was a genuine insight — but it rested on the assumption that readable and performant code must coexist in the same language. That assumption no longer holds. AI coding agents make it increasingly practical to write performance-critical kernels in Rust or C++ with much less human effort, while keeping the user-facing API clean and idiomatic Python. The two-language split becomes a *feature*, not a burden: Python for clarity and composability, C++/CUDA or Rust for raw speed. Julia's compromise position — neither the simplest nor the fastest — becomes progressively harder to justify. Nicole embraces this division directly: a clean Python API on top of PyTorch's highly optimised C++/CUDA kernels, and [Yuzuha](https://ideogenesis-ai.github.io/Yuzuha/)'s Rust backend for SU(2) recoupling.

## Summary

| | Nicole | TensorKit | ITensor | QSpace |
|---|:---:|:---:|:---:|:---:|
| Python-native | ✅ | ❌ | ❌ | ❌ |
| Python ecosystem | ✅ | ⚠️ | ⚠️ | ❌ |
| PyTorch / GPU | ✅ | ❌ | ❌ | ❌ |
| Autograd | ✅ | ❌ | ❌ | ❌ |
| SU(2) (Wigner–Eckart) | ✅ | ✅ | ⏳ | ✅ |
| Free platform | ✅ | ✅ | ✅ | ❌ |
| R-W-C decomposition | ✅ | ❌ | ❌ | ✅ |

✅ supported &nbsp;·&nbsp; ❌ not supported &nbsp;·&nbsp; ⚠️ partial &nbsp;·&nbsp; ⏳ in development

If you need rigorous, efficient, GPU-accelerated SU(2) tensor networks in Python, Nicole is the answer.
