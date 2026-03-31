# Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, implementing new symmetry groups, or improving documentation, your help is appreciated. You can also contribute by requesting new features or reporting performance bottlenecks.

## Ways to Contribute

- **Report issues and request features** via [GitHub Issues](https://github.com/Ideogenesis-AI/Nicole/issues)
- **Submit pull requests** with bug fixes or enhancements
- **Improve documentation** and add examples
- **Share your use cases** and provide constructive feedback

## Development Guidelines

When contributing to Nicole, please follow these guidelines:

- **Testing**: Ensure all contributions include appropriate tests
- **Code style**: Follow the existing code style (enforced by `ruff`)
- **Type hints**: Add type hints for new functions and classes
- **Documentation**: Update documentation for user-facing changes

## Development Setup

To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/Ideogenesis-AI/Nicole.git
cd Nicole

# Install in editable mode with all development dependencies
pip install -e ".[test,docs,lint]"

# Run tests
pytest

# Run linter
ruff check src/

# Build documentation
mkdocs serve
```

## Testing

Nicole uses `pytest` for testing. The test suite is comprehensive with 1400+ tests covering all functionality:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_contract.py

# Run with coverage
pytest --cov=nicole --cov-report=html
```

## Code Style

Nicole follows modern Python best practices:

- Use `ruff` for linting and formatting
- Follow PEP 8 conventions
- Use type hints throughout
- Write clear docstrings in NumPy style

## Submitting Changes

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `develop`
3. **Make your changes** with clear commit messages
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request** with a clear description

## Pull Request Guidelines

When submitting a pull request:

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Update the changelog if applicable
- Keep the scope focused and atomic

## Authors and Maintainers

Nicole is created and maintained by [Changkai Zhang](https://chx-zh.cc) as part of the Ideogenesis-AI effort in studying quantum many-body systems. If you have questions about contributing to the project or are interested in collaboration opportunities, please feel free to open an issue on GitHub or contact the maintainer directly.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful, constructive, and professional in all interactions.

## Questions?

If you have questions about contributing:

- Open a [GitHub Issue](https://github.com/Ideogenesis-AI/Nicole/issues)
- Check existing documentation and examples
- Review closed pull requests for guidance

Thank you for contributing to Nicole!
