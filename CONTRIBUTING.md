# Contributing to Indic Tokenization Lab

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/indic-tokenizer.git`
3. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Install dev dependencies: `pip install pytest pytest-cov black mypy ruff`

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest tests/ -v`
4. Run linting: `ruff check .` and `black --check .`
5. Commit your changes: `git commit -m "Add feature: description"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Open a Pull Request

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for all public functions and classes
- Keep functions focused and small
- Add tests for new functionality

## Testing

- Write tests for new features in `tests/`
- Run tests with: `pytest tests/ -v`
- Aim for good test coverage

## Documentation

- Update relevant documentation in `docs/` when adding features
- Update `README.md` if adding new user-facing features
- Add docstrings to all public APIs

## Areas for Contribution

We welcome contributions in:

- **Tokenizer implementations**: New tokenizer adapters or improvements
- **Evaluation metrics**: New metrics or improvements to existing ones
- **Documentation**: Improvements, examples, tutorials
- **Bug fixes**: Report issues and submit fixes
- **Test coverage**: More comprehensive tests
- **Examples**: More usage examples and demos

## Reporting Issues

When reporting issues, please include:

- Description of the problem
- Steps to reproduce
- Expected behavior
- Actual behavior
- Python version and environment details

## Questions?

Feel free to open an issue for questions or discussions.

