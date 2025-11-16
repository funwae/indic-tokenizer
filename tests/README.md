# Tests

This directory contains unit tests for the Indic Tokenization Lab.

## Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=tokenizers --cov=eval

# Run specific test file
pytest tests/test_grapheme_segmentation.py

# Run specific test
pytest tests/test_grapheme_segmentation.py::test_basic_grapheme_segmentation
```

## Test Structure

- `test_grapheme_segmentation.py` - Tests for grapheme segmentation
- `test_cbpe_constraints.py` - Tests for CBPE constraints
- `test_evaluation.py` - Tests for evaluation metrics
- `test_pretokenizer.py` - Tests for pretokenizer

## Adding Tests

When adding new functionality, add corresponding tests:

1. Create test file: `tests/test_<module>.py`
2. Import the module to test
3. Write test functions starting with `test_`
4. Use assertions to verify behavior
5. Run tests to ensure they pass

## Test Coverage

Aim for:
- Unit tests for all core functions
- Edge case testing
- Integration tests for workflows
- Regression tests for bugs

