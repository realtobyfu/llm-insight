# Contributing to LLM Interpretability Toolkit

Thank you for your interest in contributing to the LLM Interpretability Toolkit! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and considerate in all interactions.

## Getting Started

1. **Fork the Repository**
   ```bash
   git clone https://github.com/yourusername/llm-interpretability-toolkit.git
   cd llm-interpretability-toolkit
   ```

2. **Set Up Development Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev]"
   pre-commit install
   ```

3. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### 1. Making Changes

- Write clean, readable code following the project's style guidelines
- Add type hints to all function signatures
- Update or add docstrings for any new or modified functions
- Keep commits focused and atomic

### 2. Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Ruff**: Linting
- **MyPy**: Type checking

Run all checks:
```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Type checking
mypy src/

# Or run all at once
make lint
```

### 3. Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting PR
- Aim for >90% code coverage

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_attention.py

# Run tests in parallel
pytest -n auto
```

### 4. Documentation

- Update README.md if adding new features
- Add docstrings following NumPy style
- Update API documentation if changing endpoints

Example docstring:
```python
def analyze_attention(tokens: List[str], model_name: str = "gpt2") -> Dict[str, Any]:
    """
    Analyze attention patterns for given tokens.
    
    Parameters
    ----------
    tokens : List[str]
        List of input tokens to analyze
    model_name : str, optional
        Name of the model to use, by default "gpt2"
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing attention analysis results
        
    Examples
    --------
    >>> tokens = ["Hello", "world"]
    >>> results = analyze_attention(tokens)
    >>> print(results["max_attention"])
    0.85
    """
```

## Pull Request Process

1. **Before Submitting**
   - Ensure all tests pass
   - Run linting and formatting
   - Update documentation
   - Add entry to CHANGELOG.md

2. **PR Guidelines**
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changes were made and why
   - Include screenshots for UI changes

3. **PR Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Tests pass locally
   - [ ] New tests added
   - [ ] Coverage maintained/improved
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] CHANGELOG.md updated
   ```

## Areas for Contribution

### High Priority
- Performance optimizations
- Support for additional models
- Improved visualization tools
- Better error handling
- Documentation improvements

### Feature Ideas
- New interpretability methods
- Integration with MLOps platforms
- Additional export formats
- Multi-GPU support
- Real-time monitoring capabilities

### Good First Issues
Look for issues labeled `good first issue` or `help wanted` on GitHub.

## Development Tips

### Running Local API
```bash
# Start with auto-reload
uvicorn src.api.main:app --reload --port 8000

# With Redis caching
REDIS_URL=redis://localhost:6379 uvicorn src.api.main:app --reload
```

### Testing Dashboard Changes
```bash
# Run dashboard locally
streamlit run src/dashboard/app.py

# With custom port
streamlit run src/dashboard/app.py --server.port 8502
```

### Debugging
```python
# Add breakpoints
import pdb; pdb.set_trace()

# Or use VS Code debugger with launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": ["src.api.main:app", "--reload"],
            "jinja": true
        }
    ]
}
```

## Project Structure

When adding new features, follow the existing structure:

```
src/
├── core/           # Core algorithms (new analysis methods go here)
├── api/            # API endpoints (new routes go here)
├── dashboard/      # UI components (new visualizations go here)
├── visualization/  # Plotting utilities
└── utils/          # Helper functions
```

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. Build and publish to PyPI

## Questions?

- Open a [GitHub Discussion](https://github.com/yourusername/llm-interpretability-toolkit/discussions)
- Check existing [Issues](https://github.com/yourusername/llm-interpretability-toolkit/issues)
- Review the [Documentation](https://llm-interpretability-toolkit.readthedocs.io)

Thank you for contributing! 🎉