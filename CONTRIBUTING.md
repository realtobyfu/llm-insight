# Contributing to LLM Interpretability Toolkit

Thank you for your interest in contributing to the LLM Interpretability Toolkit! We welcome contributions from the community and are excited to work with you.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/llm-interpretability-toolkit.git
   cd llm-interpretability-toolkit
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure tests pass:
   ```bash
   pytest tests/
   ```

3. Format your code:
   ```bash
   black src tests
   isort src tests
   ```

4. Run linting:
   ```bash
   flake8 src tests
   mypy src
   ```

5. Commit your changes:
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

6. Push to your fork and create a pull request

## Code Style

- We use Black for code formatting (line length: 88)
- We use isort for import sorting
- Type hints are required for all public functions
- Docstrings should follow Google style

## Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use pytest fixtures for reusable test components
- Place unit tests in `tests/unit/` and integration tests in `tests/integration/`

## Documentation

- Update docstrings for any API changes
- Add examples to docstrings where helpful
- Update README.md if adding new features
- Create or update notebooks in `notebooks/` for new functionality

## Types of Contributions

### Bug Reports

- Use the GitHub issue tracker
- Include a minimal reproducible example
- Specify your environment (OS, Python version, package versions)

### Feature Requests

- Open an issue to discuss the feature first
- Explain the use case and benefits
- Consider implementing it yourself!

### Code Contributions

Areas where we especially welcome contributions:

1. **New Interpretability Methods**
   - Implement papers from recent research
   - Add novel visualization techniques
   - Extend support for new model architectures

2. **Performance Improvements**
   - Optimize memory usage for large models
   - Improve inference speed
   - Add caching strategies

3. **Documentation & Examples**
   - Create tutorials and guides
   - Add more example notebooks
   - Improve API documentation

4. **Testing**
   - Increase test coverage
   - Add integration tests
   - Create benchmarks

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add notes to CHANGELOG.md
4. Request review from maintainers
5. Address review feedback
6. Celebrate when merged! 🎉

## Community

- Join our Discord: [link]
- Follow us on Twitter: [@llm_interpret]
- Read our blog: [blog.llm-interpretability.org]

## Questions?

Feel free to open an issue or reach out to the maintainers. We're here to help!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.