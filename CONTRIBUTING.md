# Contributing to GUI Agent

Thank you for your interest in contributing to GUI Agent! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Testing Requirements](#testing-requirements)
- [Issue Reporting](#issue-reporting)

---

## Code of Conduct

- Be respectful and inclusive in all interactions
- Focus on constructive feedback and helpful discussions
- Welcome contributors of all experience levels
- Keep discussions professional and on-topic

---

## Getting Started

### Where to Start

- **Beginners**: Look for issues labeled `good first issue`
- **Experienced**: Check issues labeled `help wanted` or `enhancement`
- **Not sure?** Open a discussion to find good starting points

### Types of Contributions

We welcome:
- Bug fixes
- New features
- Documentation improvements
- Test additions
- Performance optimizations
- Security improvements

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone:
git clone https://github.com/YOUR_USERNAME/gui-agent.git
cd gui-agent
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Install in development mode
pip install -e .

# Install development dependencies (if available)
pip install -r requirements-dev.txt
```

### 4. Set Up Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your API key
```

---

## Code Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints for function signatures

### Code Organization

- Keep functions focused and single-purpose
- Use descriptive variable and function names
- Add docstrings to public functions and classes
- Separate concerns into appropriate modules

### Example Code Structure

```python
"""
Module description.
"""

from typing import Optional


class ClassName:
    """Class docstring describing purpose."""

    def __init__(self, param: str) -> None:
        """Initialize the class.

        Args:
            param: Description of parameter.
        """
        self.param = param

    def public_method(self, arg: int) -> str:
        """Method docstring.

        Args:
            arg: Description of argument.

        Returns:
            Description of return value.
        """
        return str(arg)
```

### Pre-commit Hooks (Optional)

We recommend using pre-commit hooks for consistent code quality:

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install
```

---

## Pull Request Process

### 1. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### 2. Make Changes

- Make focused, atomic commits
- Write clear commit messages
- Keep changes related to a single topic

### Commit Message Guidelines

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Example:**
```
feat(cli): add color output support

Implemented rich-based colored output for CLI.
- Added LogLevel colors (INFO=blue, WARN=yellow, ERROR=red)
- Added progress spinners for API calls
- Added table formatting for action history

Closes #123
```

### 3. Test Your Changes

```bash
# Run all tests
python -m pytest test-archive/

# Run specific test file
python -m pytest test-archive/test_agent.py

# Run with coverage (if configured)
pytest --cov=gui_agent
```

### 4. Submit Pull Request

1. Push your branch: `git push origin feature/your-feature-name`
2. Go to the repository on GitHub
3. Click "New Pull Request"
4. Fill in the PR template
5. Wait for review

### PR Review Process

- All PRs require at least one review
- Address review feedback promptly
- Maintain focus on the original scope
- CI checks must pass before merging

---

## Testing Requirements

### Running Tests

```bash
# Run all tests
python -m pytest test-archive/

# Run with verbose output
python -m pytest test-archive/ -v

# Run specific test
python -m pytest test-archive/test_agent.py::test_screenshot

# Run with coverage
pytest --cov=. --cov-report=html
```

### Writing Tests

- Add tests for new features
- Maintain or improve code coverage
- Use descriptive test names: `test_<method>_<scenario>_<expected_result>`
- Follow the AAA pattern: Arrange, Act, Assert

### Test Categories

- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

### Test Requirements

All PRs must:
- Pass all existing tests
- Include tests for new functionality
- Not decrease overall code coverage

---

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Description**: Clear description of the issue
2. **Environment**:
   - Python version
   - OS and version
   - GUI Agent version
3. **Steps to Reproduce**: Detailed steps to reproduce the issue
4. **Expected Behavior**: What should happen
5. **Actual Behavior**: What actually happens
6. **Logs**: Relevant error messages or logs
7. **Screenshots**: If applicable

### Feature Requests

When requesting features, include:

1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Use Cases**: Examples of how it would be used
4. **Alternatives**: Other solutions you've considered

---

## Documentation

### README Updates

Update the README.md when:
- Adding new features
- Changing existing behavior
- Adding configuration options
- Modifying installation requirements

### Code Comments

- Explain _why_, not _what_
- Document non-obvious decisions
- Reference issues or discussions for context

---

## Security

### Reporting Security Issues

**Do not** report security vulnerabilities in public issues.

Instead:
1. Email: [security email or GitHub private vulnerability reporting]
2. Include detailed description
3. Allow reasonable time for response

### Security Best Practices

- Never commit credentials
- Use environment variables for secrets
- Follow OWASP guidelines
- Report potential vulnerabilities responsibly

---

## Questions?

- **General questions**: Open a GitHub Discussion
- **Chat**: [Link to Discord/Slack if available]
- **Twitter**: [@handle if available]

---

## License

By contributing to GUI Agent, you agree that your contributions will be licensed under the [MIT License](LICENSE).
