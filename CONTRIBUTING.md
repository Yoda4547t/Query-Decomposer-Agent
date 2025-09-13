# Contributing to Query Decomposer Agent

Thank you for your interest in contributing to the Query Decomposer Agent! This document provides guidelines and information for contributors.

## How to Contribute

### 1. Fork the Repository

1. Go to [https://github.com/Yoda4547t/Query-Decomposer-Agent](https://github.com/Yoda4547t/Query-Decomposer-Agent)
2. Click the "Fork" button in the top right corner
3. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Query-Decomposer-Agent.git
   cd Query-Decomposer-Agent
   ```

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Your Changes

- Follow the existing code style
- Add comments for complex logic
- Ensure your code works with Python 3.7+
- Test your changes thoroughly

### 4. Test Your Changes

```bash
# Run the basic tests
python query_decomposer_agent.py

# Test specific functionality
python -c "from query_decomposer_agent import test_enhanced_decomposition; test_enhanced_decomposition()"
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "Add: Brief description of your changes"
```

### 6. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 7. Create a Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Select your feature branch
4. Provide a clear description of your changes
5. Submit the pull request

## Development Guidelines

### Code Style

- Use meaningful variable and function names
- Add docstrings for all public functions and classes
- Follow PEP 8 style guidelines
- Keep functions focused and single-purpose

### Testing

- Test your changes with various query types
- Ensure the agent generates appropriate sub-queries
- Verify RAG integration still works correctly
- Test with different entity configurations

### Documentation

- Update README.md if you add new features
- Add examples for new functionality
- Update docstrings for any new functions

## Types of Contributions

### Bug Fixes

- Fix issues with query decomposition
- Improve entity detection accuracy
- Fix integration problems

### New Features

- Add new decomposition strategies
- Improve entity detection patterns
- Add new entity types
- Enhance RAG integration capabilities

### Documentation

- Improve README.md
- Add more examples
- Create tutorials
- Fix typos and improve clarity

### Performance Improvements

- Optimize decomposition algorithms
- Improve entity detection speed
- Reduce memory usage

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the problem
2. **Steps to Reproduce**: How to reproduce the issue
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, operating system
6. **Sample Query**: The query that causes the issue (if applicable)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and improve
- Follow GitHub's community guidelines

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## Questions?

If you have questions about contributing, please:

1. Check existing issues and discussions
2. Create a new issue with the "question" label
3. Contact the maintainer

Thank you for contributing to the Query Decomposer Agent! 🚀
