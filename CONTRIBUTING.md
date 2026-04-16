# Contributing to AksaraLLM 🇮🇩

Thank you for your interest in contributing to AksaraLLM!

## How to Contribute

### 1. Report Issues
- Open a GitHub Issue with a clear description
- Include error logs, screenshots, or reproduction steps

### 2. Submit Code
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `python -m unittest discover -s tests`
5. Run release gate: `python scripts/release_check.py`
6. Never commit live secrets or hardcoded access tokens
7. Submit a Pull Request

### 3. Improve Data
We always need more Indonesian language data:
- SFT conversation pairs
- Knowledge QA pairs
- Safety alignment examples
- Translation pairs

### 4. Improve Documentation
- Fix typos, improve explanations
- Add examples and tutorials
- Translate docs to Indonesian

## Code Style
- Python 3.10+
- Use type hints
- Add docstrings to public functions
- Follow PEP 8

## License
By contributing, you agree that your contributions will be licensed under Apache 2.0.
