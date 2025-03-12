Contributing
============

We welcome contributions to ``archetypax``! This document outlines the process for contributing to the project and provides guidelines to ensure a smooth collaboration experience.

Development Environment
---------------------

To set up a development environment:

1. Fork the repository on GitHub.
2. Clone your fork locally:

   .. code-block:: bash

       git clone https://github.com/your-username/archetypax.git
       cd archetypax

3. Create a virtual environment and install development dependencies:

   .. code-block:: bash

       python -m venv .venv
       source .venv/bin/activate  # On Windows: .venv\Scripts\activate
       pip install -e ".[dev,docs,examples]"

4. Set up pre-commit hooks:

   .. code-block:: bash

       pre-commit install

Code Style
---------

``archetypax`` follows a consistent code style enforced by ruff and black. The configuration is defined in ``pyproject.toml``. Key style guidelines include:

- Use Google-style docstrings
- Maximum line length of 100 characters
- Type annotations for function signatures
- Comprehensive test coverage

To check your code style:

.. code-block:: bash

    ruff check .
    black --check .

To automatically format your code:

.. code-block:: bash

    ruff check --fix .
    black .

Pull Request Process
------------------

1. **Create a branch**: Create a new branch for your feature or bugfix:

   .. code-block:: bash

       git checkout -b feature/your-feature-name

2. **Make changes**: Implement your changes, following the code style guidelines.

3. **Write tests**: Add tests that verify your changes work as expected.

4. **Update documentation**: Update relevant documentation, including docstrings and this documentation site if necessary.

5. **Run tests locally**: Ensure all tests pass:

   .. code-block:: bash

       pytest

6. **Submit a pull request**: Push your branch to your fork and submit a pull request to the main repository.

   In your pull request description, clearly explain:

   - The purpose of your changes
   - Any issues they address
   - How to test the changes
   - Any dependencies introduced

7. **Code review**: Respond to any feedback on your pull request.

Testing
------

``archetypax`` uses pytest for testing. Tests are located in the ``tests/`` directory.

To run the test suite:

.. code-block:: bash

    pytest

For more verbose output:

.. code-block:: bash

    pytest -v

To run a specific test file:

.. code-block:: bash

    pytest tests/test_specific_file.py

Documentation
------------

Documentation is written in reStructuredText and built using Sphinx. To build the documentation locally:

.. code-block:: bash

    cd docs
    make html

The built documentation will be available in ``docs/_build/html/``.

When contributing new features, please include:

1. Docstrings for all public functions, classes, and methods
2. Updates to relevant documentation pages
3. Example usage in docstrings or example files

Versioning
---------

``archetypax`` follows semantic versioning (SemVer):

- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

Issue Reporting
-------------

If you encounter a bug or have a feature request, please submit an issue on GitHub. When reporting bugs, please include:

- A clear, descriptive title
- A detailed description of the issue
- Steps to reproduce the problem
- Expected behavior
- Actual behavior
- Environment information (OS, Python version, package versions)

Code of Conduct
-------------

We expect all contributors to adhere to our Code of Conduct. Please be respectful and constructive in all interactions.

License
------

By contributing to ``archetypax``, you agree that your contributions will be licensed under the project's Apache 2.0 license.
