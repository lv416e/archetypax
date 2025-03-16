Installation
============

Prerequisites
------------

``archetypax`` requires Python 3.8 or later. The package leverages JAX for GPU acceleration, which necessitates compatible hardware for optimal performance.

Standard Installation
--------------------

You can install ``archetypax`` directly from PyPI using pip:

.. code-block:: bash

    pip install archetypax

This will install the core package with all required dependencies.

Development Installation
-----------------------

For development purposes, we recommend installing from source with development dependencies:

.. code-block:: bash

    git clone https://github.com/yourusername/archetypax.git
    cd archetypax
    pip install -e ".[dev,docs,examples]"

This installs the package in editable mode with additional dependencies for development, documentation, and examples.

GPU Acceleration
---------------

To leverage GPU acceleration, ensure you have installed the appropriate JAX version for your hardware:

.. code-block:: bash

    # For CUDA-compatible NVIDIA GPUs
    pip install --upgrade "jax[cuda12]"

    # For TPUs
    pip install --upgrade "jax[tpu]"

Verification
-----------

To verify your installation, you can run:

.. code-block:: python

    import archetypax
    print(archetypax.__version__)

    # Check if JAX can detect your accelerator
    import jax
    print(jax.devices())
