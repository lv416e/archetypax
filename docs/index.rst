Welcome to archetypax's documentation!
===================================

``archetypax`` is a GPU-accelerated implementation of Archetypal Analysis using JAX.

Overview
--------

Archetypal Analysis is a statistical method for dimensionality reduction that represents data points as convex combinations of extreme points (archetypes). Unlike PCA, which finds orthogonal directions of maximum variance, archetypal analysis finds extreme points that lie on the convex hull of the data, making it particularly useful for interpretable feature extraction and data exploration.

``archetypax`` provides:

* High-performance implementation using JAX for GPU acceleration
* Scikit-learn compatible API for seamless integration
* Advanced optimization techniques for improved convergence
* Comprehensive visualization tools for result interpretation
* Extensive evaluation metrics for model assessment

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   examples
   reference/index
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
