Welcome to ArchetypAX's documentation!
=====================================

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
* Multiple analysis variants, including Sparse Archetypal Analysis and Biarchetypal Analysis
* Innovative ArchetypeTracker for monitoring archetype evolution during optimization
* Trajectory analysis tools for visualizing archetype movements and convergence patterns
* Boundary proximity tracking with historical metrics for evolutionary pattern analysis

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   examples
   reference/index
   architecture
   migration_guide
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
