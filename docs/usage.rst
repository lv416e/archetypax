Usage
=====

Basic Usage
----------

``archetypax`` provides a scikit-learn compatible API for performing archetypal analysis. Here's a basic example:

.. code-block:: python

    import numpy as np
    from archetypax import ArchetypalAnalysis

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features

    # Initialize and fit the model
    model = ArchetypalAnalysis(n_archetypes=3, random_state=42)
    model.fit(X)

    # Get archetypes
    archetypes = model.archetypes_
    print("Archetypes shape:", archetypes.shape)

    # Transform new data to get weights
    X_new = np.random.rand(10, 5)  # 10 new samples
    weights = model.transform(X_new)
    print("Weights shape:", weights.shape)

    # Reconstruct data from weights and archetypes
    X_reconstructed = model.inverse_transform(weights)
    print("Reconstructed data shape:", X_reconstructed.shape)

Advanced Usage
-------------

Customizing Optimization Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``archetypax`` allows fine-tuning of the optimization process:

.. code-block:: python

    from archetypax import ArchetypalAnalysis

    model = ArchetypalAnalysis(
        n_archetypes=5,
        max_iter=1000,
        tol=1e-6,
        learning_rate=0.01,
        batch_size=32,
        random_state=42
    )
    model.fit(X)

Using the Improved Model
~~~~~~~~~~~~~~~~~~~~~~~

For enhanced performance, ``archetypax`` offers an improved implementation:

.. code-block:: python

    from archetypax import ImprovedArchetypalAnalysis

    model = ImprovedArchetypalAnalysis(
        n_archetypes=4,
        convex_hull_init=True,  # Initialize archetypes near the convex hull
        regularization=0.001,   # Add regularization for stability
        random_state=42
    )
    model.fit(X)

Visualization
~~~~~~~~~~~~

``archetypax`` includes visualization utilities for exploring archetypal analysis results:

.. code-block:: python

    import matplotlib.pyplot as plt
    from archetypax.visualization import plot_archetypes_2d, plot_weights_distribution

    # Fit model
    model = ArchetypalAnalysis(n_archetypes=3)
    model.fit(X)

    # Plot archetypes in 2D projection
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_archetypes_2d(model, X, ax=ax)
    plt.show()

    # Plot weights distribution
    weights = model.transform(X)
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_weights_distribution(weights, ax=ax)
    plt.show()

Evaluation Metrics
~~~~~~~~~~~~~~~~

Evaluate the quality of your archetypal analysis:

.. code-block:: python

    from archetypax.evaluation import reconstruction_error, explained_variance

    # Fit model
    model = ArchetypalAnalysis(n_archetypes=4)
    model.fit(X)

    # Calculate reconstruction error
    error = reconstruction_error(model, X)
    print(f"Reconstruction error: {error:.4f}")

    # Calculate explained variance
    variance = explained_variance(model, X)
    print(f"Explained variance: {variance:.4f}")

Integration with scikit-learn
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``archetypax`` integrates seamlessly with scikit-learn pipelines:

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from archetypax import ArchetypalAnalysis

    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('archetype', ArchetypalAnalysis(n_archetypes=3))
    ])

    # Fit and transform
    pipeline.fit(X)
    weights = pipeline.transform(X)
