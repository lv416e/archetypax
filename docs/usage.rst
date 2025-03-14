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

Using Biarchetypal Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

For more expressive representations, ``archetypax`` provides biarchetypal analysis which uses two sets of archetypes:

.. code-block:: python

    import numpy as np
    from archetypax import BiarchetypalAnalysis

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(100, 5)  # 100 samples, 5 features

    # Initialize and fit the model with two sets of archetypes
    model = BiarchetypalAnalysis(
        n_archetypes_first=2,   # Number of archetypes in the first set
        n_archetypes_second=2,  # Number of archetypes in the second set
        mixture_weight=0.5,     # Weight for mixing the two archetype sets (0-1)
        max_iter=500,
        random_state=42
    )
    model.fit(X)

    # Get both sets of archetypes
    positive_archetypes, negative_archetypes = model.get_all_archetypes()
    print("Positive archetypes shape:", positive_archetypes.shape)
    print("Negative archetypes shape:", negative_archetypes.shape)

    # Get both sets of weights
    positive_weights, negative_weights = model.get_all_weights()
    print("Positive weights shape:", positive_weights.shape)
    print("Negative weights shape:", negative_weights.shape)

    # Reconstruct data using both sets of archetypes
    X_reconstructed = model.reconstruct()
    print("Reconstructed data shape:", X_reconstructed.shape)

Visualization
~~~~~~~~~~~~

``archetypax`` includes visualization utilities for exploring archetypal analysis results:

.. code-block:: python

    import matplotlib.pyplot as plt
    from archetypax.tools.visualization import ArchetypalAnalysisVisualizer

    # Fit model
    model = ArchetypalAnalysis(n_archetypes=3)
    model.fit(X)

    # Plot archetypes in 2D projection
    ArchetypalAnalysisVisualizer.plot_archetypes_2d(model, X)
    plt.show()

    # Plot membership weights
    weights = model.transform(X)
    ArchetypalAnalysisVisualizer.plot_membership_weights(weights)
    plt.show()

    # Plot archetype profiles
    ArchetypalAnalysisVisualizer.plot_archetype_profiles(model, feature_names=['F1', 'F2', 'F3', 'F4', 'F5'])
    plt.show()

Evaluation Metrics
~~~~~~~~~~~~~~~~

Evaluate the quality of your archetypal analysis:

.. code-block:: python

    from archetypax.tools.evaluation import ArchetypalAnalysisEvaluator

    # Fit model
    model = ArchetypalAnalysis(n_archetypes=4)
    model.fit(X)

    # Create an evaluator
    evaluator = ArchetypalAnalysisEvaluator(model)

    # Calculate reconstruction error
    error = evaluator.reconstruction_error(X, metric="frobenius")
    print(f"Reconstruction error: {error:.4f}")

    # Calculate explained variance
    variance = evaluator.explained_variance(X)
    print(f"Explained variance: {variance:.4f}")

    # Get comprehensive evaluation metrics
    metrics = evaluator.comprehensive_evaluation(X)
    print("Clustering metrics:", metrics["clustering"])
    print("Separation metrics:", evaluator.archetype_separation())

Interpretation Tools
~~~~~~~~~~~~~~~~~~

Interpret your archetypal analysis results:

.. code-block:: python

    from archetypax.tools.interpret import ArchetypalAnalysisInterpreter

    # Create an interpreter
    interpreter = ArchetypalAnalysisInterpreter()

    # Add a fitted model
    model = ArchetypalAnalysis(n_archetypes=3)
    model.fit(X)
    interpreter.add_model(3, model)  # Add model with key=3 (number of archetypes)

    # Add more models with different numbers of archetypes
    model4 = ArchetypalAnalysis(n_archetypes=4)
    model4.fit(X)
    interpreter.add_model(4, model4)

    # Calculate feature distinctiveness for archetypes
    distinctiveness = interpreter.feature_distinctiveness(model.archetypes)
    print("Feature distinctiveness:", distinctiveness)

    # Find optimal number of archetypes
    optimal_k = interpreter.find_optimal_k(X, k_range=[2, 3, 4, 5, 6])
    print(f"Optimal number of archetypes: {optimal_k}")

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
