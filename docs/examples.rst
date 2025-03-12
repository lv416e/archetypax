Examples
========

This section provides comprehensive examples demonstrating the application of archetypal analysis in various domains.

Synthetic Data Example
---------------------

This example illustrates the fundamental concepts of archetypal analysis using synthetic data:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    from archetypax import ArchetypalAnalysis

    # Generate synthetic data with clear cluster structure
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

    # Fit archetypal analysis model
    model = ArchetypalAnalysis(n_archetypes=4, random_state=42)
    model.fit(X)

    # Get archetypes and transform data
    archetypes = model.archetypes_
    weights = model.transform(X)

    # Visualize results
    plt.figure(figsize=(12, 10))

    # Plot original data points
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label='Data points')

    # Plot archetypes
    plt.scatter(archetypes[:, 0], archetypes[:, 1], s=200, c='red',
                marker='*', label='Archetypes', edgecolors='black')

    # Draw convex hull of archetypes
    from scipy.spatial import ConvexHull
    hull = ConvexHull(archetypes)
    for simplex in hull.simplices:
        plt.plot(archetypes[simplex, 0], archetypes[simplex, 1], 'k-', lw=2)

    plt.title('Archetypal Analysis on Synthetic Data', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

Text Data Analysis
----------------

Archetypal analysis can be applied to text data to discover archetypal topics or document types:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from archetypax import ArchetypalAnalysis

    # Load text data from 20 newsgroups dataset
    categories = [
        'comp.graphics', 'comp.os.ms-windows.misc',
        'comp.sys.mac.hardware', 'comp.sys.ibm.pc.hardware',
        'rec.autos', 'rec.motorcycles',
        'rec.sport.baseball', 'rec.sport.hockey'
    ]

    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )

    # Extract features using TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.7,
        stop_words='english'
    )

    X_tfidf = vectorizer.fit_transform(newsgroups.data)
    feature_names = vectorizer.get_feature_names_out()

    # Reduce dimensionality with SVD for computational efficiency
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_svd = svd.fit_transform(X_tfidf)

    # Apply archetypal analysis
    model = ArchetypalAnalysis(n_archetypes=8, random_state=42)
    model.fit(X_svd)

    # Get archetypes and transform back to TF-IDF space
    archetypes_svd = model.archetypes_
    archetypes_tfidf = svd.inverse_transform(archetypes_svd)

    # Get weights for each document
    weights = model.transform(X_svd)

    # Function to extract top terms for each archetype
    def get_top_terms(archetype_vector, feature_names, top_n=15):
        # Get indices of top terms
        top_indices = archetype_vector.argsort()[-top_n:][::-1]
        # Return top terms and their weights
        return [(feature_names[i], archetype_vector[i]) for i in top_indices]

    # Print top terms for each archetype
    print("Top terms for each archetype:")
    for i, archetype in enumerate(archetypes_tfidf):
        print(f"\nArchetype {i+1}:")
        top_terms = get_top_terms(archetype, feature_names)
        for term, weight in top_terms:
            print(f"  {term}: {weight:.4f}")

    # Visualize document weights for each archetype
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        weights,
        cmap='viridis',
        xticklabels=[f'Archetype {i+1}' for i in range(weights.shape[1])],
        yticklabels=False
    )
    plt.title('Document Weights for Each Archetype', fontsize=16)
    plt.xlabel('Archetypes', fontsize=14)
    plt.ylabel('Documents', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Visualize category distribution for each archetype
    # Assign each document to its dominant archetype
    dominant_archetype = np.argmax(weights, axis=1)

    # Create a DataFrame with document categories and dominant archetypes
    df = pd.DataFrame({
        'category': [categories[newsgroups.target[i]] for i in range(len(newsgroups.target))],
        'dominant_archetype': [f'Archetype {i+1}' for i in dominant_archetype]
    })

    # Count documents by category and archetype
    category_counts = df.groupby(['dominant_archetype', 'category']).size().unstack(fill_value=0)

    # Normalize by archetype to get percentages
    category_percentages = category_counts.div(category_counts.sum(axis=1), axis=0) * 100

    # Plot category distribution
    plt.figure(figsize=(16, 12))
    category_percentages.plot(
        kind='bar',
        stacked=True,
        colormap='tab10',
        figsize=(16, 10)
    )
    plt.title('Category Distribution for Each Archetype', fontsize=16)
    plt.xlabel('Archetype', fontsize=14)
    plt.ylabel('Percentage of Documents', fontsize=14)
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # Visualize archetypes in 2D space
    # Project archetypes and data to 2D using t-SNE
    from sklearn.manifold import TSNE

    # Apply t-SNE to SVD-reduced data
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_svd)
    archetypes_tsne = tsne.fit_transform(archetypes_svd)

    # Create a scatter plot
    plt.figure(figsize=(14, 10))

    # Plot documents colored by category
    for i, category in enumerate(categories):
        indices = np.where(newsgroups.target == i)[0]
        plt.scatter(
            X_tsne[indices, 0],
            X_tsne[indices, 1],
            alpha=0.5,
            label=category,
            s=30
        )

    # Plot archetypes
    plt.scatter(
        archetypes_tsne[:, 0],
        archetypes_tsne[:, 1],
        s=300,
        c='black',
        marker='*',
        label='Archetypes',
        edgecolors='white',
        linewidths=1.5
    )

    # Add archetype labels
    for i, (x, y) in enumerate(archetypes_tsne):
        plt.annotate(
            f'A{i+1}',
            (x, y),
            fontsize=12,
            fontweight='bold',
            color='white',
            ha='center',
            va='center'
        )

    plt.title('t-SNE Projection of Documents and Archetypes', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Analyze a specific document
    # Find a document with high weight for a particular archetype
    archetype_idx = 0  # Change this to analyze different archetypes
    top_doc_idx = np.argsort(weights[:, archetype_idx])[-1]

    print(f"\nExample document with high weight for Archetype {archetype_idx+1}:")
    print(f"Category: {categories[newsgroups.target[top_doc_idx]]}")
    print(f"Weights: {weights[top_doc_idx]}")
    print("\nDocument text:")
    print(newsgroups.data[top_doc_idx][:500] + "...")  # Show first 500 chars

Image Data Analysis
------------------

Archetypal analysis can be applied to image data to extract representative patterns:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_olivetti_faces
    from archetypax import ArchetypalAnalysis

    # Load face dataset
    faces = fetch_olivetti_faces()
    X = faces.data  # (400, 4096) - 400 images, 64x64 pixels flattened

    # Fit archetypal analysis
    model = ArchetypalAnalysis(n_archetypes=10, random_state=42)
    model.fit(X)

    # Get archetypes (archetypal faces)
    archetypes = model.archetypes_

    # Visualize archetypal faces
    fig, axes = plt.subplots(2, 5, figsize=(15, 6),
                            subplot_kw={'xticks': [], 'yticks': []})

    for i, ax in enumerate(axes.flat):
        # Reshape to 64x64 image
        face = archetypes[i].reshape(64, 64)
        ax.imshow(face, cmap='gray')
        ax.set_title(f'Archetype {i+1}')

    plt.suptitle('Archetypal Faces', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

    # Reconstruct a face using archetypes
    sample_idx = 150
    sample_face = X[sample_idx]
    sample_weights = model.transform(sample_face.reshape(1, -1))

    reconstructed_face = model.inverse_transform(sample_weights)

    # Visualize original vs reconstructed
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5),
                                  subplot_kw={'xticks': [], 'yticks': []})

    ax1.imshow(sample_face.reshape(64, 64), cmap='gray')
    ax1.set_title('Original Face')

    ax2.imshow(reconstructed_face.reshape(64, 64), cmap='gray')
    ax2.set_title('Reconstructed Face')

    plt.tight_layout()
    plt.show()

Genomic Data Analysis
-------------------

Archetypal analysis can identify archetypal expression patterns in genomic data:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from archetypax import ArchetypalAnalysis

    # Simulate gene expression data
    # In practice, you would load real data
    np.random.seed(42)
    n_samples = 200  # patients
    n_genes = 1000   # genes

    # Simulate gene expression matrix
    X = np.random.exponential(scale=1.0, size=(n_samples, n_genes))

    # Add some structure to the data
    for i in range(0, n_samples, 50):
        X[i:i+50, i//50*250:(i//50+1)*250] *= 3

    # Fit archetypal analysis
    model = ArchetypalAnalysis(n_archetypes=4, random_state=42)
    model.fit(X)

    # Get archetypes and weights
    archetypes = model.archetypes_
    weights = model.transform(X)

    # Visualize archetype weights for each sample
    plt.figure(figsize=(12, 8))
    sns.heatmap(weights, cmap='viridis',
                xticklabels=[f'Archetype {i+1}' for i in range(weights.shape[1])],
                yticklabels=False)
    plt.title('Sample Weights for Each Archetype', fontsize=16)
    plt.xlabel('Archetypes', fontsize=14)
    plt.ylabel('Samples', fontsize=14)
    plt.tight_layout()
    plt.show()

    # Visualize gene expression patterns in archetypes
    plt.figure(figsize=(15, 10))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.plot(archetypes[i])
        plt.title(f'Archetype {i+1} Gene Expression Pattern', fontsize=14)
        plt.xlabel('Gene Index', fontsize=12)
        plt.ylabel('Expression Level', fontsize=12)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

Market Segmentation Example
-------------------------

Archetypal analysis can be used for customer segmentation in marketing:

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from archetypax import ArchetypalAnalysis

    # Simulate customer data
    np.random.seed(42)
    n_customers = 500

    # Features: age, income, spending, online_activity, store_visits
    X = np.zeros((n_customers, 5))

    # Generate different customer profiles
    X[:125, 0] = np.random.normal(25, 5, 125)  # Young
    X[125:250, 0] = np.random.normal(35, 5, 125)  # Middle-aged
    X[250:375, 0] = np.random.normal(45, 5, 125)  # Older middle-aged
    X[375:, 0] = np.random.normal(65, 5, 125)  # Senior

    X[:125, 1] = np.random.normal(40000, 10000, 125)  # Lower income
    X[125:250, 1] = np.random.normal(70000, 15000, 125)  # Middle income
    X[250:375, 1] = np.random.normal(100000, 20000, 125)  # Upper middle income
    X[375:, 1] = np.random.normal(60000, 15000, 125)  # Retirement income

    # Other features with correlations to age/income
    for i in range(n_customers):
        age_factor = X[i, 0] / 40  # Normalized by average age
        income_factor = X[i, 1] / 70000  # Normalized by average income

        # Spending (younger and higher income spend more)
        X[i, 2] = np.random.normal(5000 * (2 - age_factor) * income_factor, 1000)

        # Online activity (younger are more active online)
        X[i, 3] = np.random.normal(10 * (2 - age_factor), 2)

        # Store visits (older visit stores more)
        X[i, 4] = np.random.normal(20 * age_factor, 5)

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit archetypal analysis
    model = ArchetypalAnalysis(n_archetypes=4, random_state=42)
    model.fit(X_scaled)

    # Get archetypes and weights
    archetypes = model.archetypes_
    archetypes_original = scaler.inverse_transform(archetypes)
    weights = model.transform(X_scaled)

    # Create feature names for better visualization
    feature_names = ['Age', 'Income', 'Spending', 'Online Activity', 'Store Visits']

    # Visualize archetypes
    plt.figure(figsize=(14, 10))

    # Create a radar chart for each archetype
    from matplotlib.path import Path
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D

    def radar_factory(num_vars, frame='circle'):
        theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

        class RadarAxes(plt.PolarAxes):
            name = 'radar'

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.set_theta_zero_location('N')

            def fill(self, *args, **kwargs):
                return super().fill_between(*args, **kwargs)

            def plot(self, *args, **kwargs):
                lines = super().plot(*args, **kwargs)
                self._close_polygon(lines)
                return lines

            def _close_polygon(self, lines):
                for line in lines:
                    x, y = line.get_data()
                    if x[0] != x[-1]:
                        x = np.concatenate((x, [x[0]]))
                        y = np.concatenate((y, [y[0]]))
                        line.set_data(x, y)

            def set_varlabels(self, labels):
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                return plt.Circle((0.5, 0.5), 0.5)

        register_projection(RadarAxes)
        return theta

    from matplotlib.projections import register_projection

    # Normalize archetype values for radar chart
    archetypes_radar = np.zeros_like(archetypes_original)
    for i in range(archetypes_original.shape[1]):
        min_val = np.min(X[:, i])
        max_val = np.max(X[:, i])
        archetypes_radar[:, i] = (archetypes_original[:, i] - min_val) / (max_val - min_val)

    # Create radar chart
    theta = radar_factory(len(feature_names))

    fig, axes = plt.subplots(figsize=(15, 12), nrows=2, ncols=2,
                            subplot_kw=dict(projection='radar'))

    colors = ['b', 'g', 'r', 'c']

    for ax, color, archetype, archetype_orig in zip(axes.flat, colors,
                                                  archetypes_radar,
                                                  archetypes_original):
        ax.plot(theta, archetype, color=color)
        ax.fill(theta, archetype, facecolor=color, alpha=0.25)
        ax.set_varlabels(feature_names)

        # Add values in original scale
        for i, value in enumerate(archetype_orig):
            angle = i * 2 * np.pi / len(feature_names)
            ax.text(angle, 1.15, f"{value:.0f}",
                   horizontalalignment='center', size='small')

    # Add titles
    titles = ['Young Digital Shoppers', 'Affluent Professionals',
             'Traditional Shoppers', 'Senior Conservatives']

    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                    horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

    # Assign each customer to dominant archetype
    dominant_archetype = np.argmax(weights, axis=1)

    # Visualize customer segments
    plt.figure(figsize=(12, 10))

    # Create scatter plot of age vs income colored by dominant archetype
    plt.scatter(X[:, 0], X[:, 1], c=dominant_archetype, cmap='viridis',
               alpha=0.7, s=50)

    plt.colorbar(ticks=range(4), label='Dominant Archetype')
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Income', fontsize=14)
    plt.title('Customer Segmentation by Age and Income', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
