# ArchetypAX Examples

This directory contains examples demonstrating the usage of the ArchetypAX library.

## Basic Usage

[basic_usage.ipynb](basic_usage.ipynb) - A Jupyter notebook showcasing the fundamental capabilities and usage patterns of ArchetypAX. The notebook covers:

1. **Synthetic Data Generation**: Creating two-dimensional data with cluster structures to visually demonstrate the effectiveness of archetypal analysis.

2. **Model Fitting**: Demonstrating how to fit models to data using the `ArchetypalAnalysis` class.

3. **Archetype Visualization**: Visualizing the relationship between identified archetypes and data points.

4. **Evaluation Metrics**: Using the `ArchetypalAnalysisEvaluator` class to assess model quality.

5. **Weight Distribution Analysis**: Analyzing the distribution of weights (attribution scores) of data points to archetypes.

6. **Feature Importance**: Computing and visualizing feature importance across different archetypes.

## Running the Examples

To run the Jupyter notebooks, you'll need the following dependencies:

```bash
pip install -e ".[examples]"
```

Or install dependencies individually:

```bash
pip install matplotlib pandas seaborn jupyter
```

Then launch Jupyter and open the notebook:

```bash
jupyter notebook basic_usage.ipynb
```

## Future Examples

We plan to provide additional examples including:

- Archetypal analysis with real-world datasets
- Analysis and visualization of high-dimensional data
- Interpretation of archetypal analysis results
- Comparison with other dimensionality reduction techniques

## Contributing

We welcome suggestions for new examples and tutorials. Please contribute through pull requests.
