# ArchetypAX Examples

This directory contains examples demonstrating the usage of the ArchetypAX library.

## Available Examples

### [Basic Archetypal Analysis](01_archetypal_analysis_tutorial.ipynb)

A foundational introduction to archetypal analysis covering:

1. **Synthetic Data Generation**: Creating structured data to visually demonstrate the effectiveness of archetypal analysis
2. **Model Fitting**: Implementing and configuring the `ArchetypalAnalysis` and `ImprovedArchetypalAnalysis` classes
3. **Archetype Visualization**: Rendering the geometric relationship between archetypes and data points
4. **Evaluation Metrics**: Assessing model quality using the comprehensive `ArchetypalAnalysisEvaluator`
5. **Weight Distribution Analysis**: Examining attribution patterns between data points and archetypes
6. **Feature Importance**: Calculating and visualizing feature contributions across different archetypes

### [Archetypal Analysis Interpreter](02_archetypal_interpreter_tutorial.ipynb)

Demonstrates advanced interpretation techniques using the `ArchetypalAnalysisInterpreter`:

1. **Model Comparison**: Assessing different archetype configurations to determine optimal complexity
2. **Interpretability Metrics**: Quantifying the semantic clarity of discovered archetypes
3. **Feature Analysis**: Identifying distinctive features that characterize each archetype
4. **Pattern Discovery**: Revealing hidden structures and relationships in high-dimensional data

### [Biarchetypal Analysis](04_biarchetypal_analysis_tutorial.ipynb)

Explores dual-perspective pattern discovery with `BiarchetypalAnalysis`:

1. **Dual-Space Modeling**: Simultaneous discovery of patterns in both observation and feature spaces
2. **Cross-Modal Analysis**: Examining relationships between row and column archetypes
3. **Feature Clustering**: Identifying related features through column archetype patterns
4. **Enhanced Interpretability**: Leveraging the biarchetypal framework for clearer insights

### [Archetype Tracking](00_demonstration_archetypal_analysis.ipynb)

Demonstrates optimization trajectory analysis with `ArchetypeTracker`:

1. **Convergence Visualization**: Monitoring archetype movement during optimization
2. **Boundary Dynamics**: Analyzing how archetypes approach the data convex hull over iterations
3. **Stability Assessment**: Evaluating convergence quality and potential degenerate solutions
4. **Diagnostic Tools**: Using trajectory information to improve model configuration

## Running the Examples

To run the Jupyter notebooks, you'll need the following dependencies:

```bash
pip install -e ".[examples]"
```

Or install dependencies individually:

```bash
pip install matplotlib pandas seaborn jupyter
```

Then launch Jupyter and open any notebook:

```bash
jupyter notebook examples/
```

## Planned Examples

Future tutorials will cover:

- **Real-World Applications**: Archetypal analysis of diverse datasets from different domains
- **Advanced Visualization**: Sophisticated techniques for high-dimensional archetype visualization
- **Comparison Studies**: Benchmarking against alternative dimensionality reduction and clustering methods
- **Integration Patterns**: Incorporating archetypal analysis into broader machine learning pipelines
- **Optimization Strategies**: Fine-tuning model parameters for specific analytical objectives

## Contributing

We welcome contributions of new examples, extensions to existing tutorials, or suggestions for improvements. Please feel free to submit pull requests or open issues with your ideas.
