# ecoILP: Ecological Link Prediction Package

## Installation
Clone the repository and install the package:
```bash
git clone https://github.com/Ecological-Complexity-Lab/eco_ILP.git
cd link-predict/package
pip install .
```

## Usage
```python
from ecoILP import handleEdgeList, extractFeatures, predictLinks, plotMetrics, plotProbsMatrix

dataframe = handleEdgeList(
    ...
    )

dataframe_with_features = extractFeatures(dataframe)

probabilities, classifications = predictLinks(dataframe_with_features)

plotMetrics(
    dataframe_with_features, 
    probabilities, 
    plots=['confusion_matrix', 'single_evaluation', 'roc_curve', 'pr_curve', 'probs_distribution', '...'],
    )

plotProbsMatrix(dataframe, probabilities, figsize=(14,8))
```

Better instructions and examples can be found in the [notebooks] folder.