# README for Iris Classification with MLPClassifier

## Overview

This project demonstrates how to classify the Iris dataset using a Multi-layer Perceptron (MLP) classifier implemented in Python. The notebook utilizes popular libraries such as NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn to preprocess the data, train the model, and visualize the results.

## Requirements

To run this notebook, you will need the following packages:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `piplite` (for package installation in Pyodide)


## Dataset

The Iris dataset is a classic dataset in machine learning that includes three species of iris flowers (Setosa, Versicolor, and Virginica) with four features:

- Sepal length
- Sepal width
- Petal length
- Petal width

## Workflow

The notebook follows these steps:

1. **Load the Iris Dataset**: The dataset is loaded using `sklearn.datasets.load_iris()`.
2. **Data Preparation**:
   - Convert the dataset into a Pandas DataFrame for better visualization.
   - Split the data into training (80%) and testing (20%) sets.
   - Standardize the features using `StandardScaler`.
3. **Model Training**:
   - Train an MLP classifier with two hidden layers of 10 neurons each.
4. **Model Evaluation**:
   - Predict the species of iris flowers in the test set.
   - Generate and display a confusion matrix and classification report.
5. **Visualization**:
   - Plot the confusion matrix using Seaborn for better interpretation of results.
6. **Save Predictions**: Save the actual and predicted values to a CSV file named `iris_predictions.csv`.

## Code Snippet

Here’s a brief code snippet demonstrating how to load libraries and prepare the dataset:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['species'] = y
print(iris_df.head())
```

## Running the Notebook

To run this notebook:

1. Ensure you have a Python environment with the required packages installed.
2. Open a Jupyter Notebook or any compatible environment that supports Python (Pyodide).
3. Execute each cell sequentially.

## Output

After running the notebook, you will see:

- A printed confusion matrix and classification report in your console.
- A visual representation of the confusion matrix displayed as a heatmap.
- A CSV file named `iris_predictions.csv` containing actual vs predicted values.

## Conclusion

This project serves as an excellent introduction to machine learning classification tasks using neural networks. You can modify hyperparameters or experiment with different classifiers to further enhance your understanding of model performance.
For any questions or contributions, feel free to reach out!
