# Adversarial Prototype Decomposition - APD

The core concept of Adversarial Prototype Decomposition (APD) is based on identifying prototypes (reference vectors) within a dataset and partitioning the feature space into smaller regions defined by pairs of adversarial prototypes. Adversarial prototypes are defined as prototypes that represent different classes.

## How it works
The APD framework follows a distinct two-phase process: Training and Inference.
1. Training Phase (Decomposition and Local Modeling)
   
  APD decomposes the global dataset into localized regions. By focusing on these smaller subsets of the input data, the algorithm enables highly efficient training of predictive models. A key advantage of APD is that it ensures each region encompasses a specific subspace of the decision boundary, providing the necessary context for models to learn complex local patterns. Then any standard prediction model can be trained within an identified region.

 Once training is complete, the individual local models are stored in a model dictionary. Each entry in this dictionary maps a specific regional subspace to its corresponding optimized predictive model.

2. Inference Phase (Prediction)

  When generating a prediction for a new data point, the APD algorithm follows these steps:

  Region Identification: The system calculates the distance between the input sample and the available pairs of adversarial prototypes to determine which localized region the sample belongs to.

  Model Retrieval: The algorithm retrieves the specific prediction model associated with that identified region from the dictionary.

  Final Prediction: The identified model is applied to the input data to produce the final output.

## How to get prototypes
Several approaches to prototype vectors identification of the APD exist, such as:
  - Random Identification: Selecting initial reference vectors at random.
  - Clustering Methods: Using algorithms to find representative centroids.
  - Learning Vector Quantization (LVQ): An iterative supervised learning process to refine prototype positions.

A more specialized strategy involves retaining only border prototypes—those located near the decision boundary (APD2). A prototype is considered a border sample if at least one of its neighbors belongs to an opposing class. The identification of these border samples can be efficiently achieved using graph-based methods, such as the Relative Nearest Neighbor Graph (RNNG) or the Gabriel Graph.



