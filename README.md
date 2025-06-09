# 🧠 NeuralNet2D: Custom Neural Network for Binary Classification in C#

This project implements a fully custom neural network framework in C# for 2D binary classification tasks. Designed for learning and experimentation, it includes robust numerical safety, gradient clipping, custom activation functions, and support for both SGD and Adam optimizers — all with **zero external dependencies**.

## 🔧 Features

- **Layer-based Architecture**: Define networks by stacking customizable layers with various activation functions (Sigmoid, Tanh, ReLU, LeakyReLU).
- **Optimizers**: Supports both SGD and Adam with gradient clipping (`ClipByNorm`) to prevent exploding gradients.
- **Loss Functions**: Includes Mean Squared Error (MSE) and Binary Cross-Entropy with stable derivatives.
- **2D Dataset Generator**: Built-in utility to generate clustered 2D data for binary classification.
- **Evaluation Metrics**: Compute Accuracy, Precision, Recall, and F1 Score.
- **Training Engine**: Custom training loop with epoch reporting and detailed error metrics.
- **NaN Detection**: All operations check for `NaN` or `Infinity` to prevent silent failures.
- **Export Tools**: Save prediction heatmaps in CSV format for easy visualization.

## 📂 Structure Overview

- `NeuralNetwork.cs`: Core logic for training and inference.
- `Layer.cs`: Defines individual layers with forward pass and gradient tracking.
- `Matrix.cs`: Custom matrix math engine with dot product, transpose, Hadamard product, and clipping.
- `AdamOptimizer.cs`: Implementation of the Adam optimizer with bias correction and norm clipping.
- `BinaryCrossEntropy.cs`: Loss function and its derivative, stable for probabilities.
- `Dataset2D.cs`: Synthetic 2D dataset generator with class clusters.
- `Exporter.cs`: Tool to export a prediction grid to CSV.
- `Metrics.cs`: Evaluation metrics for classification tasks.

## 🚀 Getting Started

1. Clone the repo.
2. Open the project in your C# IDE (e.g., Visual Studio).
3. Run training with your preferred settings (epochs, learning rate, optimizer).
4. Visualize predictions with `Exporter.Export2DClassificationGrid()`.

## 📊 Example Use Case

The network is especially suited for visual classification tasks in 2D space — ideal for students, researchers, or hobbyists wanting to understand how neural networks work under the hood.
