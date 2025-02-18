# Liver Cirrhosis Stage Predictor using Federated Learning

This repository contains an implementation of a federated learning system for predicting liver cirrhosis stages. The model uses decentralized training across multiple clients while preserving data privacy.

![Federated Learning Setup](fl_architecture.png)

## Overview

Federated Learning allows multiple clients to collaboratively train a model without sharing their original data. Instead, each client trains a local model on their private data, and only model updates are shared with a central server. The server aggregates these updates to improve a global model, which is then redistributed to clients for the next round of training.

This implementation focuses on predicting liver cirrhosis stages (1, 2, or 3) using patient data distributed across multiple simulated clients.

## Project Structure

```
liver-cirrhosis-stage-predictor-fl/
├── src/
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── federated_learning.py
│   ├── evaluation.py
│   └── main.py
├── data/
├── results/
└── README.md
```

## Components

### 1. Data Preprocessing (`data_preprocessing.py`)

This module handles data loading, cleaning, transformation, and splitting:
- Loads the liver cirrhosis dataset
- Drops missing values
- Encodes categorical variables
- Standardizes numerical features
- Splits data among multiple clients for federated learning

### 2. Model Architecture (`model.py`)

Defines the neural network architecture for cirrhosis prediction:
- Multi-layer perceptron with 3 hidden layers (256, 128, 64 neurons)
- ReLU activation functions
- Dropout layers (30% dropout) for regularization
- Softmax output for 3-class classification

### 3. Federated Learning Implementation (`federated_learning.py`)

Manages the federated learning process:
- Distributes the global model to clients
- Trains local models on client data
- Aggregates model updates (FedAvg algorithm)
- Evaluates global model performance after each round

### 4. Evaluation Metrics (`evaluation.py`)

Provides functions for model assessment:
- Accuracy calculation
- Per-class precision, recall, and F1-score
- Formatted display of evaluation results

### 5. Main Execution (`main.py`)

Orchestrates the entire workflow:
- Data loading and preprocessing
- Client data distribution 
- Global model initialization
- Federated learning loop execution
- Final model evaluation and metrics reporting

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- pandas
- numpy
- scikit-learn

### Installation

1. Clone this repository:
```bash
git clone https://github.com/fkayensu/liver-cirrhosis-stage-predictor-federated-learning.git
cd liver-cirrhosis-stage-predictor-federated-learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Data

Place your liver cirrhosis dataset (CSV format) in the `data/` directory. The dataset should contain patient features and a 'Stage' column with values 1, 2, or 3.

### Running the Project

```bash
python src/main.py
```

## Federated Learning Process

1. The central server initializes a global model
2. For each round of training:
   - The server distributes the current global model to all clients
   - Each client trains the model on their local data
   - Clients send model updates (not the data) back to the server
   - The server aggregates updates using FedAvg algorithm
   - The updated global model is evaluated on test data
3. After multiple rounds, the final global model achieves good performance without directly accessing client data

## Results

The model performance metrics are stored in the `results/` directory after training completes. These include:
- Overall accuracy
- Per-class precision, recall, and F1-scores
- Training curves showing performance across federated learning rounds

## Privacy Considerations

This implementation ensures privacy by:
- Keeping all client data local
- Only sharing model parameters, not actual data
- Using secure aggregation (implemented through parameter averaging)

## Future Improvements

- Implement differential privacy mechanisms
- Support for heterogeneous client data
- Experiment with different aggregation algorithms
- Add client selection strategies
- Incorporate more advanced neural network architectures

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The dataset used in this project is sourced from [provide source if applicable]
- Inspired by the work on federated learning by [relevant references]