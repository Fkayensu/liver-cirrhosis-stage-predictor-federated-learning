import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from model import CirrhosisPredictor
from evaluation import evaluate_model
from collections import deque

def train_local_model(model, data, epochs=30, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    X, y = data
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    return model.state_dict()

def aggregate_models(global_model, client_models):
    global_state_dict = global_model.state_dict()
    for key in global_state_dict.keys():
        global_state_dict[key] = torch.stack([client_model[key] for client_model in client_models]).mean(0)
    global_model.load_state_dict(global_state_dict)

def federated_learning_with_early_stopping(global_model, client_data, X_test_tensor, y_test_tensor, patience=5, min_delta=0.001):
    max_rounds = 100  # Set a maximum number of rounds as a safeguard
    best_accuracy = 0
    rounds_without_improvement = 0
    accuracy_history = deque(maxlen=patience)

    for round in range(max_rounds):
        client_models = []

        for client_X, client_y in client_data:
            local_model = CirrhosisPredictor(global_model.fc[0].in_features)
            local_model.load_state_dict(global_model.state_dict())
            client_model_state = train_local_model(local_model, (client_X, client_y))
            client_models.append(client_model_state)

        aggregate_models(global_model, client_models)
        test_accuracy = evaluate_model(global_model, X_test_tensor, y_test_tensor)
        print(f"Round {round + 1}, Test Accuracy: {test_accuracy:.4f}")

        accuracy_history.append(test_accuracy)

        if test_accuracy > best_accuracy + min_delta:
            best_accuracy = test_accuracy
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if rounds_without_improvement >= patience:
            print(f"Early stopping triggered. Best accuracy: {best_accuracy:.4f}")
            break

        if len(accuracy_history) == patience and max(accuracy_history) - min(accuracy_history) < min_delta:
            print(f"Converged. Best accuracy: {best_accuracy:.4f}")
            break

    return global_model