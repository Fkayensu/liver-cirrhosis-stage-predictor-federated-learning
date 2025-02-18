import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from model import CirrhosisPredictor
from evaluation import evaluate_model
from collections import deque
from encryption import EncryptionSimulator, encrypt_vector, decrypt_vector

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

def aggregate_models(global_model, client_models, client_data_sizes, encryption_simulator):
    global_state_dict = global_model.state_dict()
    total_data = sum(client_data_sizes)
    
    for key in global_state_dict.keys():
        weighted_sum = torch.zeros_like(global_state_dict[key])
        for client_model, data_size in zip(client_models, client_data_sizes):
            weight = data_size / total_data
            decrypted_param = decrypt_vector(encryption_simulator, client_model[key])
            decrypted_param = torch.tensor(decrypted_param).reshape(global_state_dict[key].shape)
            weighted_sum += weight * decrypted_param
        global_state_dict[key] = weighted_sum
    
    global_model.load_state_dict(global_state_dict)

def federated_learning_with_early_stopping(global_model, client_data, X_test_tensor, y_test_tensor, patience=7, min_delta=0.001):
    max_rounds = 100  # Set a maximum number of rounds as a safeguard
    best_accuracy = 0
    rounds_without_improvement = 0
    accuracy_history = deque(maxlen=patience)
    encryption_simulator = EncryptionSimulator()

    for round in range(max_rounds):
        client_models = []
        client_data_sizes = []

        for client_X, client_y in client_data:
            local_model = CirrhosisPredictor(global_model.fc[0].in_features)
            local_model.load_state_dict(global_model.state_dict())
            client_model_state = train_local_model(local_model, (client_X, client_y))

            # Encrypt the client model state
            encrypted_client_model_state = {k: encrypt_vector(encryption_simulator, v.flatten()) for k, v in client_model_state.items()}
    
            client_models.append(encrypted_client_model_state)
            client_data_sizes.append(len(client_X))

        # Decrypt and aggregate models
        aggregate_models(global_model, client_models, client_data_sizes, encryption_simulator)

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