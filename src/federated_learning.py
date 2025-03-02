import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from model import CirrhosisPredictor
from evaluation import evaluate_model
from collections import deque
from encryption import EncryptionSimulator, encrypt_vector, decrypt_vector
from server_defense import FederatedDefender
from clients_defense import (
    enhanced_local_data_validation, 
    enhanced_local_model_validation,
    client_local_train  # This can be your updated training routine with defense mechanisms.
)

def train_local_model(model, data, epochs=30, lr=0.001):
    """
    Train a local model using the provided data.

    This function trains the given model using CrossEntropyLoss and AdamW optimizer.
    It uses mini-batch training with a batch size of 64.

    Args:
        model (nn.Module): The neural network model to be trained.
        data (tuple): A tuple containing training data (X, y), where X is the input features
                      and y is the corresponding labels.
        epochs (int, optional): The number of training epochs. Defaults to 30.
        lr (float, optional): The learning rate for the optimizer. Defaults to 0.001.

    Returns:
        dict: The state dictionary of the trained model.

    Note:
        - The function uses a weight decay of 1e-5 for regularization.
        - The data is shuffled before each epoch.
        - The model is set to training mode before the training loop.
    """
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
    """
    Aggregate multiple client models into a global model using federated averaging.

    This function performs a weighted average of the client models' parameters,
    where the weights are proportional to the size of each client's dataset.
    The client models' parameters are assumed to be encrypted and are decrypted
    before aggregation.

    Args:
        global_model (nn.Module): The global model to be updated with aggregated parameters.
        client_models (list): A list of state dictionaries from client models.
        client_data_sizes (list): A list of integers representing the data size for each client.
        encryption_simulator (object): An object with a decrypt_vector method for decrypting model parameters.

    Returns:
        None. The global_model is updated in-place.

    Note:
        - The function assumes that all client models have the same architecture as the global model.
        - Client model parameters are expected to be encrypted and are decrypted using the provided encryption_simulator.
        - The aggregation is done layer-wise and parameter-wise.
        - The global model's state dict is directly updated with the aggregated parameters.
    """
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

def federated_learning_with_early_stopping(
    global_model,
    client_data,
    X_test_tensor,
    y_test_tensor,
    patience=5,
    min_delta=0.001,
    enable_defense=True,
    monitor=None
):
    """
    Perform federated learning with early stopping and enhanced security measures.

    This function implements a federated learning process with various security enhancements,
    including data validation, model validation, secure aggregation, and early stopping.

    Args:
        global_model (nn.Module): The initial global model to be trained.
        client_data (list): A list of tuples, where each tuple contains client features (X) and labels (y).
        X_test_tensor (torch.Tensor): The features of the test dataset.
        y_test_tensor (torch.Tensor): The labels of the test dataset.
        patience (int, optional): The number of rounds to wait for improvement before early stopping. Defaults to 5.
        min_delta (float, optional): The minimum change in accuracy to qualify as an improvement. Defaults to 0.001.
        enable_defense (bool, optional): Whether to enable defense mechanisms. Defaults to True.
        monitor (object, optional): An object to record metrics during the training process. Defaults to None.

    Returns:
        nn.Module: The trained global model.

    Notes:
        - The function uses enhanced data validation, model validation, and secure aggregation.
        - It implements early stopping based on the test accuracy.
        - Differential privacy and adversarial training are applied during client training.
        - The function can handle up to 100 rounds of federated learning.
        - A FederatedDefender is used for secure aggregation and model verification.
        - Performance metrics are recorded if a monitor object is provided.
    """
    max_rounds = 100
    best_accuracy = 0
    rounds_without_improvement = 0
    accuracy_history = deque(maxlen=patience)
    encryption_simulator = EncryptionSimulator()
    defender = FederatedDefender(
        validation_data=(X_test_tensor, y_test_tensor),
        warmup_rounds=10,
        min_clients=max(5, len(client_data) // 4),
        encryption_simulator=encryption_simulator
    )

    monitor.record_scalability_metrics(len(client_data), sum(len(c[0]) for c in client_data))
    
    for round in range(max_rounds):
        if monitor:
            monitor.start_timer('round_time')

        client_models = []
        client_data_sizes = []
        
        # Client training phase
        for client_X, client_y in client_data:
            # Enhanced data validation step
            if not enhanced_local_data_validation(client_X, client_y):
                print("Skipping client update due to poor data quality.")
                continue
            
            # Split client data into training and validation sets (e.g., 90/10 split)
            val_split = int(0.2 * len(client_X))
            x_train, x_val = client_X[:-val_split], client_X[-val_split:]
            y_train, y_val = client_y[:-val_split], client_y[-val_split:]
            
            # Set up and initialize the client model
            local_model = CirrhosisPredictor(global_model.fc[0].in_features)
            local_model.load_state_dict(global_model.state_dict())
            
            # Train with enhanced defenses (differential privacy, adversarial training, etc.)
            client_model_state = client_local_train(
                local_model,
                data=(x_train, y_train),
                epochs=30,
                lr=0.001,
                enable_dp=True,          # Differential privacy enabled
                dp_clip=1.0,
                dp_noise_scale=0.01,
                enable_adv=True,         # Adversarial training enabled
                adv_epsilon=0.1,
                adv_ratio=0.5,
                local_val_data=(x_val, y_val)  # Validate during training
            )
            
            # Perform enhanced model validation on the client’s local validation set
            if not enhanced_local_model_validation(
                local_model, x_val, y_val,
                accuracy_threshold=0.5,
                loss_threshold=1.0,
                adv_epsilon=0.1,
                consistency_threshold=0.72
            ):
                print("Skipping client update due to model poisoning detection.")
                continue
            
            # Encrypt the local model state for secure aggregation
            encrypted_state = {
                k: encrypt_vector(encryption_simulator, v.flatten())
                for k, v in client_model_state.items()
            }
            client_models.append(encrypted_state)
            client_data_sizes.append(len(x_train))
        
        # Secure aggregation with defense mechanisms
        if enable_defense:
            global_model = defender.secure_aggregate(global_model, client_models, client_data_sizes)
        
        # Verify global model quality; rollback if necessary
        if not defender.verify_global_model(global_model):
            print("Model rollback due to verification failure")
            rounds_without_improvement += 1
            continue
        
        defender.update_reference(global_model.state_dict())

        if monitor:
            # Ensure timer is always stopped even if error occurs
            try:
                monitor.stop_timer('round_time')
            except KeyError:
                pass
        
        # Evaluate updated global model
        test_accuracy = evaluate_model(global_model, X_test_tensor, y_test_tensor)
        print(f"Round {round+1}, Test Accuracy: {test_accuracy:.4f}")

        if monitor:
            monitor.metrics['performance']['accuracy'].append(test_accuracy)
        
        # Early stopping check
        if test_accuracy > best_accuracy + min_delta:
            best_accuracy = test_accuracy
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        
        if rounds_without_improvement >= patience:
            print(f"Early stopping triggered. Best accuracy: {best_accuracy:.4f}")
            break
    
    return global_model