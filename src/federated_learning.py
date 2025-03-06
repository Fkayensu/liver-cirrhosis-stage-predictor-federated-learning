import torch
import random
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
from attack_simulation import (
    model_poisoning_attack,
    backdoor_attack,
    data_poisoning_attack,
    mitm_attack
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
    global_state_dict = global_model.state_dict()
    total_data = sum(client_data_sizes)
    
    for key in global_state_dict.keys():
        if isinstance(global_state_dict[key], bool):
            # For boolean values, use majority voting
            votes = [model[key] for model in client_models]
            global_state_dict[key] = sum(votes) > len(votes) / 2
        else:
            weighted_sum = torch.zeros_like(global_state_dict[key])
            for client_model, data_size in zip(client_models, client_data_sizes):
                weight = data_size / total_data
                if isinstance(client_model[key], bool):
                    decrypted_param = client_model[key]
                else:
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
    monitor=None,
    simulate_attacks=False  # New flag to control attack simulation
):
    """
    Perform federated learning with early stopping and enhanced security measures.

    Args:
        global_model (nn.Module): The initial global model to be trained.
        client_data (list): A list of tuples, where each tuple contains client features (X) and labels (y).
        X_test_tensor (torch.Tensor): The features of the test dataset.
        y_test_tensor (torch.Tensor): The labels of the test dataset.
        patience (int, optional): The number of rounds to wait for improvement before early stopping. Defaults to 5.
        min_delta (float, optional): The minimum change in accuracy to qualify as an improvement. Defaults to 0.001.
        enable_defense (bool, optional): Whether to enable defense mechanisms. Defaults to True.
        monitor (object, optional): An object to record metrics during the training process. Defaults to None.
        simulate_attacks (bool, optional): Whether to simulate attacks during training. Defaults to False.

    Returns:
        nn.Module: The trained global model.
    """
    
    max_rounds = 100
    rollback_count = 0
    max_rollbacks = 3
    best_accuracy = 0
    rounds_without_improvement = 0
    accuracy_history = deque(maxlen=patience)
    encryption_simulator = EncryptionSimulator()
    
    defender = FederatedDefender(
        validation_data=(X_test_tensor, y_test_tensor),
        warmup_rounds=3,
        min_clients=max(5, len(client_data) // 4),
        encryption_simulator=encryption_simulator,
        sensitivity=5,
        monitor=monitor
    )

    monitor.record_scalability_metrics(len(client_data), sum(len(c[0]) for c in client_data))

    for round in range(max_rounds):
        print(f"\nStarting Round {round+1}")
        attack_probability = min(0.1 + (round / 50), 0.1)  # Cap at 30%
        if monitor:
            monitor.start_timer('round_time')

        client_models = []
        client_data_sizes = []

        for client_id, (client_X, client_y) in enumerate(client_data):
            attack_info = {'is_malicious': False}  # Default no attack
            print(f"Client {client_id + 1} training:")

            if simulate_attacks:
                # Data Poisoning Attack
                if random.random() < attack_probability:
                    client_X, client_y, attack_info = data_poisoning_attack(client_X, client_y, poison_ratio=0.1)
                    monitor.record_attack('data_poisoning')

                # Backdoor Attack
                if random.random() < attack_probability:
                    trigger_pattern = torch.ones_like(client_X[0]) * 0.1
                    target_label = 0
                    client_X, client_y, attack_info = backdoor_attack(client_X, client_y, trigger_pattern, target_label, backdoor_ratio=0.1)
                    monitor.record_attack('backdoor')

            # Enhanced data validation step
            if not enhanced_local_data_validation(client_X, client_y):
                print(f"Skipping client {client_id} update due to poor data quality.")
                continue

            # Split client data into training and validation sets
            val_split = int(0.2 * len(client_X))
            x_train, x_val = client_X[:-val_split], client_X[-val_split:]
            y_train, y_val = client_y[:-val_split], client_y[-val_split:]

            local_model = CirrhosisPredictor(global_model.fc[0].in_features)
            local_model.load_state_dict(global_model.state_dict())

            client_model_state = client_local_train(
                local_model,
                data=(x_train, y_train),
                epochs=75,
                lr=0.005,
                enable_dp=True,
                dp_clip=1.0,
                dp_noise_scale=0.01,
                enable_adv=True,
                adv_epsilon=0.1,
                adv_ratio=0.5,
                local_val_data=(x_val, y_val)
            )

            # Model Poisoning Attack
            if simulate_attacks and random.random() < attack_probability:
                client_model_state = model_poisoning_attack(client_model_state, attack_strength=2.0)
                attack_info['is_malicious'] = True  # Update attack info
                monitor.record_attack('model_poisoning')

            # Enhanced model validation
            if not enhanced_local_model_validation(
                local_model, x_val, y_val,
                accuracy_threshold=0.5,
                loss_threshold=1.0,
                adv_epsilon=0.1,
                consistency_threshold=0.72,
                monitor=monitor
            ):
                print(f"Skipping client {client_id} update due to model poisoning detection.")
                continue

            # Encrypt the local model state with attack info
            encrypted_state = {
                k: encrypt_vector(encryption_simulator, v.flatten()) if not isinstance(v, bool) else v
                for k, v in client_model_state.items()
            }

            encrypted_state.update(attack_info)  # Add attack info (e.g., is_malicious)

            # MITM Attack
            if simulate_attacks and random.random() < attack_probability:
                encrypted_state = mitm_attack(encrypted_state, attack_strength=2.0)
                monitor.record_attack('mitm')

                if 'is_malicious' not in encrypted_state:
                    encrypted_state['is_malicious'] = True

            client_models.append(encrypted_state)
            client_data_sizes.append(len(x_train))

        # Secure aggregation with defense mechanisms
        if enable_defense:
            filtered_models = defender.analyze_models(client_models, encryption_simulator)  # Analyze before aggregation
            filtered_sizes = [client_data_sizes[i] for i, m in enumerate(client_models) if m in filtered_models]
            if not filtered_models:
                print(f"Round {round+1}: No valid client updates, skipping aggregation")
                continue
            global_model = defender.secure_aggregate(global_model, filtered_models, filtered_sizes)
            if enable_defense and not defender.verify_global_model(global_model) and rollback_count < max_rollbacks:
                print("Model rollback due to verification failure")
                if defender.best_global_model:
                    global_model.load_state_dict(defender.best_global_model)
                rollback_count += 1
                # Allow partial update with filtered models if any
                if filtered_models:
                    global_model = defender.secure_aggregate(global_model, filtered_models, filtered_sizes)
                continue
        else:
            aggregate_models(global_model, client_models, client_data_sizes, encryption_simulator)

        # Verify global model quality; rollback if necessary
        if enable_defense and not defender.verify_global_model(global_model):
            print("Model rollback due to verification failure")
            rounds_without_improvement += 1
            continue

        if enable_defense:
            defender.update_reference(global_model.state_dict())

        if monitor:
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
        
        print(f"Round {round+1} completed. Test Accuracy: {test_accuracy:.4f}")
        print(f"Current best accuracy: {best_accuracy:.4f}")
        print(f"Rounds without improvement: {rounds_without_improvement}")

        if rounds_without_improvement >= patience:
            print(f"Early stopping triggered. Best accuracy: {best_accuracy:.4f}")
            break

    return global_model
