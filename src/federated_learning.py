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

# def federated_learning_with_early_stopping(
#     global_model, 
#     client_data, 
#     X_test_tensor, 
#     y_test_tensor, 
#     patience=7, 
#     min_delta=0.001, 
#     enable_defense=True
# ):
#     max_rounds = 100
#     best_accuracy = 0
#     rounds_without_improvement = 0
#     accuracy_history = deque(maxlen=patience)
#     encryption_simulator = EncryptionSimulator()
    
#     # Initialize defense system with same encryption simulator
#     defender = FederatedDefender(
#         validation_data=(X_test_tensor, y_test_tensor),
#         warmup_rounds=10,
#         min_clients=max(5, len(client_data)//4),
#         encryption_simulator=encryption_simulator  # Pass the same instance
#     )

#     for round in range(max_rounds):
#         client_models = []
#         client_data_sizes = []

#         # Inside your client loop in federated_learning.py:
#         for client_X, client_y in client_data:
#             # Validate local data first
#             if not local_data_validation(client_X, client_y):
#                 print("Skipping client with poor data quality.")
#                 continue

#             local_model = CirrhosisPredictor(global_model.fc[0].in_features)
#             local_model.load_state_dict(global_model.state_dict())

#             # Optionally, set aside a small part of the client’s data for local validation.
#             # For example, split client_X and client_y into training and validation:
#             val_split = int(0.1 * len(client_X))
#             x_train, x_val = client_X[:-val_split], client_X[-val_split:]
#             y_train, y_val = client_y[:-val_split], client_y[-val_split:]
            
#             # Train with defenses enabled
#             client_model_state = client_local_train(
#                 local_model,
#                 data=(x_train, y_train),
#                 epochs=30,
#                 lr=0.001,
#                 enable_dp=True,          # Differential privacy
#                 dp_clip=1.0,
#                 dp_noise_scale=0.01,
#                 enable_adv=True,         # Adversarial training
#                 adv_epsilon=0.1,
#                 adv_ratio=0.5,
#                 local_val_data=(x_val, y_val)
#             )

#         # Client training phase
#         # for client_X, client_y in client_data:
#         #     local_model = CirrhosisPredictor(global_model.fc[0].in_features)
#         #     local_model.load_state_dict(global_model.state_dict())
            
#         #     # Train and encrypt model
#         #     client_model_state = train_local_model(local_model, (client_X, client_y))
#             encrypted_state = {
#                 k: encrypt_vector(encryption_simulator, v.flatten()) 
#                 for k, v in client_model_state.items()
#             }
#             client_models.append(encrypted_state)
#             client_data_sizes.append(len(client_X))

#         # Secure aggregation with defense mechanisms
#         if enable_defense:
#             global_model = defender.secure_aggregate(
#                 global_model, 
#                 client_models, 
#                 client_data_sizes
#             )
            
#             # Model verification and rollback if needed
#             if not defender.verify_global_model(global_model):
#                 print("Model rollback due to verification failure")
#                 rounds_without_improvement += 1
#                 continue

#         # Update reference models for adaptive defense
#         defender.update_reference(global_model.state_dict())

#         # Model evaluation
#         test_accuracy = evaluate_model(global_model, X_test_tensor, y_test_tensor)
#         print(f"Round {round+1}, Test Accuracy: {test_accuracy:.4f}")

#         # Early stopping logic
#         if test_accuracy > best_accuracy + min_delta:
#             best_accuracy = test_accuracy
#             rounds_without_improvement = 0
#         else:
#             rounds_without_improvement += 1

#         if rounds_without_improvement >= patience:
#             print(f"Early stopping triggered. Best accuracy: {best_accuracy:.4f}")
#             break

#     return global_model

def federated_learning_with_early_stopping(
    global_model,
    client_data,
    X_test_tensor,
    y_test_tensor,
    patience=5,
    min_delta=0.001,
    enable_defense=True
):
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
    
    for round in range(max_rounds):
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
        
        # Evaluate updated global model
        test_accuracy = evaluate_model(global_model, X_test_tensor, y_test_tensor)
        print(f"Round {round+1}, Test Accuracy: {test_accuracy:.4f}")
        
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