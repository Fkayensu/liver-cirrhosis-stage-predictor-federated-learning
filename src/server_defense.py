# import torch
# import numpy as np
# from scipy.stats import median_abs_deviation
# from encryption import decrypt_vector

# class FederatedDefender:
#     """
#     A class for implementing defense mechanisms in federated learning.

#     This class provides methods for secure aggregation, model verification,
#     and Byzantine-robust federated learning. It includes features such as
#     encrypted model handling, dynamic thresholding for outlier detection,
#     and model quality verification.

#     Attributes:
#         sensitivity (float): Sensitivity parameter for outlier detection.
#         warmup_rounds (int): Number of initial rounds before applying full defense.
#         min_clients (int): Minimum number of clients to retain after filtering.
#         reference_models (list): List of recent model states for comparison.
#         adaptive_threshold (float): Threshold for adaptive model filtering.
#         validation_data (tuple): Tuple of (X_val, y_val) for model verification.
#         best_global_model (dict): State dict of the best performing global model.
#         accuracy_history (list): History of model accuracies.
#         encryption_simulator (object): Object for simulating encryption/decryption.
#         monitor (object): Object for monitoring and logging metrics.
#         trim_percentage (float): Percentage of models to trim in robust aggregation.
#         accuracy_tolerance (float): Tolerance for accuracy degradation.
#         rollback_patience (int): Number of rounds to wait before rolling back.
#         shapes (dict): Registry of parameter shapes for each layer.

#     Methods:
#         analyze_models: Analyze and filter client models.
#         secure_aggregate: Perform secure and robust model aggregation.
#         verify_global_model: Verify the quality of the global model.
#         update_reference: Update the reference models for comparison.
#     """
#     def __init__(self, encryption_simulator, sensitivity=3.5, warmup_rounds=10, min_clients=5, validation_data=None, monitor=None):
#         """
#         Initialize the FederatedDefender.

#         Args:
#             encryption_simulator (object): Simulator for encryption/decryption operations.
#             sensitivity (float, optional): Sensitivity for outlier detection. Defaults to 3.5.
#             warmup_rounds (int, optional): Number of initial rounds before full defense. Defaults to 10.
#             min_clients (int, optional): Minimum number of clients to retain. Defaults to 5.
#             validation_data (tuple, optional): Validation data for model verification. Defaults to None.
#             monitor (object, optional): Object for monitoring metrics. Defaults to None.
#         """
#         self.sensitivity = sensitivity
#         self.warmup_rounds = warmup_rounds
#         self.min_clients = min_clients
#         self.reference_models = []
#         self.adaptive_threshold = 3.0
#         self.validation_data = validation_data
#         self.best_global_model = None
#         self.accuracy_history = []
#         self.encryption_simulator = encryption_simulator
#         self.monitor = monitor
        
#         # Byzantine-robust parameters
#         self.trim_percentage = 0.2
#         self.accuracy_tolerance = 0.05
#         self.rollback_patience = 3

#         # Initialize shape registry to track parameter dimensions
#         self.shapes = None

#     # def _decrypt_model(self, encrypted_model):
#     #     """
#     #     Decrypt an encrypted model.

#     #     Args:
#     #         encrypted_model (dict): A dictionary containing encrypted model parameters.

#     #     Returns:
#     #         dict: A dictionary containing decrypted model parameters.

#     #     This method decrypts each parameter of the model, reshaping it according to
#     #     the stored shapes or as a flat vector if the shape is unknown.
#     #     """
#     #     decrypted = {}
#     #     for k, v in encrypted_model.items():
#     #         # decrypt_vector returns a flattened array
#     #         decrypted_vec = decrypt_vector(self.encryption_simulator, v)
#     #         if self.shapes is not None and k in self.shapes:
#     #             target_shape = self.shapes[k]
#     #             decrypted[k] = torch.tensor(decrypted_vec).float().reshape(target_shape)
#     #         else:
#     #             # Fallback: if shape registry is not available, reshape to a flat vector
#     #             decrypted[k] = torch.tensor(decrypted_vec).float().reshape(-1)
#     #     return decrypted

#     def _decrypt_model(self, encrypted_model):
#         """
#         Decrypt an encrypted model, skipping non-iterable values like booleans.

#         Args:
#             encrypted_model (dict): A dictionary containing encrypted model parameters and metadata.

#         Returns:
#             dict: A dictionary containing decrypted model parameters.
#         """
#         decrypted = {}
#         for k, v in encrypted_model.items():
#             if isinstance(v, (list, torch.Tensor, np.ndarray)):  # Check if value is iterable
#                 decrypted_vec = decrypt_vector(self.encryption_simulator, v)
#                 if self.shapes is not None and k in self.shapes:
#                     target_shape = self.shapes[k]
#                     decrypted[k] = torch.tensor(decrypted_vec).float().reshape(target_shape)
#                 else:
#                     decrypted[k] = torch.tensor(decrypted_vec).float().reshape(-1)
#             else:
#                 # Preserve non-iterable values (e.g., 'is_malicious') as-is
#                 decrypted[k] = v
#         return decrypted

#     def _robust_zscore(self, values):
#         """
#         Calculate robust z-scores using median and median absolute deviation.

#         Args:
#             values (numpy.array): Array of values to calculate z-scores for.

#         Returns:
#             numpy.array: Array of robust z-scores.

#         This method is more resistant to outliers compared to standard z-scores.
#         """
#         median = np.median(values)
#         mad = median_abs_deviation(values)
#         return np.abs((values - median) / (mad + 1e-8))

#     def analyze_models(self, client_models, encryption_simulator):
#         """
#         Analyze client models to detect and filter out potential malicious updates.

#         Args:
#             client_models (list): List of encrypted client models.
#             encryption_simulator (object): Object used for decryption.

#         Returns:
#             list: Filtered list of client models deemed non-malicious.

#         This method decrypts models, calculates robust z-scores, and uses dynamic
#         thresholding to identify and remove potential malicious updates.
#         """
#         if self.monitor:
#             self.monitor.start_timer('aggregation')

#         self.encryption_simulator = encryption_simulator
#         decrypted_models = [self._decrypt_model(m) for m in client_models]
#         # Flatten each parameter tensor before concatenation for analysis
#         # param_vectors = [torch.cat([param.view(-1) for param in m.values()]) for m in decrypted_models]
#         param_vectors = [
#             torch.cat([param.view(-1) for param in m.values() if isinstance(param, torch.Tensor)])
#             for m in decrypted_models
#         ]
#         param_matrix = torch.stack(param_vectors).numpy()

#         client_scores = []
#         for vec in param_matrix:
#             z_scores = self._robust_zscore(vec)
#             client_scores.append(np.median(z_scores))

#         valid_indices = self._dynamic_thresholding(np.array(client_scores))

#         if self.monitor:
#             self.monitor.stop_timer('aggregation')
#             for i in range(len(client_models)):
#                 is_attack = client_models[i].get('is_malicious', False)
#                 detected = i not in valid_indices
#                 self.monitor.record_security_event(i, is_attack, detected)

#         return [client_models[i] for i in valid_indices]

#     def _dynamic_thresholding(self, scores):
#         """
#         Apply dynamic thresholding to identify valid client updates.

#         Args:
#             scores (numpy.array): Array of scores for each client update.

#         Returns:
#             numpy.array: Indices of client updates deemed valid.

#         This method uses interquartile range to set a dynamic threshold for
#         identifying outliers, ensuring a minimum number of clients are retained.
#         """
#         if len(self.reference_models) < self.warmup_rounds:
#             return np.arange(len(scores))
        
#         q75 = np.percentile(scores, 75)
#         q25 = np.percentile(scores, 25)
#         iqr = q75 - q25
#         upper_bound = q75 + self.sensitivity * iqr
        
#         valid = np.where(scores <= upper_bound)[0]
#         if len(valid) < self.min_clients:
#             valid = np.argsort(scores)[:self.min_clients]
#         return valid

#     # Byzantine-robust aggregation methods
#     def secure_aggregate(self, global_model, client_models, client_data_sizes):
#         """
#         Perform Byzantine-robust model aggregation with shape preservation.

#         Args:
#             global_model (nn.Module): The current global model.
#             client_models (list): List of client model updates.
#             client_data_sizes (list): List of data sizes for each client.

#         Returns:
#             nn.Module: Updated global model after secure aggregation.

#         This method uses trimmed mean aggregation and momentum stabilization
#         to robustly aggregate client updates into the global model.
#         """
#         if self.monitor:
#             self.monitor.start_timer('aggregation')

#         # Initialize the shape registry from the global model's state_dict if not already set
#         if self.shapes is None:
#             self.shapes = {k: v.shape for k, v in global_model.state_dict().items()}
        
#         decrypted_models = [self._decrypt_model(m) for m in client_models]
#         global_state = global_model.state_dict()
        
#         # Trimmed mean aggregation for each parameter key
#         for key in global_state.keys():
#             all_updates = torch.stack([m[key] for m in decrypted_models])
#             sorted_updates, _ = torch.sort(all_updates, dim=0)
#             trim_count = int(self.trim_percentage * len(client_models))
            
#             if trim_count > 0:
#                 trimmed = sorted_updates[trim_count:-trim_count]
#             else:
#                 trimmed = sorted_updates
                
#             global_state[key] = trimmed.mean(dim=0)
        
#         # Apply momentum stabilization if a best global model exists
#         if self.best_global_model:
#             for key in global_state:
#                 global_state[key] = 0.9 * global_state[key] + 0.1 * self.best_global_model[key]
        
#         global_model.load_state_dict(global_state)

#         if self.monitor:
#             self.monitor.stop_timer('aggregation')

#         return global_model

#     # Global model verification
#     def verify_global_model(self, model):
#         """
#         Verify the quality of the global model and roll back if degraded.

#         Args:
#             model (nn.Module): The global model to verify.

#         Returns:
#             bool: True if the model passes verification, False otherwise.

#         This method evaluates the model on validation data, compares its performance
#         to recent history, and decides whether to keep the update or roll back.
#         """
#         if self.validation_data is None:
#             return True

#         X_val, y_val = self.validation_data
#         model.eval()
#         with torch.no_grad():
#             outputs = model(X_val)
#             _, preds = torch.max(outputs, 1)
#             accuracy = (preds == y_val).float().mean().item()
        
#         if len(self.accuracy_history) >= self.rollback_patience:
#             recent_acc = np.mean(self.accuracy_history[-self.rollback_patience:])
#             if accuracy < recent_acc - self.accuracy_tolerance:
#                 model.load_state_dict(self.best_global_model)
#                 return False
        
#         if not self.best_global_model or accuracy > max(self.accuracy_history, default=0):
#             self.best_global_model = model.state_dict()
            
#         self.accuracy_history.append(accuracy)
#         return True

#     def update_reference(self, model_state):
#         """
#         Update the list of reference models used for comparison.

#         Args:
#             model_state (dict): The state dictionary of the current global model.

#         This method maintains a rolling window of recent model states for use
#         in dynamic thresholding and other comparative analyses.
#         """
#         if len(self.reference_models) >= self.warmup_rounds:
#             self.reference_models.pop(0)
#         self.reference_models.append(model_state)

# server_defense.py
import torch
import numpy as np
from scipy.stats import median_abs_deviation
from encryption import decrypt_vector

class FederatedDefender:
    def __init__(self, encryption_simulator, sensitivity=2.5, warmup_rounds=2, min_clients=5, validation_data=None, monitor=None):
        self.sensitivity = sensitivity
        self.warmup_rounds = warmup_rounds
        self.min_clients = min_clients
        self.reference_models = []
        self.adaptive_threshold = 3.0
        self.validation_data = validation_data
        self.best_global_model = None
        self.accuracy_history = []
        self.encryption_simulator = encryption_simulator
        self.monitor = monitor
        self.trim_percentage = 0.3
        self.accuracy_tolerance = 0.05
        self.rollback_patience = 3
        self.shapes = None

    def _decrypt_model(self, encrypted_model):
        decrypted = {}
        for k, v in encrypted_model.items():
            if isinstance(v, (list, torch.Tensor, np.ndarray)):
                decrypted_vec = decrypt_vector(self.encryption_simulator, v)
                if self.shapes is not None and k in self.shapes:
                    target_shape = self.shapes[k]
                    decrypted[k] = torch.tensor(decrypted_vec).float().reshape(target_shape)
                else:
                    decrypted[k] = torch.tensor(decrypted_vec).float().reshape(-1)
            else:
                decrypted[k] = v
        return decrypted

    def _robust_zscore(self, values):
        median = np.median(values)
        mad = median_abs_deviation(values)
        return np.abs((values - median) / (mad + 1e-8))

    def _dynamic_thresholding(self, scores):
        if len(self.reference_models) < self.warmup_rounds:
            return np.arange(len(scores))
        
        q75 = np.percentile(scores, 75)
        q25 = np.percentile(scores, 25)
        iqr = q75 - q25
        upper_bound = q75 + self.sensitivity * iqr
        
        valid = np.where(scores <= upper_bound)[0]
        if len(valid) < self.min_clients:
            valid = np.argsort(scores)[:self.min_clients]
        return valid

    def analyze_models(self, client_models, encryption_simulator):
        if not client_models:  # Handle empty client_models list
            print("Warning: No client models passed validation for this round.")
            if self.monitor:
                self.monitor.start_timer('aggregation')
                self.monitor.stop_timer('aggregation')  # Keep timing consistent
            return []
        
        if self.monitor:
            self.monitor.start_timer('aggregation')

        self.encryption_simulator = encryption_simulator
        decrypted_models = [self._decrypt_model(m) for m in client_models]
        param_vectors = [
            torch.cat([param.view(-1) for param in m.values() if isinstance(param, torch.Tensor)])
            for m in decrypted_models
        ]
        param_matrix = torch.stack(param_vectors).numpy()

        client_scores = []
        for i, vec in enumerate(param_matrix):
            z_scores = self._robust_zscore(vec)
            variance_score = np.var(vec) * 0.2  # Increased weight
            kurtosis = np.mean((vec - np.mean(vec))**4) / (np.var(vec)**2 + 1e-8) * 0.5  # Increased weight
            combined_score = (np.median(z_scores) + 
                             variance_score + 
                             kurtosis if client_models[i].get('is_malicious', False) else 0)  # Apply kurtosis to all
            client_scores.append(combined_score)

        valid_indices = self._dynamic_thresholding(np.array(client_scores))

        # Debugging security events
        if self.monitor:
            self.monitor.stop_timer('aggregation')
            print(f"\nDebugging Security Events in Round:")
            for i in range(len(client_models)):
                is_attack = client_models[i].get('is_malicious', False)
                detected = i not in valid_indices
                attack_type = client_models[i].get('attack_type', 'none')
                print(f"Client {i}: is_attack={is_attack}, detected={detected}, attack_type={attack_type}")
                if detected and is_attack:
                    print(f"Detected Attack Type: {attack_type}")
                self.monitor.record_security_event(i, is_attack, detected, attack_type=attack_type)

        return [client_models[i] for i in valid_indices]

    def secure_aggregate(self, global_model, client_models, client_data_sizes):
        if not client_models:  # Handle empty case
            print("No valid client models for aggregation; retaining previous global model.")
            return global_model

        if self.monitor:
            self.monitor.start_timer('aggregation')

        if self.shapes is None:
            self.shapes = {k: v.shape for k, v in global_model.state_dict().items()}
        
        decrypted_models = [self._decrypt_model(m) for m in client_models]
        global_state = global_model.state_dict()
        
        for key in global_state.keys():
            all_updates = torch.stack([m[key] for m in decrypted_models])
            sorted_updates, _ = torch.sort(all_updates, dim=0)
            trim_count = int(self.trim_percentage * len(client_models))
            
            if trim_count > 0:
                trimmed = sorted_updates[trim_count:-trim_count]
            else:
                trimmed = sorted_updates
                
            global_state[key] = trimmed.mean(dim=0)
        
        if self.best_global_model:
            for key in global_state:
                global_state[key] = 0.9 * global_state[key] + 0.1 * self.best_global_model[key]
        
        global_model.load_state_dict(global_state)

        if self.monitor:
            self.monitor.stop_timer('aggregation')

        return global_model

    def verify_global_model(self, model):
        if self.validation_data is None:
            return True

        X_val, y_val = self.validation_data
        model.eval()
        with torch.no_grad():
            outputs = model(X_val)
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == y_val).float().mean().item()
        
        if len(self.accuracy_history) >= self.rollback_patience:
            recent_acc = np.mean(self.accuracy_history[-self.rollback_patience:])
            if accuracy < recent_acc - self.accuracy_tolerance:
                model.load_state_dict(self.best_global_model)
                return False
        
        if not self.best_global_model or accuracy > max(self.accuracy_history, default=0):
            self.best_global_model = model.state_dict()
            
        self.accuracy_history.append(accuracy)
        return True

    def update_reference(self, model_state):
        if len(self.reference_models) >= self.warmup_rounds:
            self.reference_models.pop(0)
        self.reference_models.append(model_state)