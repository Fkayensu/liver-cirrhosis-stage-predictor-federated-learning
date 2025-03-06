import torch
import numpy as np
from scipy.stats import median_abs_deviation
from encryption import decrypt_vector

class FederatedDefender:
    """
    A class for implementing defense mechanisms in federated learning.

    This class provides methods for secure aggregation, model verification,
    and Byzantine-robust federated learning. It includes features such as
    encrypted model handling, dynamic thresholding for outlier detection,
    and model quality verification.

    Attributes:
        sensitivity (float): Sensitivity parameter for outlier detection.
        warmup_rounds (int): Number of initial rounds before applying full defense.
        min_clients (int): Minimum number of clients to retain after filtering.
        reference_models (list): List of recent model states for comparison.
        adaptive_threshold (float): Threshold for adaptive model filtering.
        validation_data (tuple): Tuple of (X_val, y_val) for model verification.
        best_global_model (dict): State dict of the best performing global model.
        accuracy_history (list): History of model accuracies.
        encryption_simulator (object): Object for simulating encryption/decryption.
        monitor (object): Object for monitoring and logging metrics.
        trim_percentage (float): Percentage of models to trim in robust aggregation.
        accuracy_tolerance (float): Tolerance for accuracy degradation.
        rollback_patience (int): Number of rounds to wait before rolling back.
        shapes (dict): Registry of parameter shapes for each layer.

    Methods:
        analyze_models: Analyze and filter client models.
        secure_aggregate: Perform secure and robust model aggregation.
        verify_global_model: Verify the quality of the global model.
        update_reference: Update the reference models for comparison.
    """
    def __init__(self, encryption_simulator, sensitivity=5.0, warmup_rounds=3, min_clients=5, validation_data=None, monitor=None):
        """
        Initialize the FederatedDefender.

        Args:
            encryption_simulator (object): Simulator for encryption/decryption operations.
            sensitivity (float, optional): Sensitivity for outlier detection. Defaults to 3.5.
            warmup_rounds (int, optional): Number of initial rounds before full defense. Defaults to 10.
            min_clients (int, optional): Minimum number of clients to retain. Defaults to 5.
            validation_data (tuple, optional): Validation data for model verification. Defaults to None.
            monitor (object, optional): Object for monitoring metrics. Defaults to None.
        """
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
        self.attack_history = []
        
        # Byzantine-robust parameters
        self.trim_percentage = 0.2
        self.accuracy_tolerance = 0.1
        self.rollback_patience = 5

        # Initialize shape registry to track parameter dimensions
        self.shapes = None

    def update_defense_parameters(self):
        recent_attacks = self.attack_history[-10:]
        if sum(recent_attacks) > 5:
            self.sensitivity *= 1.1  # Increase sensitivity if many recent attacks
        else:
            self.sensitivity = max(self.sensitivity * 0.9, 3.0)  # Decrease, but not below 3.0

    def _decrypt_model(self, encrypted_model):
        decrypted = {}
        for k, v in encrypted_model.items():
            if isinstance(v, bool):
                decrypted[k] = v
            else:
                try:
                    if not isinstance(v, dict) or 'data' not in v:
                        print(f"Warning: Invalid encrypted parameter {k}: {v}")
                        continue
                    decrypted_vec, is_malicious = decrypt_vector(self.encryption_simulator, v)
                    if not isinstance(decrypted_vec, (list, tuple, np.ndarray)):
                        print(f"Warning: Decrypted vector for {k} is invalid: {decrypted_vec}")
                        continue
                    if self.shapes is not None and k in self.shapes:
                        target_shape = self.shapes[k]
                        decrypted[k] = torch.tensor(decrypted_vec).float().reshape(target_shape)
                    else:
                        decrypted[k] = torch.tensor(decrypted_vec).float().reshape(-1)
                except Exception as e:
                    print(f"Error decrypting parameter {k}: {e}")
                    continue
        return decrypted if decrypted else None

    def _robust_zscore(self, values):
        """
        Calculate robust z-scores using median and median absolute deviation.

        Args:
            values (numpy.array): Array of values to calculate z-scores for.

        Returns:
            numpy.array: Array of robust z-scores.

        This method is more resistant to outliers compared to standard z-scores.
        """
        median = np.median(values)
        mad = median_abs_deviation(values)
        mad = max(mad, 1e-2) # Prevent division by near-zero MAD
        return np.abs((values - median) / (mad + 1e-8))

    def analyze_models(self, client_models, encryption_simulator):
        print("Entering analyze_models")
        if self.monitor:
            self.monitor.start_timer('model_analysis')

        decrypted_models = []
        for i, m in enumerate(client_models):
            decrypted = self._decrypt_model(m)
            if decrypted is None or not decrypted:
                print(f"Client {i}: Decryption failed or empty")
                continue
            decrypted_models.append(decrypted)
        
        if not decrypted_models:
            print("No valid decrypted models")
            return client_models
        
        try:
            param_vectors = [
                torch.cat([param.view(-1) for param in m.values() if isinstance(param, torch.Tensor)])
                for m in decrypted_models
            ]
            param_matrix = torch.stack(param_vectors).numpy()

            global_norm = np.linalg.norm(torch.cat([p.view(-1) for p in self.best_global_model.values() if isinstance(p, torch.Tensor)]).numpy())
            print(f"Global model norm: {global_norm}")
            # Debug: Check parameter norms
            norms = [np.linalg.norm(vec) for vec in param_matrix]
            print(f"Parameter norms: {norms}")
            client_scores = [abs(norm - global_norm) for norm in norms]
            print(f"Client scores (vs global): {client_scores}")
        except Exception as e:
            print(f"Error constructing param matrix: {e}")
            return client_models

        # Use norm differences from median instead of z-scores
        median_norm = np.median(norms)
        client_scores = [abs(norm - median_norm) for norm in norms]
        print(f"Client scores (norm differences): {client_scores}")
        
        valid_indices = self._dynamic_thresholding(np.array(client_scores))
        print(f"Valid indices: {valid_indices}")

        if self.monitor:
            self.monitor.stop_timer('model_analysis')
            print("Recording security events")
            for i, model in enumerate(client_models):
                is_attack = model.get('is_malicious', False)
                detected = i not in valid_indices
                print(f"Client {i}: is_attack={is_attack}, detected={detected}")
                self.monitor.record_security_event(i, is_attack, detected)

        self.attack_history.append(len(valid_indices) < len(client_models))
        self.update_defense_parameters()

        return [client_models[i] for i in valid_indices]

    def _dynamic_thresholding(self, scores):
        if len(self.reference_models) < self.warmup_rounds:
            print("Warmup active, no filtering")
            return np.arange(len(scores))
        
        q75 = np.percentile(scores, 75)
        q25 = np.percentile(scores, 25)
        iqr = max(q75 - q25, 1e-2)
        iqr = min(iqr, 5.0)  # Tighter cap for better sensitivity
        median_score = np.median(scores)
        upper_bound = median_score + 1.5 * iqr  # Reduce multiplier to 2.0
        print(f"Q25: {q25}, Q75: {q75}, IQR: {iqr}, Median: {median_score}, Upper bound: {upper_bound}")
        
        valid = np.where(scores <= upper_bound)[0]
        if len(valid) < self.min_clients:
            valid = np.argsort(scores)[:self.min_clients]
        return valid

    # Byzantine-robust aggregation methods
    def secure_aggregate(self, global_model, client_models, client_data_sizes):
        """
        Perform Byzantine-robust model aggregation with shape preservation.

        Args:
            global_model (nn.Module): The current global model.
            client_models (list): List of client model updates.
            client_data_sizes (list): List of data sizes for each client.

        Returns:
            nn.Module: Updated global model after secure aggregation.

        This method uses trimmed mean aggregation and momentum stabilization
        to robustly aggregate client updates into the global model.
        """
        if not client_models:
            print("No client models to aggregate, returning current global model")
            return global_model
        if self.monitor:
            self.monitor.start_timer('secure_aggregation')

        # Initialize the shape registry from the global model's state_dict if not already set
        if self.shapes is None:
            self.shapes = {k: v.shape for k, v in global_model.state_dict().items()}
        
        decrypted_models = [self._decrypt_model(m) for m in client_models]
        global_state = global_model.state_dict()
        
        # Trimmed mean aggregation for each parameter key
        for key in global_state.keys():
            all_updates = torch.stack([m[key] for m in decrypted_models])
            sorted_updates, _ = torch.sort(all_updates, dim=0)
            trim_count = int(self.trim_percentage * len(client_models))
            
            if trim_count > 0:
                trimmed = sorted_updates[trim_count:-trim_count]
            else:
                trimmed = sorted_updates
                
            global_state[key] = trimmed.mean(dim=0)
        
        # Apply momentum stabilization if a best global model exists
        if self.best_global_model:
            for key in global_state:
                global_state[key] = 0.9 * global_state[key] + 0.1 * self.best_global_model[key]
        
        global_model.load_state_dict(global_state)

        if self.monitor:
            self.monitor.stop_timer('secure_aggregation')

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

        self.accuracy_history.append(accuracy)
        median_acc = np.median(self.accuracy_history[-self.rollback_patience:]) if len(self.accuracy_history) >= self.rollback_patience else accuracy

        # Soft rollback: blend current model with best if accuracy drops significantly
        if self.best_global_model and accuracy < median_acc - self.accuracy_tolerance:
            print(f"Soft rollback triggered: Accuracy {accuracy:.4f} < {median_acc - self.accuracy_tolerance:.4f}")
            current_state = model.state_dict()
            best_state = self.best_global_model
            blended_state = {}
            for key in current_state:
                blended_state[key] = 0.9 * best_state[key] + 0.1 * current_state[key]  # Blend weights
            model.load_state_dict(blended_state)
            return False

        if not self.best_global_model or accuracy > max(self.accuracy_history[:-1], default=0):
            self.best_global_model = model.state_dict()

        return True

    def update_reference(self, model_state):
        """
        Update the list of reference models used for comparison.

        Args:
            model_state (dict): The state dictionary of the current global model.

        This method maintains a rolling window of recent model states for use
        in dynamic thresholding and other comparative analyses.
        """
        if len(self.reference_models) >= self.warmup_rounds:
            self.reference_models.pop(0)
        self.reference_models.append(model_state)