import torch
import numpy as np
from scipy.stats import median_abs_deviation
from encryption import decrypt_vector

class FederatedDefender:
    def __init__(self, encryption_simulator, sensitivity=3.5, warmup_rounds=10, min_clients=5, validation_data=None):
        self.sensitivity = sensitivity
        self.warmup_rounds = warmup_rounds
        self.min_clients = min_clients
        self.reference_models = []
        self.adaptive_threshold = 3.0
        self.validation_data = validation_data
        self.best_global_model = None
        self.accuracy_history = []
        self.encryption_simulator = encryption_simulator
        
        # Byzantine-robust parameters
        self.trim_percentage = 0.2
        self.accuracy_tolerance = 0.05
        self.rollback_patience = 3

        # Initialize shape registry to track parameter dimensions
        self.shapes = None

    def _decrypt_model(self, encrypted_model):
        decrypted = {}
        for k, v in encrypted_model.items():
            # decrypt_vector returns a flattened array
            decrypted_vec = decrypt_vector(self.encryption_simulator, v)
            if self.shapes is not None and k in self.shapes:
                target_shape = self.shapes[k]
                decrypted[k] = torch.tensor(decrypted_vec).float().reshape(target_shape)
            else:
                # Fallback: if shape registry is not available, reshape to a flat vector
                decrypted[k] = torch.tensor(decrypted_vec).float().reshape(-1)
        return decrypted

    def _robust_zscore(self, values):
        median = np.median(values)
        mad = median_abs_deviation(values)
        return np.abs((values - median) / (mad + 1e-8))

    def analyze_models(self, client_models, encryption_simulator):
        self.encryption_simulator = encryption_simulator
        decrypted_models = [self._decrypt_model(m) for m in client_models]
        # Flatten each parameter tensor before concatenation for analysis
        param_vectors = [torch.cat([param.view(-1) for param in m.values()]) for m in decrypted_models]
        param_matrix = torch.stack(param_vectors).numpy()

        client_scores = []
        for vec in param_matrix:
            z_scores = self._robust_zscore(vec)
            client_scores.append(np.median(z_scores))

        valid_indices = self._dynamic_thresholding(np.array(client_scores))
        return [client_models[i] for i in valid_indices]

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

    # Byzantine-robust aggregation methods
    def secure_aggregate(self, global_model, client_models, client_data_sizes):
        """Perform Byzantine-robust model aggregation with shape preservation"""
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
        return global_model

    # Global model verification
    def verify_global_model(self, model):
        """Verify model quality and roll back if degraded"""
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