import torch
import numpy as np
from scipy.stats import median_abs_deviation
from encryption import decrypt_vector

class FederatedDefender:
    def __init__(self, sensitivity=3.5, warmup_rounds=10, min_clients=5):
        self.sensitivity = sensitivity
        self.warmup_rounds = warmup_rounds
        self.min_clients = min_clients
        self.reference_models = []
        self.encryption_simulator = None
        self.adaptive_threshold = 3.0  # Initial threshold

    def _decrypt_model(self, encrypted_model):
        decrypted = {}
        for k, v in encrypted_model.items():
            decrypted_vec = decrypt_vector(self.encryption_simulator, v)
            decrypted[k] = torch.tensor(decrypted_vec).float().reshape(-1)
        return decrypted

    def _robust_zscore(self, values):
        median = np.median(values)
        mad = median_abs_deviation(values)
        return np.abs((values - median) / (mad + 1e-8))

    def analyze_models(self, client_models, encryption_simulator):
        self.encryption_simulator = encryption_simulator
        decrypted_models = [self._decrypt_model(m) for m in client_models]
        
        # Parameter vector analysis
        param_vectors = [torch.cat(list(m.values())) for m in decrypted_models]
        param_matrix = torch.stack(param_vectors).numpy()
        
        # Adaptive thresholding based on research findings [5][6]
        client_scores = []
        for vec in param_matrix:
            z_scores = self._robust_zscore(vec)
            client_scores.append(np.median(z_scores))
        
        # Dynamic threshold adjustment
        valid_indices = self._dynamic_thresholding(np.array(client_scores))
        return [client_models[i] for i in valid_indices]

    def _dynamic_thresholding(self, scores):
        """Adaptive threshold calculation based on summary statistics [5][6]"""
        if len(self.reference_models) < self.warmup_rounds:
            return np.arange(len(scores))  # Return all during warmup
        
        # Calculate reference statistics
        q75 = np.percentile(scores, 75)
        q25 = np.percentile(scores, 25)
        iqr = q75 - q25
        upper_bound = q75 + self.sensitivity * iqr
        
        valid = np.where(scores <= upper_bound)[0]
        
        # Ensure minimum client participation
        if len(valid) < self.min_clients:
            valid = np.argsort(scores)[:self.min_clients]
            
        return valid

    def update_reference(self, model_state):
        if len(self.reference_models) >= self.warmup_rounds:
            self.reference_models.pop(0)
        self.reference_models.append(model_state)
