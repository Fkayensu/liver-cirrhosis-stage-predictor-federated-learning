import time
import numpy as np
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'security': defaultdict(list),
            'performance': defaultdict(list),
            'scalability': defaultdict(list)
        }
        self.timers = {}
        self.attack_ground_truth = {}
        self.client_counts = []
        self.data_sizes = []
        self.round_timestamps = []  # Track round completion order
        self.attack_counters = {
            'data_poisoning': 0,
            'model_poisoning': 0,
            'backdoor': 0,
            'mitm': 0
        }

    def record_attack(self, attack_type):
        self.attack_counters[attack_type] += 1

    def start_timer(self, name):
        self.timers[name] = time.time()

    def stop_timer(self, name):
        if name in self.timers:
            elapsed = time.time() - self.timers.pop(name)
            self.metrics['performance'][f'{name}_latency'].append(elapsed)

    def record_scalability_metrics(self, num_clients, data_size):
        self.client_counts.append(num_clients)
        self.data_sizes.append(data_size)
        self.round_timestamps.append(len(self.client_counts))  # Sync with round numbers

    def calculate_scalability_metrics(self):
        # Get matching time metrics for recorded client counts
        time_metrics = []
        client_growth = []
        
        for i, ts in enumerate(self.round_timestamps):
            if ts-1 < len(self.metrics['performance'].get('round_latency', [])):
                time_metrics.append(self.metrics['performance']['round_latency'][ts-1])
                client_growth.append(self.client_counts[i])
        
        # Fallback to last N entries if lengths mismatch
        min_length = min(len(client_growth), len(time_metrics))
        client_growth = client_growth[:min_length]
        time_metrics = time_metrics[:min_length]
        
        if len(client_growth) < 2 or len(time_metrics) < 2:
            return {
                'client_efficiency': 0,
                'data_efficiency': 0
            }
        
        try:
            client_eff = np.polyfit(client_growth, time_metrics, 1)[0]
            data_eff = np.polyfit(np.log(self.data_sizes[:min_length]), 
                                time_metrics[:min_length], 1)[0]
        except Exception as e:
            print(f"Scalability metric error: {str(e)}")
            client_eff = data_eff = 0
            
        return {
            'client_efficiency': client_eff,
            'data_efficiency': data_eff
        }

    def safe_mean(self, arr):
        if len(arr) == 0:
            return 0  # or np.nan
        return np.mean(arr)

    def safe_divide(self, numerator, denominator):
        if denominator == 0:
            return 0  # or np.nan, depending on your preference
        return numerator / denominator
    
    def calculate_performance_metrics(self):
        return {
            'avg_aggregation_latency': self.safe_mean(self.metrics['performance'].get('aggregation_latency', [])),
            'avg_validation_latency': self.safe_mean(self.metrics['performance'].get('validation_latency', [])),
            'avg_round_time': self.safe_mean(self.metrics['performance'].get('round_latency', []))
        }
    
    def generate_report(self):
        return {
            'security': self.calculate_security_metrics(),
            'performance': self.calculate_performance_metrics(),
            'scalability': self.calculate_scalability_metrics()
        }

    def print_detailed_report(self):
        report = self.generate_report()

        print("\n=== Attack Counters ===")
        for attack_type, count in self.attack_counters.items():
            print(f"{attack_type.capitalize()} Attacks: {count}")
        
        print("\n=== Security Metrics ===")
        print(f"Attack Detection Rate: {report['security']['detection_rate']:.2%}")
        print(f"False Positive Rate: {report['security']['false_positive_rate']:.2%}")
        print(f"Precision: {report['security']['precision']:.2%}")
        print(f"Recall: {report['security']['recall']:.2%}")
        
        print("\n=== Performance Metrics ===")
        print(f"Aggregation Latency: {report['performance']['avg_aggregation_latency']:.4f}s")
        print(f"Validation Latency: {report['performance']['avg_validation_latency']:.4f}s")
        print(f"Average Round Time: {report['performance']['avg_round_time']:.4f}s")
        
        print("\n=== Scalability Metrics ===")
        print(f"Client Efficiency Slope: {report['scalability']['client_efficiency']:.4f}")
        print(f"Data Efficiency Slope: {report['scalability']['data_efficiency']:.4f}")

    def record_security_event(self, client_id, is_attack, detected):
        self.metrics['security']['true_positives'].append(1 if is_attack and detected else 0)
        self.metrics['security']['false_positives'].append(1 if not is_attack and detected else 0)
        self.metrics['security']['true_negatives'].append(1 if not is_attack and not detected else 0)
        self.metrics['security']['false_negatives'].append(1 if is_attack and not detected else 0)

    def calculate_security_metrics(self):
        tp = sum(self.metrics['security']['true_positives'])
        fp = sum(self.metrics['security']['false_positives'])
        tn = sum(self.metrics['security']['true_negatives'])
        fn = sum(self.metrics['security']['false_negatives'])
        
        print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")  # Debug output
        
        detection_rate = self.safe_divide(tp, tp + fn)
        false_positive_rate = self.safe_divide(fp, fp + tn)
        precision = self.safe_divide(tp, tp + fp)
        recall = detection_rate
        
        return {
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'precision': precision,
            'recall': recall
        }