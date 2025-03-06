from data_preprocessing import load_and_preprocess_data, split_data_among_clients
from model import CirrhosisPredictor
from federated_learning import federated_learning_with_early_stopping
from evaluation import evaluate_model, calculate_metrics, print_evaluation_results
from performance import PerformanceMonitor
import numpy as np

def main():
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    # Load and preprocess data
    file_path = '/Users/frederickayensu/jupyter-1.0.0/Federated_Learning/liver_cirrhosis.csv'
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data(file_path)
    
    # Split data among clients without attack simulation
    num_clients = 20
    client_data = split_data_among_clients(X_train_tensor, y_train_tensor, num_clients)
    
    # Initialize global model
    input_dim = X_train_tensor.shape[1]
    global_model = CirrhosisPredictor(input_dim)
    
    # Run federated learning with integrated monitoring
    trained_model = federated_learning_with_early_stopping(
        global_model,
        client_data,
        X_test_tensor,
        y_test_tensor,
        monitor=monitor,
        enable_defense=True,
        simulate_attacks=True
    )
    
    # Final evaluation
    final_accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)
    print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
    
    # Comprehensive metrics report
    final_metrics = calculate_metrics(trained_model, X_test_tensor, y_test_tensor)
    print_evaluation_results(final_metrics)
    
    # Generate defense framework performance report
    print("\nDefense Framework Performance Analysis:")
    monitor.print_detailed_report()

if __name__ == "__main__":
    main()
