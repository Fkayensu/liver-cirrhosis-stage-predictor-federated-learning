# from data_preprocessing import load_and_preprocess_data, split_data_among_clients
# from model import CirrhosisPredictor
# from federated_learning import federated_learning_with_early_stopping
# from evaluation import evaluate_model, calculate_metrics, print_evaluation_results
# from performance import PerformanceMonitor
# import numpy as np
# import matplotlib.pyplot as plt

# def main():
#     # Initialize performance monitor
#     monitor = PerformanceMonitor()
    
#     # Load and preprocess data
#     file_path = '/Users/frederickayensu/jupyter-1.0.0/Federated_Learning/liver_cirrhosis.csv'
#     X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data(file_path)
    
#     # Split data among clients without attack simulation
#     num_clients = 20
#     client_data = split_data_among_clients(X_train_tensor, y_train_tensor, num_clients)
    
#     # Initialize global model
#     input_dim = X_train_tensor.shape[1]
#     global_model = CirrhosisPredictor(input_dim)
    
#     # Run federated learning with integrated monitoring
#     trained_model = federated_learning_with_early_stopping(
#         global_model,
#         client_data,
#         X_test_tensor,
#         y_test_tensor,
#         monitor=monitor,
#         enable_defense=True
#     )
    
#     # Final evaluation
#     final_accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)
#     print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
    
#     # Comprehensive metrics report
#     final_metrics = calculate_metrics(trained_model, X_test_tensor, y_test_tensor)
#     print_evaluation_results(final_metrics)
    
#     # Generate defense framework performance report
#     print("\nDefense Framework Performance Analysis:")
#     monitor.print_detailed_report()

# if __name__ == "__main__":
#     main()

from data_preprocessing import load_and_preprocess_data, split_data_among_clients
from model import CirrhosisPredictor
from federated_learning import federated_learning_with_early_stopping
from evaluation import evaluate_model, calculate_metrics, print_evaluation_results
from performance import PerformanceMonitor
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # List to store accuracies for each run
    all_round_accuracies = []
    
    # Run the program three times
    for run in range(3):
        print(f"\n--- Test Run {run + 1} ---")
        
        # Initialize performance monitor
        monitor = PerformanceMonitor()
        
        # Load and preprocess data
        file_path = '/Users/frederickayensu/jupyter-1.0.0/Federated_Learning/liver_cirrhosis.csv'
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data(file_path)
        
        # Split data among clients
        num_clients = 20
        client_data = split_data_among_clients(X_train_tensor, y_train_tensor, num_clients)
        
        # Initialize global model
        input_dim = X_train_tensor.shape[1]
        global_model = CirrhosisPredictor(input_dim)
        
        # Run federated learning and get both the trained model and round accuracies
        trained_model, round_accuracies = federated_learning_with_early_stopping(
            global_model, client_data, X_test_tensor, y_test_tensor, monitor=monitor, enable_defense=True
        )
        
        # Store accuracies for plotting
        all_round_accuracies.append(round_accuracies)
        
        # Final evaluation
        final_accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)
        print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
        
        # Comprehensive metrics report
        final_metrics = calculate_metrics(trained_model, X_test_tensor, y_test_tensor)
        print_evaluation_results(final_metrics)
        
        # Defense framework performance report
        print("\nDefense Framework Performance Analysis:")
        monitor.print_detailed_report()
    
    # Plot accuracy trends for all three runs
    plt.figure(figsize=(10, 6))
    for i, accuracies in enumerate(all_round_accuracies):
        plt.plot(range(1, len(accuracies) + 1), accuracies, label=f'Test Run {i + 1}', marker='o')
    plt.xlabel('Federation Round')
    plt.ylabel('Test Accuracy')
    plt.title('Accuracy Across Federation Rounds for 3 Test Runs')
    plt.legend()
    plt.grid(True)
    plt.savefig('defense_fl_model_accuracy.png')
    plt.show()