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

# from data_preprocessing import load_and_preprocess_data, split_data_among_clients
# from model import CirrhosisPredictor
# from federated_learning import federated_learning_with_early_stopping
# from evaluation import evaluate_model, calculate_metrics, print_evaluation_results
# from performance import PerformanceMonitor
# import numpy as np
# import matplotlib.pyplot as plt

# if __name__ == "__main__":
#     # List to store accuracies for each run
#     all_round_accuracies = []
    
#     # Run the program three times
#     for run in range(3):
#         print(f"\n--- Test Run {run + 1} ---")
        
#         # Initialize performance monitor
#         monitor = PerformanceMonitor()
        
#         # Load and preprocess data
#         file_path = '/Users/frederickayensu/jupyter-1.0.0/Federated_Learning/liver_cirrhosis.csv'
#         X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data(file_path)
        
#         # Split data among clients
#         num_clients = 20
#         client_data = split_data_among_clients(X_train_tensor, y_train_tensor, num_clients)
        
#         # Initialize global model
#         input_dim = X_train_tensor.shape[1]
#         global_model = CirrhosisPredictor(input_dim)
        
#         # Run federated learning and get both the trained model and round accuracies
#         trained_model, round_accuracies = federated_learning_with_early_stopping(
#             global_model, client_data, X_test_tensor, y_test_tensor, monitor=monitor, enable_defense=True
#         )
        
#         # Store accuracies for plotting
#         all_round_accuracies.append(round_accuracies)
        
#         # Final evaluation
#         final_accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)
#         print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
        
#         # Comprehensive metrics report
#         final_metrics = calculate_metrics(trained_model, X_test_tensor, y_test_tensor)
#         print_evaluation_results(final_metrics)
        
#         # Defense framework performance report
#         print("\nDefense Framework Performance Analysis:")
#         monitor.print_detailed_report()
    
#     # Plot accuracy trends for all three runs
#     plt.figure(figsize=(10, 6))
#     for i, accuracies in enumerate(all_round_accuracies):
#         plt.plot(range(1, len(accuracies) + 1), accuracies, label=f'Test Run {i + 1}', marker='o')
#     plt.xlabel('Federation Round')
#     plt.ylabel('Test Accuracy')
#     plt.title('Accuracy Across Federation Rounds for 3 Test Runs')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('defense_fl_model_accuracy.png')
#     plt.show()


from data_preprocessing import load_and_preprocess_data, split_data_among_clients
from model import CirrhosisPredictor
from federated_learning import federated_learning_with_early_stopping
from evaluation import evaluate_model, calculate_metrics, print_evaluation_results
from performance import PerformanceMonitor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

if __name__ == "__main__":
    all_round_accuracies = []
    # *** Add this line to store performance metrics across runs ***
    run_metrics = []

    for run in range(1):
        print(f"\n--- Test Run {run + 1} ---")
        
        monitor = PerformanceMonitor()
        file_path = '/Users/frederickayensu/jupyter-1.0.0/Federated_Learning/liver_cirrhosis.csv'  # Update with your file path
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data(file_path)
        num_clients = 20
        client_data = split_data_among_clients(X_train_tensor, y_train_tensor, num_clients)
        input_dim = X_train_tensor.shape[1]
        global_model = CirrhosisPredictor(input_dim)

        # Run federated learning
        trained_model, round_accuracies, security_true_labels, security_pred_labels, security_scores = federated_learning_with_early_stopping(
            global_model, client_data, X_test_tensor, y_test_tensor, monitor=monitor, enable_defense=True
        )

        all_round_accuracies.append(round_accuracies)

        # Evaluate final model
        final_accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)
        print(f"\nFinal Test Accuracy: {final_accuracy:.4f}")
        final_metrics = calculate_metrics(trained_model, X_test_tensor, y_test_tensor)
        print_evaluation_results(final_metrics)
        
        # Defense framework performance report
        print("\nDefense Framework Performance Analysis:")
        monitor.print_detailed_report()

        # *** Add this line to collect performance metrics after each run ***
        report = monitor.generate_report()
        run_metrics.append(report['performance'])

        # --- Model Performance Metrics ---
        y_true = y_test_tensor.numpy()
        y_scores = trained_model(X_test_tensor).detach().numpy()
        y_pred = np.argmax(y_scores, axis=1)

        # Model Confusion Matrix
        cm_model = confusion_matrix(y_true, y_pred)
        print("\nModel Performance Confusion Matrix:")
        plt.figure(figsize=(8, 6))
        classes = ['1', '2', '3']
        sns.heatmap(cm_model, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label', fontsize=16)
        plt.ylabel('True Label', fontsize=16)
        plt.title(f'Confusion Matrix - Test Run {run + 1}', fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.savefig(f'confusion_matrix_defense_fl_run_{run + 1}.png')
        plt.show()

        # Model ROC Curves (one per class)
        print("\nPlotting ROC Curves for Model Performance:")
        plt.figure(figsize=(10, 6))  # Single figure size
        for i in range(3):  # Assuming 3 classes
            fpr, tpr, _ = roc_curve(y_true == i, y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title(f'ROC Curves - All Classes (Run {run + 1})', fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'ROC_curve_defense_fl_run_{run + 1}.png')
        plt.show()

        # --- Security Metrics ---
        security_true = np.array(security_true_labels)
        security_pred = np.array(security_pred_labels)
        security_scores = np.array(security_scores)

        # Security Confusion Matrix
        cm_security = confusion_matrix(security_true, security_pred)
        print("\nSecurity Metrics Confusion Matrix:")
        plt.figure(figsize=(8, 6))
        classes = ['No', 'Yes']
        sns.heatmap(cm_security, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted Label', fontsize=16)
        plt.ylabel('True Label', fontsize=16)
        plt.title(f'Confusion Matrix - Test Run {run + 1}', fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.savefig(f'cm_sec_{run + 1}.png')
        plt.show()

        # Security ROC Curve
        print("\nPlotting ROC Curve for Security Metrics:")
        fpr, tpr, _ = roc_curve(security_true, security_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'Security Detection (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title('ROC Curve - Security Detection', fontsize=16)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.legend()
        plt.savefig(f'ROC_sec_{run + 1}.png')
        plt.show()

    # Plot accuracy trends across runs
    plt.figure(figsize=(10, 6))
    for i, accuracies in enumerate(all_round_accuracies):
        plt.plot(range(1, len(accuracies) + 1), accuracies, label=f'Test Run {i + 1}', marker='o')
    plt.xlabel('Federation Round', fontsize=16)
    plt.ylabel('Test Accuracy', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title('Accuracy Across Federation Rounds for 3 Test Runs', fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'acc_trend_sec_{run + 1}.png')
    plt.show()

    # *** Add this code after the loop to generate the table and bar chart ***
    # Print latency table
    print("\n=== Latency Metrics Across Runs ===")
    print(f"{'Run':<5} | {'Aggregation Latency':<20} | {'Validation Latency':<20} | {'Average Round Time':<20}")
    print("-" * 70)
    for i, metrics in enumerate(run_metrics, 1):
        agg_lat = metrics.get('avg_aggregation_latency', 0)
        val_lat = metrics.get('avg_validation_latency', 0)
        round_lat = metrics.get('avg_round_latency', 0)
        print(f"{i:<5} | {agg_lat:<20.4f} | {val_lat:<20.4f} | {round_lat:<20.4f}")

    # Calculate averages across runs
    avg_agg = np.mean([m.get('avg_aggregation_latency', 0) for m in run_metrics])
    avg_val = np.mean([m.get('avg_validation_latency', 0) for m in run_metrics])
    avg_round = np.mean([m.get('avg_round_latency', 0) for m in run_metrics])
    print(f"{'Avg':<5} | {avg_agg:<20.4f} | {avg_val:<20.4f} | {avg_round:<20.4f}")

    # Plot bar chart
    metrics_names = ['Aggregation Latency', 'Validation Latency', 'Average Round Time']
    runs = ['Run 1', 'Run 2', 'Run 3']
    data = {
        'Aggregation Latency': [max(m.get('avg_aggregation_latency', 0), 1e-6) for m in run_metrics],
        'Validation Latency': [max(m.get('avg_validation_latency', 0), 1e-6) for m in run_metrics],
        'Average Round Time': [max(m.get('avg_round_latency', 0), 1e-6) for m in run_metrics]
    }

    # # Create figure
    # plt.figure(figsize=(12, 6))
    # x = np.arange(len(runs))
    # width = 0.25
    # plt.bar(x - width, [data['Aggregation Latency'][i] for i in range(len(runs))], width, label='Aggregation Latency', color='skyblue')
    # plt.bar(x, [data['Validation Latency'][i] for i in range(len(runs))], width, label='Validation Latency', color='lightgreen')
    # plt.bar(x + width, [data['Average Round Time'][i] for i in range(len(runs))], width, label='Average Round Time', color='salmon')
    # plt.yscale('log')
    # plt.xlabel('Test Run', fontsize=16)
    # plt.ylabel('Time (seconds, log scale)', fontsize=16)
    # plt.title('Performance Metrics Across Test Runs', fontsize=16)
    # plt.xticks(x, runs, fontsize=13)
    # plt.yticks(fontsize=13)
    # plt.legend()
    # plt.grid(True, alpha=0.3, which="both")
    # plt.tight_layout()
    # plt.savefig('performance_metrics.png')
    # plt.show()

    # print("\n=== Scalability Experiment ===")
    # client_counts = list(range(2, 21, 2))
    # scalability_data = []

    # for num_clients in client_counts:
    #     print(f"\nRunning with {num_clients} clients")
    #     monitor = PerformanceMonitor()
        
    #     # Load and preprocess data
    #     file_path = '/Users/frederickayensu/jupyter-1.0.0/Federated_Learning/liver_cirrhosis.csv'
    #     X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data(file_path)
        
    #     # Split data among clients
    #     client_data = split_data_among_clients(X_train_tensor, y_train_tensor, num_clients)
        
    #     # Initialize model
    #     input_dim = X_train_tensor.shape[1]
    #     global_model = CirrhosisPredictor(input_dim)
        
    #     # Run federated learning for 5 rounds
    #     trained_model, round_accuracies, security_true_labels, security_pred_labels, security_scores = federated_learning_with_early_stopping(
    #         global_model,
    #         client_data,
    #         X_test_tensor,
    #         y_test_tensor,
    #         monitor=monitor,
    #         enable_defense=True,
    #         max_rounds=5,
    #         patience=10
    #     )
        
    #     # Extract performance metrics
    #     report = monitor.generate_report()
    #     performance = report['performance']
    #     agg_lat = performance.get('avg_aggregation_latency', 0)
    #     val_lat = performance.get('avg_validation_latency', 0)
    #     round_lat = performance.get('avg_round_latency', 0)
        
    #     # Store results
    #     scalability_data.append({
    #         'num_clients': num_clients,
    #         'agg_latency': agg_lat,
    #         'val_latency': val_lat,
    #         'round_latency': round_lat
    #     })

    # # Create and print the table
    # print("\n=== Scalability Metrics Table ===")
    # print(f"{'Clients':<10} | {'Agg Latency':<15} | {'Val Latency':<15} | {'Round Time':<15}")
    # print("-" * 60)
    # for data in scalability_data:
    #     print(f"{data['num_clients']:<10} | {data['agg_latency']:<15.4f} | {data['val_latency']:<15.4f} | {data['round_latency']:<15.4f}")

    # # Create the line graph
    # plt.figure(figsize=(10, 6))
    # plt.plot(
    #     [d['num_clients'] for d in scalability_data],
    #     [d['agg_latency'] for d in scalability_data],
    #     label='Aggregation Latency',
    #     marker='o'
    # )
    # plt.plot(
    #     [d['num_clients'] for d in scalability_data],
    #     [d['val_latency'] for d in scalability_data],
    #     label='Validation Latency',
    #     marker='o'
    # )
    # plt.plot(
    #     [d['num_clients'] for d in scalability_data],
    #     [d['round_latency'] for d in scalability_data],
    #     label='Average Round Time',
    #     marker='o'
    # )
    # plt.xlabel('Number of Clients')
    # plt.ylabel('Time (s)')
    # plt.title('Scalability Metrics vs Number of Clients')
    # plt.yscale('log')  # Logarithmic scale for y-axis
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('scalability_metrics.png')
    # plt.show()

    # --- Experiment 1: Accuracy vs Number of Clients ---
    # print("\n=== Experiment: Accuracy vs Number of Clients ===")
    # client_counts = list(range(2, 21, 2))  # [2, 4, 6, ..., 20]
    # accuracies_vs_clients = []

    # for num_clients in client_counts:
    #     print(f"Running with {num_clients} clients")
    #     monitor = PerformanceMonitor()
        
    #     # Load and preprocess data
    #     X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data(file_path)
        
    #     # Split data among clients
    #     client_data = split_data_among_clients(X_train_tensor, y_train_tensor, num_clients)
        
    #     # Initialize global model
    #     input_dim = X_train_tensor.shape[1]
    #     global_model = CirrhosisPredictor(input_dim)
        
    #     # Run federated learning for 5 rounds
    #     trained_model, round_accuracies, _, _, _ = federated_learning_with_early_stopping(
    #         global_model,
    #         client_data,
    #         X_test_tensor,
    #         y_test_tensor,
    #         monitor=monitor,
    #         enable_defense=True,
    #         max_rounds=5,
    #         patience=10  # Ensures all 5 rounds run without early stopping
    #     )
        
    #     # Evaluate final accuracy
    #     accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)
    #     accuracies_vs_clients.append(accuracy)

    # # Table for Accuracy vs Number of Clients
    # print("\nAccuracy vs Number of Clients")
    # print(f"{'Clients':<10} | {'Accuracy':<10}")
    # print("-" * 22)
    # for clients, acc in zip(client_counts, accuracies_vs_clients):
    #     print(f"{clients:<10} | {acc:.4f}")

    # # Line Graph for Accuracy vs Number of Clients
    # plt.figure(figsize=(8, 5))
    # plt.plot(client_counts, accuracies_vs_clients, marker='o', label='Accuracy')
    # plt.xlabel('Number of Clients', fontsize=16)
    # plt.ylabel('Accuracy', fontsize=16)
    # plt.ylim(0.5, 1)
    # plt.xticks(fontsize=13)
    # plt.yticks(fontsize=13)
    # plt.title('Accuracy vs Number of Clients', fontsize=16)
    # plt.grid(True)
    # plt.savefig('accuracy_vs_clients.png')
    # plt.show()

    # # --- Experiment 2: Accuracy vs Data Fraction ---
    # print("\n=== Experiment: Accuracy vs Data Fraction ===")
    # data_fractions = [i / 10 for i in range(1, 11)]  # [0.1, 0.2, ..., 1.0]
    # accuracies_vs_data = []
    # fixed_num_clients = 10  # Fixed number of clients for this experiment

    # for fraction in data_fractions:
    #     print(f"Running with data fraction {fraction:.1f}")
    #     monitor = PerformanceMonitor()
        
    #     # Load and preprocess data
    #     X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data(file_path)
        
    #     # Select a random subset of training data based on the fraction
    #     num_samples = int(len(X_train_tensor) * fraction)
    #     indices = np.random.choice(len(X_train_tensor), num_samples, replace=False)
    #     X_train_subset = X_train_tensor[indices]
    #     y_train_subset = y_train_tensor[indices]
        
    #     # Split the subset among fixed number of clients
    #     client_data = split_data_among_clients(X_train_subset, y_train_subset, fixed_num_clients)
        
    #     # Initialize global model
    #     input_dim = X_train_tensor.shape[1]
    #     global_model = CirrhosisPredictor(input_dim)
        
    #     # Run federated learning for 5 rounds
    #     trained_model, round_accuracies, _, _, _ = federated_learning_with_early_stopping(
    #         global_model,
    #         client_data,
    #         X_test_tensor,
    #         y_test_tensor,
    #         monitor=monitor,
    #         enable_defense=True,
    #         max_rounds=5,
    #         patience=10  # Ensures all 5 rounds run without early stopping
    #     )
        
    #     # Evaluate final accuracy
    #     accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)
    #     accuracies_vs_data.append(accuracy)

    # # Table for Accuracy vs Data Fraction
    # print("\nAccuracy vs Data Fraction")
    # print(f"{'Data Fraction':<15} | {'Accuracy':<10}")
    # print("-" * 28)
    # for fraction, acc in zip(data_fractions, accuracies_vs_data):
    #     print(f"{fraction:<15.1f} | {acc:.4f}")

    # # Line Graph for Accuracy vs Data Fraction
    # plt.figure(figsize=(8, 5))
    # plt.plot(data_fractions, accuracies_vs_data, marker='o', label='Accuracy')
    # plt.xlabel('Data Fraction', fontsize=13)
    # plt.ylabel('Accuracy', fontsize=13)
    # plt.xticks(fontsize=13)
    # plt.yticks(fontsize=13)
    # plt.title('Accuracy vs Data Fraction', fontsize=13)
    # plt.grid(True)
    # plt.savefig('accuracy_vs_data_fraction.png')
    # plt.show()