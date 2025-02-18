from data_preprocessing import load_and_preprocess_data, split_data_among_clients
from model import CirrhosisPredictor
from federated_learning import federated_learning_with_early_stopping
from evaluation import evaluate_model, calculate_metrics, print_evaluation_results

def main():
    file_path = '/Users/frederickayensu/jupyter-1.0.0/Federated_Learning/liver_cirrhosis.csv'
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = load_and_preprocess_data(file_path)

    num_clients = 20
    client_data = split_data_among_clients(X_train_tensor, y_train_tensor, num_clients)

    input_dim = X_train_tensor.shape[1]
    global_model = CirrhosisPredictor(input_dim)

    trained_model = federated_learning_with_early_stopping(global_model, client_data, X_test_tensor, y_test_tensor)

    final_accuracy = evaluate_model(trained_model, X_test_tensor, y_test_tensor)
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    
    final_metrics = calculate_metrics(trained_model, X_test_tensor, y_test_tensor)
    print_evaluation_results(final_metrics)

if __name__ == "__main__":
    main()
