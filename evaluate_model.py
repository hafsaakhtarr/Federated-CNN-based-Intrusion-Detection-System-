import numpy as np
import tensorflow as tf
from model import CNNModel
from data_preprocessing import DataPreprocessor
import logging
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def train_centralized_model(X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray,
                          epochs: int = 50, batch_size: int = 32) -> dict:
    """Train a centralized model for comparison"""
    model = CNNModel(input_shape=(32, 32, 1), num_classes=len(np.unique(y_train)))
    
    # Train the model
    history = model.model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return {
        'history': history.history,
        'model': model
    }

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on test data"""
    # Make predictions
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate accuracy
    accuracy = float(np.mean(y_pred_classes == y_test))
    
    # Generate classification report
    report = classification_report(
        y_test, 
        y_pred_classes,
        target_names=[str(i) for i in range(len(np.unique(y_test)))],
        zero_division=0
    )
    
    return accuracy, report

def plot_comparison(federated_history: dict, centralized_history: dict) -> None:
    """Plot comparison between federated and centralized training"""
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy comparison
    plt.subplot(1, 2, 1)
    plt.plot(federated_history['round'], federated_history['accuracy'], 'b-', label='Federated')
    plt.plot(range(1, len(centralized_history['accuracy']) + 1), 
             centralized_history['accuracy'], 'r-', label='Centralized')
    plt.xlabel('Epoch/Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot loss comparison
    plt.subplot(1, 2, 2)
    plt.plot(federated_history['round'], federated_history['loss'], 'b-', label='Federated')
    plt.plot(range(1, len(centralized_history['loss']) + 1), 
             centralized_history['loss'], 'r-', label='Centralized')
    plt.xlabel('Epoch/Round')
    plt.ylabel('Loss')
    plt.title('Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def plot_confusion_matrix(cm: np.ndarray, title: str) -> None:
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    train_data, test_data = preprocessor.load_data(
        train_path='UNSW_NB15_training-set.csv',
        test_path='UNSW_NB15_testing-set.csv'
    )
    
    # Preprocess features
    X_train, X_test, y_train, y_test = preprocessor.preprocess_features(train_data, test_data)
    
    # Load federated learning history
    try:
        federated_history = np.load('federated_learning_history.npy', allow_pickle=True).item()
        logging.info("Loaded federated learning history")
    except FileNotFoundError:
        logging.error("No federated learning history found. Please run federated_learning.py first.")
        return
    
    # Train centralized model
    logging.info("Training centralized model for comparison...")
    centralized_results = train_centralized_model(X_train, y_train, X_test, y_test)
    
    # Plot comparison
    plot_comparison(federated_history, centralized_results['history'])
    
    # Evaluate both models
    logging.info("Evaluating federated model...")
    federated_metrics = evaluate_model(CNNModel(), X_test, y_test)
    
    logging.info("Evaluating centralized model...")
    centralized_metrics = evaluate_model(centralized_results['model'], X_test, y_test)
    
    # Plot confusion matrices
    plot_confusion_matrix(federated_metrics['confusion_matrix'], 'Federated Model Confusion Matrix')
    plot_confusion_matrix(centralized_metrics['confusion_matrix'], 'Centralized Model Confusion Matrix')
    
    # Print comparison
    logging.info("\nFederated Model Performance:")
    logging.info(federated_metrics[1])
    
    logging.info("\nCentralized Model Performance:")
    logging.info(centralized_metrics[1])

if __name__ == "__main__":
    main() 