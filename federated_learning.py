import numpy as np
import tensorflow as tf
from typing import List, Dict, Tuple
from model import CNNModel
from data_preprocessing import DataPreprocessor
from evaluate_model import evaluate_model
import logging
import os
import time
import pickle
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

def save_weights(weights, path):
    try:
        os.makedirs('pickle_files', exist_ok=True)
        with open(os.path.join('pickle_files', path), 'wb') as f:
            pickle.dump(weights, f)
        return True
    except Exception as e:
        print(f"Error saving weights: {str(e)}")
        return False

def load_weights(path):
    try:
        with open(os.path.join('pickle_files', path), 'rb') as f:
            weights = pickle.load(f)
        return weights
    except Exception as e:
        print(f"Error loading weights: {str(e)}")
        return None

class FederatedServer:
    def __init__(self, num_classes):
        self.global_model = CNNModel(num_classes)
        self.num_classes = num_classes

    def aggregate_weights(self, client_weights):
        """Aggregate client weights using FedAvg"""
        avg_weights = [np.mean(weights, axis=0) for weights in zip(*client_weights)]
        self.global_model.set_weights(avg_weights)
        return avg_weights

    def get_weights(self):
        """Get current global model weights"""
        return self.global_model.get_weights()

class FederatedClient:
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # Initialize SMOTE with conservative parameters
        self.smote = SMOTE(
            random_state=42,
            k_neighbors=3,  # Reduced neighbors for more conservative sampling
            sampling_strategy='auto'
        )

    def set_weights(self, weights):
        """Update client model with new weights"""
        self.model.set_weights(weights)

    def fit(self, global_weights, config):
        """Train client model with SMOTE and class weights"""
        self.model.set_weights(global_weights)
        
        # Print class distribution before SMOTE
        print("\nClass distribution before SMOTE:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} samples ({count/len(self.y_train)*100:.2f}%)")
        
        # Calculate target number of samples for each class
        class_counts = np.bincount(self.y_train)
        max_count = np.max(class_counts)
        
        # Create conservative sampling strategy
        sampling_strategy = {}
        for class_idx in range(len(class_counts)):
            if class_counts[class_idx] > 0:  # Only include classes that exist
                # For minority classes (less than 20% of max), increase up to 1.3x
                if class_counts[class_idx] < max_count * 0.2:
                    target_count = min(int(class_counts[class_idx] * 1.3), int(max_count * 0.2))
                else:
                    # For other classes, keep original count
                    target_count = class_counts[class_idx]
                sampling_strategy[class_idx] = target_count
        
        print("\nSMOTE sampling strategy:")
        for class_idx, target in sampling_strategy.items():
            print(f"Class {class_idx}: {class_counts[class_idx]} -> {target} samples")
        
        # Update SMOTE sampling strategy
        self.smote.sampling_strategy = sampling_strategy
        
        # Reshape data for SMOTE
        X_reshaped = self.X_train.reshape(self.X_train.shape[0], -1)
        
        # Apply SMOTE
        try:
            X_resampled, y_resampled = self.smote.fit_resample(X_reshaped, self.y_train)
            X_resampled = X_resampled.reshape(-1, 32, 32, 1)
            
            # Print class distribution after SMOTE
            print("\nClass distribution after SMOTE:")
            unique, counts = np.unique(y_resampled, return_counts=True)
            for label, count in zip(unique, counts):
                print(f"Class {label}: {count} samples ({count/len(y_resampled)*100:.2f}%)")
            
            # Train the model
            history = self.model.fit(
                X_resampled, y_resampled,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                class_weight=config["class_weight"],
                validation_split=config["validation_split"]
            )
            
            # Print training metrics
            print("\nTraining metrics:")
            print(f"Final Loss: {history.history['loss'][-1]:.4f}")
            print(f"Final Accuracy: {history.history['accuracy'][-1]:.4f}")
            print(f"Validation Loss: {history.history['val_loss'][-1]:.4f}")
            print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
            
            weights = self.model.get_weights()
            num_examples = len(self.y_train)
            metrics = {
                'loss': history.history['loss'][-1],
                'accuracy': history.history['accuracy'][-1],
                'val_loss': history.history['val_loss'][-1],
                'val_accuracy': history.history['val_accuracy'][-1]
            }
            
            return weights, num_examples, metrics
            
        except Exception as e:
            print(f"Error in SMOTE: {str(e)}")
            # Fallback to original data if SMOTE fails
            print("Using original data without SMOTE")
            history = self.model.fit(
                self.X_train, self.y_train,
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                class_weight=config["class_weight"],
                validation_split=config["validation_split"]
            )
            
            weights = self.model.get_weights()
            num_examples = len(self.y_train)
            metrics = {
                'loss': history.history['loss'][-1],
                'accuracy': history.history['accuracy'][-1],
                'val_loss': history.history['val_loss'][-1],
                'val_accuracy': history.history['val_accuracy'][-1]
            }
            
            return weights, num_examples, metrics

    def get_weights(self):
        """Get current client model weights"""
        return self.model.get_weights()

def calculate_class_weights(y_train):
    """Calculate class weights for balanced training"""
    # Get unique classes present in the data
    unique_classes = np.unique(y_train)
    
    # Calculate weights for existing classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=y_train
    )
    
    # Create dictionary with weights for all classes
    weight_dict = {i: 1.0 for i in range(int(np.max(y_train)) + 1)}  # Initialize all weights to 1.0
    for class_idx, weight in zip(unique_classes, class_weights):
        weight_dict[class_idx] = weight
    
    # Apply additional boosting for minority classes with a maximum limit
    class_counts = np.bincount(y_train)
    max_count = np.max(class_counts)
    for class_idx in unique_classes:
        if class_counts[class_idx] < max_count * 0.1:  # If class has less than 10% of max count
            # Limit the maximum oversampling ratio to 3x
            current_weight = weight_dict[class_idx]
            max_allowed_weight = 3.0
            weight_dict[class_idx] = min(current_weight * 2.0, max_allowed_weight)
    
    print("\nClass weights:")
    for class_idx, weight in weight_dict.items():
        print(f"Class {class_idx}: {weight:.4f}")
    
    return weight_dict

def train_federated_model(num_clients=3, num_rounds=10, epochs_per_client=3):
    """Main federated learning training loop"""
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.load_data(
        "UNSW_NB15_training-set.csv",
        "UNSW_NB15_testing-set.csv"
    )
    
    # Initialize server
    server = FederatedServer(num_classes=len(preprocessor.attack_mapping))
    
    # Initialize client data arrays
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]
    
    # Split data by class to ensure balanced distribution
    for class_idx in range(len(preprocessor.attack_mapping)):
        class_mask = y_train == class_idx
        X_class = X_train[class_mask]
        y_class = y_train[class_mask]
        
        # Shuffle class data
        indices = np.random.permutation(len(X_class))
        X_class = X_class[indices]
        y_class = y_class[indices]
        
        # Split class data among clients
        samples_per_client = len(X_class) // num_clients
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            
            if len(client_data[i]) == 0:  # First time adding data to this client
                client_data[i] = X_class[start_idx:end_idx]
                client_labels[i] = y_class[start_idx:end_idx]
            else:
                client_data[i] = np.concatenate([client_data[i], X_class[start_idx:end_idx]])
                client_labels[i] = np.concatenate([client_labels[i], y_class[start_idx:end_idx]])
    
    # Convert lists to numpy arrays and shuffle
    for i in range(num_clients):
        client_data[i] = np.array(client_data[i])
        client_labels[i] = np.array(client_labels[i])
        indices = np.random.permutation(len(client_data[i]))
        client_data[i] = client_data[i][indices]
        client_labels[i] = client_labels[i][indices]
    
    # Train for multiple rounds
    best_accuracy = 0.0
    best_weights = None
    
    for round_num in range(num_rounds):
        print(f"\nFederated Round {round_num + 1}/{num_rounds}")
        
        # Initialize list to store client weights
        client_weights = []
        
        # Train each client
        for i in range(num_clients):
            print(f"\nTraining Client {i + 1}/{num_clients}")
            
            # Get client's portion of data
            X_client = client_data[i]
            y_client = client_labels[i]
            
            # Initialize client and set global weights
            client = FederatedClient(
                model=CNNModel(len(preprocessor.attack_mapping)),
                X_train=X_client,
                y_train=y_client,
                X_test=X_test,
                y_test=y_test
            )
            client.set_weights(server.get_weights())
            
            # Calculate class weights for this client
            client_class_weights = calculate_class_weights(y_client)
            
            # Train client with optimized parameters
            config = {
                "epochs": epochs_per_client,
                "batch_size": 128,  # Reduced batch size for faster training
                "class_weight": client_class_weights,
                "validation_split": 0.2
            }
            
            weights, num_examples, metrics = client.fit(server.get_weights(), config)
            client_weights.append(weights)
            
            # Print training metrics
            print(f"\nClient {i + 1} Training Metrics:")
            print(f"Final Loss: {metrics['loss']:.4f}")
            print(f"Final Accuracy: {metrics['accuracy']:.4f}")
            print(f"Validation Loss: {metrics['val_loss']:.4f}")
            print(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")
        
        # Aggregate weights
        server.aggregate_weights(client_weights)
        
        # Evaluate global model
        loss, accuracy = server.global_model.evaluate(X_test, y_test)
        print(f"\nGlobal Model - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = server.get_weights()
            print(f"\nNew best accuracy: {best_accuracy:.4f}")
            save_weights(best_weights, 'best_global_model_weights.pkl')
    
    # Load best weights
    server.global_model.set_weights(best_weights)
    
    return server, X_test, y_test

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Train federated model
        server, X_test, y_test = train_federated_model()
        
        print("\nTraining complete!")
        print("\nFinal model performance:")
        accuracy, report = evaluate_model(server.global_model.model, X_test, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nFinal Classification Report:")
        print(report)
        
        # Save final model
        model_path = "models/federated_model.h5"
        server.global_model.model.save(model_path)
        print(f"\nSaved final model to '{model_path}'")
        
        # Save evaluation report
        report_path = "models/evaluation_report.txt"
        with open(report_path, 'w') as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"Saved evaluation report to '{report_path}'")
        
        print("\nAll post-training steps completed successfully!")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        print("Current working directory:", os.getcwd())
        print("Available files:", os.listdir('.'))
        raise

if __name__ == "__main__":
    main() 