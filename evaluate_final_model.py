import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from data_preprocessing import DataPreprocessor
from model import CNNModel
import pickle
import os

def evaluate_final_model(model, X_test, y_test, attack_mapping):
    print("\nEvaluating Final Model Performance...")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracy = np.mean(y_pred_classes == y_test)
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    cm = confusion_matrix(y_test, y_pred_classes)
    
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    
    print("\nDetailed Classification Report:")
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[attack_mapping[i] for i in range(len(attack_mapping))],
                yticklabels=[attack_mapping[i] for i in range(len(attack_mapping))])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Attack Type')
    plt.ylabel('True Attack Type')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('final_model_confusion_matrix.png')
    plt.close()
    
    unique, counts = np.unique(y_test, return_counts=True)
    plt.figure(figsize=(12, 6))
    plt.bar([attack_mapping[i] for i in unique], counts)
    plt.title('Test Data Class Distribution')
    plt.xlabel('Attack Type')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('test_data_distribution.png')
    plt.close()
    
    return accuracy, report, cm

def main():
    try:
        with open(os.path.join('pickle_files', 'global_model_weights.pkl'), 'rb') as f:
            weights = pickle.load(f)
        
        preprocessor = DataPreprocessor()
        _, _, _, _ = preprocessor.load_data(
            'UNSW_NB15_training-set.csv',
            'UNSW_NB15_testing-set.csv'
        )
        
        input_shape = (32, 32, 1)
        num_classes = len(preprocessor.attack_mapping)
        model = CNNModel(input_shape, num_classes)
        
        model.set_weights(weights)
        print("Successfully loaded the trained model weights")
        
        _, X_test, _, y_test = preprocessor.load_data(
            'UNSW_NB15_training-set.csv',
            'UNSW_NB15_testing-set.csv'
        )
        
        def reshape_features(X):
            target_size = 32 * 32
            current_size = X.shape[1]
            if current_size < target_size:
                padding = np.zeros((X.shape[0], target_size - current_size))
                X = np.hstack([X, padding])
            elif current_size > target_size:
                X = X[:, :target_size]
            return X.reshape(-1, 32, 32, 1)
        
        X_test = reshape_features(X_test)
        y_test = np.ravel(y_test)
        
        accuracy, report, cm = evaluate_final_model(
            model.model, X_test, y_test, preprocessor.attack_mapping
        )
        
        print("\nEvaluation complete! Results saved in:")
        print("- final_model_confusion_matrix.png")
        print("- test_data_distribution.png")
        
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 