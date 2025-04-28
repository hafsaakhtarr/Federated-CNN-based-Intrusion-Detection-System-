import numpy as np
from data_preprocessing import DataPreprocessor
from model import CNNModel
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf

def save_evaluation_results(model, X_test, y_test, preprocessor):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "evaluation_results"
    
    os.makedirs(results_dir, exist_ok=True)
    
    print("\nEvaluating model on test data...")
    loss, accuracy = model.model.evaluate(X_test, y_test, verbose=0)
    
    y_pred = model.model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    class_names = [preprocessor.reverse_mapping[i] for i in range(len(preprocessor.attack_mapping))]
    
    report = classification_report(
        y_test,
        y_pred_classes,
        target_names=class_names,
        zero_division=0
    )
    
    metrics_file = os.path.join(results_dir, f"metrics_{timestamp}.txt")
    with open(metrics_file, 'w') as f:
        f.write(f"Evaluation Results - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Loss: {loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_file = os.path.join(results_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    predictions_file = os.path.join(results_dir, f"predictions_{timestamp}.csv")
    np.savetxt(predictions_file, 
               np.column_stack((y_test, y_pred_classes)),
               delimiter=',',
               header='True_Label,Predicted_Label',
               fmt='%d')
    
    print(f"\nEvaluation results saved to {results_dir}/")
    print(f"- Metrics and classification report: metrics_{timestamp}.txt")
    print(f"- Confusion matrix: confusion_matrix_{timestamp}.png")
    print(f"- Predictions: predictions_{timestamp}.csv")

def evaluate_saved_model():
    print("Loading saved model and evaluating...")
    
    try:
        preprocessor = DataPreprocessor()
        print("Loading test data...")
        _, _, X_test, y_test = preprocessor.load_data(
            "UNSW_NB15_training-set.csv",
            "UNSW_NB15_testing-set.csv"
        )
        
        print("\nLoading saved model...")
        try:
            model = tf.keras.models.load_model('models/federated_model.h5')
            print("Successfully loaded saved model")
            
            print("\nEvaluating model on test data...")
            loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
            print(f"\nTest Loss: {loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            
            y_pred = model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(y_pred, axis=1)
            
            print("\nDetailed Classification Report:")
            report = classification_report(
                y_test,
                y_pred_classes,
                target_names=[preprocessor.reverse_mapping[i] for i in range(len(preprocessor.attack_mapping))],
                zero_division=0
            )
            print(report)
            
            plt.figure(figsize=(12, 10))
            cm = confusion_matrix(y_test, y_pred_classes)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[preprocessor.reverse_mapping[i] for i in range(len(preprocessor.attack_mapping))],
                        yticklabels=[preprocessor.reverse_mapping[i] for i in range(len(preprocessor.attack_mapping))])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            cm_file = 'evaluation_results/confusion_matrix_latest.png'
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nConfusion matrix saved to {cm_file}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return
            
    except Exception as e:
        print(f"\nError during evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_saved_model() 