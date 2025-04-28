import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import os

def load_training_history():
    """Load the federated learning history"""
    try:
        with open(os.path.join('pickle_files', 'federated_learning_history.pkl'), 'rb') as f:
            history = pickle.load(f)
        return history
    except Exception as e:
        print(f"Error loading training history: {str(e)}")
        return None

def plot_accuracy_comparison(history):
    """Plot accuracy comparison between global model and clients"""
    print("\nDebug - Accuracy Values:")
    print("Global Model Accuracies:", history['accuracy'])
    
    plt.figure(figsize=(12, 6))
    
    # Plot global model accuracy
    rounds = range(1, len(history['accuracy']) + 1)
    plt.plot(rounds, history['accuracy'], 
             'b-o', linewidth=2, markersize=8, label='Global Model')
    
    # Plot client accuracies
    colors = ['g', 'r', 'c']
    for client_id in history['client_histories'].keys():
        client_accuracies = []
        for round_history in history['client_histories'][client_id]:
            if round_history and 'accuracy' in round_history:
                client_accuracies.append(round_history['accuracy'][-1])
        
        print(f"Client {client_id} Accuracies:", client_accuracies)
        
        if client_accuracies:
            plt.plot(rounds, client_accuracies,
                    f'{colors[client_id]}-s',
                    linewidth=2,
                    markersize=8,
                    label=f'Client {client_id}')
    
    plt.title('Accuracy Comparison: Global Model vs Clients', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(rounds)
    
    # Save the plot
    os.makedirs('Histograms', exist_ok=True)
    plt.savefig('Histograms/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_recall_comparison(history, attack_types):
    """Plot recall comparison for each attack type"""
    plt.figure(figsize=(15, 8))
    
    # Get recall values for each attack type
    recalls = {attack: [] for attack in attack_types}
    for round_num in range(len(history['accuracy'])):
        report = history['classification_reports'][round_num]
        for attack in attack_types:
            if attack in report and 'recall' in report[attack]:
                recalls[attack].append(report[attack]['recall'])
    
    # Only plot if we have data
    if any(len(recall_values) > 0 for recall_values in recalls.values()):
        x = range(1, len(history['accuracy']) + 1)
        for attack, recall_values in recalls.items():
            if len(recall_values) > 0:  # Only plot if we have values
                plt.plot(x, recall_values, '-o', label=attack)
        
        plt.title('Recall Comparison by Attack Type', fontsize=14)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Recall', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(x)
        
        # Save the plot
        plt.savefig('Histograms/recall_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No recall data available for plotting yet")

def plot_attack_prediction_comparison(history, attack_types):
    """Plot attack type prediction comparison"""
    plt.figure(figsize=(15, 8))
    
    # Get prediction counts for each attack type
    predictions = {attack: [] for attack in attack_types}
    for round_num in range(len(history['accuracy'])):
        cm = history['confusion_matrices'][round_num]
        for i, attack in enumerate(attack_types):
            if i < len(cm):  # Check if we have data for this attack type
                predictions[attack].append(np.sum(cm[i, :]))
    
    # Only plot if we have data
    if any(len(pred_counts) > 0 for pred_counts in predictions.values()):
        x = range(1, len(history['accuracy']) + 1)
        for attack, pred_counts in predictions.items():
            if len(pred_counts) > 0:  # Only plot if we have values
                plt.plot(x, pred_counts, '-o', label=attack)
        
        plt.title('Attack Type Prediction Comparison', fontsize=14)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Number of Predictions', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(x)
        
        # Save the plot
        plt.savefig('Histograms/attack_prediction_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No prediction data available for plotting yet")

def main():
    print("Loading training history...")
    history = load_training_history()
    
    if history is not None:
        print("\nDebug - History Contents:")
        print("Keys in history:", history.keys())
        print("Number of rounds:", len(history['accuracy']))
        print("Client histories:", list(history['client_histories'].keys()))
        
        print("\nGenerating comparison graphs...")
        
        # Define attack types
        attack_types = ['Normal', 'Backdoor', 'Analysis', 'Fuzzers', 'Shellcode', 
                       'Reconnaissance', 'Exploits', 'DoS', 'Worms', 'Generic']
        
        # Generate comparison graphs
        print("Plotting accuracy comparison...")
        plot_accuracy_comparison(history)
        print("Plotting recall comparison...")
        plot_recall_comparison(history, attack_types)
        print("Plotting attack prediction comparison...")
        plot_attack_prediction_comparison(history, attack_types)
        
        print("\nComparison graphs generated successfully!")
        print("Check the Histograms directory for:")
        print("1. accuracy_comparison.png")
        print("2. recall_comparison.png")
        print("3. attack_prediction_comparison.png")
    else:
        print("Could not load training history.")

if __name__ == "__main__":
    main() 