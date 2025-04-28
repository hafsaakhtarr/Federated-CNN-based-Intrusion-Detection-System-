# Federated Learning for Network Intrusion Detection

This project implements a federated learning approach for network intrusion detection using the UNSW-NB15 dataset. The system uses a CNN-based model to classify network attacks while preserving data privacy through federated learning.



Project Structure

```
.
├── data_preprocessing.py     # Data loading and preprocessing
├── model.py                 # CNN model architecture
├── federated_learning.py    # Federated learning implementation
├── evaluate_final_model.py  # Model evaluation and visualization
├── generate_comparison_report.py  # Performance comparison
├── pickle_files/           # Directory for saved model weights and history
├── requirements.txt        # Project dependencies
└── README.md              
```

## Requirements

- Python 3.8+
- TensorFlow 2.12.0+
- NumPy 1.21.0+
- Pandas 1.3.0+
- scikit-learn 1.0.0+
- imbalanced-learn 0.9.0+
- Matplotlib 3.5.0+
- Seaborn 0.11.0+

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd federated_ids
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Preprocessing:
```bash
python data_preprocessing.py
```

2. Train the Federated Model:
```bash
python federated_learning.py
```

3. Evaluate the Final Model:
```bash
python evaluate_final_model.py
```

4. Generate Comparison Report:
```bash
python generate_comparison_report.py
```

## Model Architecture

The project uses a CNN-based model with the following architecture:
- Input layer: 32x32x1
- Convolutional layers with ReLU activation
- MaxPooling layers
- Dense layers with dropout
- Output layer with softmax activation

## Federated Learning Process

1. Data is distributed among multiple clients
2. Each client trains the model locally
3. Model weights are aggregated at the server
4. Global model is updated and distributed back to clients
5. Process repeats for multiple rounds

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Class-wise performance metrics

## Output Files

- `final_model_confusion_matrix.png`: Confusion matrix visualization
- `test_data_distribution.png`: Class distribution in test data
- `pickle_files/`: Directory containing saved model weights and training history

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here] 