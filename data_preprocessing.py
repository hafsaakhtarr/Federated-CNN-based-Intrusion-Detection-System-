import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

class DataPreprocessor:
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.attack_mapping = None
        self.reverse_mapping = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.smote = SMOTE(random_state=42, k_neighbors=3, sampling_strategy='auto')

    def preprocess(self):
        train_data = pd.read_csv(self.train_path)
        test_data = pd.read_csv(self.test_path)

        X_train = train_data.drop(['attack_cat', 'label'], axis=1)
        y_train = train_data['attack_cat']
        X_test = test_data.drop(['attack_cat', 'label'], axis=1)
        y_test = test_data['attack_cat']

        unique_attacks = np.unique(np.concatenate([y_train.unique(), y_test.unique()]))
        self.attack_mapping = {attack: i for i, attack in enumerate(sorted(unique_attacks))}
        self.reverse_mapping = {i: attack for attack, i in self.attack_mapping.items()}

        y_train = y_train.map(self.attack_mapping)
        y_test = y_test.map(self.attack_mapping)

        X_train, X_test = self.preprocess_features(X_train, X_test)

        class_counts = np.bincount(y_train)
        max_count = np.max(class_counts)

        sampling_strategy = {}
        for class_idx in range(len(class_counts)):
            if class_counts[class_idx] > 0:
                if class_counts[class_idx] < max_count * 0.2:
                    target_count = min(int(class_counts[class_idx] * 1.3), int(max_count * 0.2))
                else:
                    target_count = class_counts[class_idx]
            sampling_strategy[class_idx] = target_count

        self.smote.sampling_strategy = sampling_strategy

        X_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_resampled, y_resampled = self.smote.fit_resample(X_reshaped, y_train)
        X_resampled = X_resampled.reshape(-1, 32, 32, 1)

        print("\nClass distribution before SMOTE:")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} samples ({count/len(y_train)*100:.2f}%)")

        print("\nClass distribution after SMOTE:")
        unique, counts = np.unique(y_resampled, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} samples ({count/len(y_resampled)*100:.2f}%)")

        self.X_train = X_resampled
        self.y_train = y_resampled
        self.X_test = X_test
        self.y_test = y_test

        return self.X_train, self.y_train, self.X_test, self.y_test

    def preprocess_features(self, X_train, X_test):
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X_train.select_dtypes(include=['object']).columns

        X_train[numeric_cols] = self.imputer.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.imputer.transform(X_test[numeric_cols])

        X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])

        if len(categorical_cols) > 0:
            X_train[categorical_cols] = X_train[categorical_cols].astype(str)
            X_test[categorical_cols] = X_test[categorical_cols].astype(str)

            train_categorical = self.encoder.fit_transform(X_train[categorical_cols])
            test_categorical = self.encoder.transform(X_test[categorical_cols])

            new_cols = []
            for i, col in enumerate(categorical_cols):
                categories = self.encoder.categories_[i]
                for cat in categories:
                    new_cols.append(f"{col}_{cat}")

            train_categorical_df = pd.DataFrame(train_categorical, columns=new_cols, index=X_train.index)
            test_categorical_df = pd.DataFrame(test_categorical, columns=new_cols, index=X_test.index)

            X_train = pd.concat([X_train[numeric_cols], train_categorical_df], axis=1)
            X_test = pd.concat([X_test[numeric_cols], test_categorical_df], axis=1)

        target_size = 32 * 32
        if X_train.shape[1] < target_size:
            X_train_padded = np.zeros((X_train.shape[0], target_size))
            X_test_padded = np.zeros((X_test.shape[0], target_size))
            X_train_padded[:, :X_train.shape[1]] = X_train
            X_test_padded[:, :X_test.shape[1]] = X_test
            X_train = X_train_padded
            X_test = X_test_padded
        else:
            X_train = X_train[:, :target_size]
            X_test = X_test[:, :target_size]

        X_train = X_train.reshape(-1, 32, 32, 1)
        X_test = X_test.reshape(-1, 32, 32, 1)

        return X_train, X_test

    def get_attack_mapping(self):
        return self.attack_mapping, self.reverse_mapping 