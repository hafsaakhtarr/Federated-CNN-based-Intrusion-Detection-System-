import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
import numpy as np

class IDSModel:
    def __init__(self, input_shape=(32, 32, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        self.history = None

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        return self.history

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = models.load_model(filepath)

class CNNModel:
    def __init__(self, num_classes):
        self.model = None
        self.num_classes = num_classes
        self.build_model()
    
    def attention_block(self, x, filters):
        """Attention mechanism to focus on important features"""
        # Channel attention
        channel_attention = layers.GlobalAveragePooling2D()(x)
        channel_attention = layers.Dense(filters//8, activation='relu')(channel_attention)
        channel_attention = layers.Dense(filters, activation='sigmoid')(channel_attention)
        channel_attention = layers.Reshape((1, 1, filters))(channel_attention)
        
        # Spatial attention
        spatial_attention = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(x)
        
        # Combine attention
        x = layers.Multiply()([x, channel_attention])
        x = layers.Multiply()([x, spatial_attention])
        return x
    
    def residual_block(self, x, filters, kernel_size=3):
        """Residual block with skip connection"""
        shortcut = x
        
        # First convolution
        x = layers.Conv2D(filters, kernel_size, padding='same', 
                         kernel_regularizer=l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Second convolution
        x = layers.Conv2D(filters, kernel_size, padding='same',
                         kernel_regularizer=l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        
        # Skip connection
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x
    
    def build_model(self):
        """Build the optimized CNN model architecture"""
        self.model = models.Sequential([
            # Input layer
            layers.Input(shape=(32, 32, 1)),
            
            # First convolutional block
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and dense layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile with Adam optimizer and learning rate scheduling
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        self.model.summary()
    
    def fit(self, X_train, y_train, epochs=3, batch_size=256, validation_split=0.2, class_weight=None):
        """Train the model with optimized parameters"""
        # Learning rate scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=1,
            min_lr=1e-6
        )
        
        # Early stopping with more aggressive monitoring
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=2,
            restore_best_weights=True,
            mode='max',
            min_delta=0.001
        )
        
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            class_weight=class_weight,
            callbacks=[lr_scheduler, early_stopping],
            verbose=1
        )
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model"""
        return self.model.evaluate(X_test, y_test, verbose=1)
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def get_weights(self):
        """Get model weights"""
        return self.model.get_weights()
    
    def set_weights(self, weights):
        """Set model weights"""
        self.model.set_weights(weights) 