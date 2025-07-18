
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import os
from time import time
from tqdm import tqdm

class SimpleFNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape=2048, hidden_units=[64, 32], learning_rate=0.001, 
                 epochs=100, batch_size=32, verbose=0, validation_split=0.1, **kwargs):
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.model = None
        self.classes_ = None
        
    def _build_model(self):
        """Build the TensorFlow model architecture."""
        start_time = time()
        model = models.Sequential()
        model.add(layers.Input(shape=(self.input_shape,)))
        
        for units in tqdm(self.hidden_units, desc="Adding Dense Layers"):
            model.add(layers.Dense(units, activation='relu'))
        
        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        end_time = time()
        print(f"Model built in {end_time - start_time:.2f} seconds.")

        return model

    def save_model(self, filepath):
        """
        Save the trained model and class parameters.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model (without extension)
            
        Returns:
        --------
        None
        """

        if self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save the TensorFlow model
        file = filepath.split("/")
        if file[-1] == "model":
            self.model.save(f"{filepath}.h5")
        else:
            self.model.save(f"{filepath}_model.h5")


    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_proba : array of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        predictions = self.model.predict(X, verbose=0)
        # Return probabilities for both classes
        return np.column_stack([1 - predictions, predictions])


    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
        """
        # Store classes for compatibility
        self.classes_ = np.unique(y)
        
        # Build model
        self.model = self._build_model()
        
        # Convert to numpy arrays if needed
        X = np.array(X)
        y = np.array(y)
        
        # Train the model
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=self.validation_split
        )
        
        return self


class ConfigurableCNN1DClassifier(BaseEstimator, ClassifierMixin):
    """
    Configurable 1D CNN Classifier following scikit-learn interface.
    """
    
    def __init__(self, input_shape=2048, conv_layers=[(64, 3)], ff_layers=[32], 
                 learning_rate=0.001, epochs=100, batch_size=32, verbose=0, 
                 validation_split=0.1, **kwargs):
        self.input_shape = input_shape
        self.conv_layers = conv_layers
        self.ff_layers = ff_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.validation_split = validation_split
        self.model = None
        self.classes_ = None
        
    def _build_model(self):
        """Build the TensorFlow model architecture."""
        start_time = time()
        model = models.Sequential()
        model.add(layers.Input(shape=(self.input_shape, 1)))
        
        # Add convolutional layers
        for filters, kernel_size in self.conv_layers:
            model.add(layers.Conv1D(filters=filters, kernel_size=kernel_size, activation='relu'))
            model.add(layers.MaxPooling1D(pool_size=2))
        
        model.add(layers.Flatten())
        
        # Add fully connected layers
        for units in tqdm(self.ff_layers, desc="Adding Dense Layers"):
            model.add(layers.Dense(units, activation='relu'))

        model.add(layers.Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        end_time = time()
        print(f"Model built in {end_time - start_time:.2f} seconds.")
        
        return model
    

    def fit(self, X, y):
        """
        Fit the model to training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
        """
        # Store classes for compatibility
        self.classes_ = np.unique(y)
        
        # Build model
        self.model = self._build_model()
        
        # Convert to numpy arrays and reshape for 1D CNN
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X to add the channel dimension for 1D CNN
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        # Train the model
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            validation_split=self.validation_split
        )
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        # Reshape X to add the channel dimension for 1D CNN
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X, verbose=0)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict
            
        Returns:
        --------
        y_proba : array of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X = np.array(X)
        
        # Reshape X to add the channel dimension for 1D CNN
        if len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1)
        
        predictions = self.model.predict(X, verbose=0)
        # Return probabilities for both classes
        return np.column_stack([1 - predictions, predictions])
    


    def save_model(self, filepath):
        """
        Save the trained model and class parameters.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model (without extension)
            
        Returns:
        --------
        None
        """

        if self.model is None:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save the TensorFlow model
        file = filepath.split("/")
        if file[-1] == "model":
            self.model.save(f"{filepath}.h5")
        else:
            self.model.save(f"{filepath}_model.h5")



