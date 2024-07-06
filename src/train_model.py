import sys
import os

# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.append(REPO_DIR_PATH)


from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.nn import softmax
from datetime import datetime
import argparse

class ModelTrainer:
    def __init__(self):
        pass

    def load_data(self) -> np.ndarray:
        """
        This function will load the transformed data that will be used for training the model.
        input: 
            None
        output:
            data: transformed data
        """
        try:
            # Load the transformed data
            data_path = os.path.join(REPO_DIR_PATH, "data/transformed_data.npy")
            data = np.load(data_path, allow_pickle=True)
            return data
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)
    
    def split_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        This function will split the data into training and testing sets.
        input: 
            None
        output:
            X_train: training data
            y_train: training labels
            X_test: testing data
            y_test: testing labels
        """
        try:    
            # Load the transformed data
            data = self.load_data()
            
            # Split the data into features and labels
            X = [d[0] for d in data]
            Y = [d[1] for d in data]

            # Convert the lists to numpy arrays
            X = np.array(X)
            Y = np.array(Y).reshape((len(Y),1))
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, Y, test_size=0.2, random_state=42)
            
            return (X_train, y_train, X_test, y_test)
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)
    
    def create_model(self) -> Sequential:
        """
        This function will create the model.
        input:
            None
        output:
            model: Sequential model
        """
        input_shape = (10,50)
        model = Sequential(
            [
                LSTM(128, return_sequences=True, input_shape = input_shape),
                Dropout(0.5),
                LSTM(128, return_sequences=True),
                Dropout(0.5),
                LSTM(128, return_sequences=False),
                Dropout(0.5),
                Dense(5, activation = "linear")
            ]
        )    
        return model
    
    def evaluate(self, y_pred, y_true) -> float:
        """
        This function will evaluate the model.
        input:
            y_pred: predicted values
            y_true: true values
        output:
            accuracy: accuracy of the model
        """
        y_pred = softmax(y_pred)
        output = np.zeros((y_true.shape[0],1), dtype = "int")
        i = 0
        for y in y_pred:
            output[i] = int(np.argmax(y))
            i += 1
        accuracy = np.sum(output==y_true)/y_true.shape[0]
        return accuracy

    def train(self, epochs, batch_size):
        """
        This function will train the model.
        input:
            None
        output:
            None
        """
        # Split the data into training and testing sets
        X_train, y_train, X_test, y_test = self.split_data()

        # Create the model
        model = self.create_model()

        # Compile the model with the optimizer, loss function and metrics
        model.compile(metrics = ['accuracy'], 
               optimizer = Adam(learning_rate = 0.01),
              loss = SparseCategoricalCrossentropy(from_logits = True))
        
        # Train the model
        model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

        # Save the model 
        date_time = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
        model.save(os.path.join(REPO_DIR_PATH, "models/other/emojifier_model_"+date_time+".h5"))

        # Evaluate the model on the training and testing data
        train_data_accuracy = self.evaluate(model.predict(X_train), y_train)
        test_data_accuracy = self.evaluate(model.predict(X_test), y_test)

        print("Model trained successfully for {} epochs and batch size of {}".format(epochs, batch_size))
        print("Model saved successfully at: models/other/emojifier_model_"+date_time+".h5")

        print("Train data accuracy: ", train_data_accuracy)
        print("Test data accuracy: ", test_data_accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Emojifier model.")
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs for training the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training the model')
    args = parser.parse_args()

    model_trainer = ModelTrainer()
    model_trainer.train(epochs=args.epochs, batch_size=args.batch_size)