import sys
import os

# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.append(REPO_DIR_PATH)

from src.exception import CustomException
from src.logger import logging
from src.data_transformation import DataTransformer
import tensorflow as tf
import numpy as np

MODEL_PATH = os.path.join(REPO_DIR_PATH, "models/best/emojifier_model.h5")

class ModelPredictor(DataTransformer):
    def __init__(self):
        super().__init__()
        # Load the model
        self.model = self.load_model()

    def load_model(self) -> tf.keras.models.Model:
        """
        This function will load the trained model.
        input: 
            None
        output:
            model: trained model
        """
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)
    
    def predict(self, sentence: str) -> int:
        """
        This function will predict the emoji for the given sentence.
        input: 
            sentence: input sentence(string)
        output:
            prediction: predicted emoji index
        """
        try:
            # Convert the sentence to GloVe vectors
            X = self.convert_sentence_to_glove_vectors(sentence)

            # Reshape the input for the model
            X = X.reshape(1, X.shape[0], X.shape[1])

            # Predict the emoji index
            prediction = self.model.predict(X)
            prediction = tf.nn.softmax(prediction)
            prediction = np.argmax(prediction)
            return prediction
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)
    
    def predict_emoji(self, sentence: str) -> str:
        """
        This function will return the emoji for the given sentence.
        input: 
            sentence: input sentence(string)
        output:
            emoji: emoji for the given sentence
        """
        try:
            prediction = self.predict(sentence)
            if prediction == 0:
                emoji = "❤️"
            elif prediction == 1:
                emoji = "⚾️"
            elif prediction == 2:
                emoji = "😄"
            elif prediction == 3:
                emoji = "😞"
            elif prediction == 4:
                emoji = "🍴"
            return emoji              
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)