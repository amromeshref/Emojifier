import sys
import os

# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.append(REPO_DIR_PATH)

from src.utils import read_glove_vectors, load_config
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd

class DataTransformer:
    def __init__(self):
        # Load the configuration
        self.config = load_config()
        # Get the maximum number of words in a sentence
        self.max_words_in_sentence = self.config["max_words_in_sentence"]
        # Load the GloVe vectors and words
        self.glove_words, self.word_to_vec_map = read_glove_vectors(
            os.path.join(REPO_DIR_PATH, "data/glove.6B.50d.txt"))


    def convert_sentence_to_glove_vectors(self, sentence: str) -> np.ndarray:
        """
        This function will convert the sentence to GloVe vectors.
        input: 
            sentence: input sentence(string)
        output:
            sentence_vector: GloVe vectors of the sentence(2D numpy array of shape (max_words_in_sentence, glove_vector_dim))
        """
        try:     
            # Get the dimension of the GloVe vectors
            glove_vector_dim = self.word_to_vec_map["unknown"].shape[0]

            # Convert the sentence to lowercase and split it into words
            words = sentence.lower().split()

            # If the number of words in the sentence is greater than the maximum allowed words, truncate the sentence
            if len(words) > self.max_words_in_sentence:
                words = words[:self.max_words_in_sentence]

            # Initialize the vector representation of the sentence
            sentence_vector = []
            
            for word in words:
                # If the word is in the GloVe words, add the vector representation to the sentence vector
                if word in self.glove_words:
                    sentence_vector.append(self.word_to_vec_map[word])
                else:
                    # If the word is not in the GloVe words, add the vector representation of "unknown" to the sentence vector
                    sentence_vector.append(self.word_to_vec_map["unknown"])
            
            # Pad the sentence vector with zeros if the number of words is less than the maximum allowed words
            while len(sentence_vector) < self.max_words_in_sentence:
                sentence_vector.append(np.zeros(glove_vector_dim))
            
            # Convert the sentence vector to a numpy array
            sentence_vector = np.array(sentence_vector)

            return sentence_vector
        
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

    def transform_data(self) -> list[tuple[np.ndarray, str]]:
        """
        This function will transform the input data to GloVe vectors.
        input: 
            data: input data(list of strings)
        output:
            transformed_data: GloVe vectors of the data(3D numpy array of shape (number_of_samples, max_words_in_sentence, glove_vector_dim))
        """
        try:
            # Initialize the transformed data
            transformed_data = []
            
            # Load the Emojify data
            emojify_data_path = os.path.join(REPO_DIR_PATH, "data/emojify_data.csv")
            emojify_data = pd.read_csv(emojify_data_path)

            # Get the sentences and labels from the Emojify data    
            sentences = []
            labels = []
            for i in range(len(emojify_data)):
                sentences.append(emojify_data.iloc[:,0][i])
                labels.append(emojify_data.iloc[:,1][i])
            
            # Initialize the transformed data
            transformed_data = []

            # Loop through each sentence in the data and convert it to GloVe vectors
            for i in range (len(sentences)):
                transformed_data.append((self.convert_sentence_to_glove_vectors(sentences[i]), labels[i]))
            
            return transformed_data
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)    

    def save_transformed_data(self) -> None:
        """
        This function will save the transformed data to a .npy file.
        input: 
            None
        output:
            None
        """
        try:
            # Get the transformed data
            transformed_data = self.transform_data()
            # Convert the transformed data to a numpy array
            transformed_data = np.array(transformed_data, dtype=object)
            # Initialize the transformed data
            transformed_data_path = os.path.join(REPO_DIR_PATH, "data/transformed_data.npy")
            # Save the transformed data to a file
            np.save(transformed_data_path, transformed_data)
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_transformer = DataTransformer()
    data_transformer.save_transformed_data()