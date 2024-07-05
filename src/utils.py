import sys
import os
# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
sys.path.append(REPO_DIR_PATH)

from src.exception import CustomException
from src.logger import logging
import numpy as np
import yaml


CONFIG_FILE_PATH = os.path.join(REPO_DIR_PATH, "config.yaml")


def read_glove_vectors(glove_file: str) -> tuple[set, dict[str, np.ndarray]]:
    """
    This function will read the glove vectors that are the embeddings of words.
    input: 
        glove_file: glove file path
    output:
        words: set of words
        word_to_vec_map: dictionary of words and their vectors
    """
    with open(glove_file, 'r') as f:
        try:
            # Initialize the words and word_to_vec_map
            words = set()
            word_to_vec_map = {}
            
            # Loop through each line in the file
            for line in f:
                # Split the line by spaces
                line = line.strip().split()
                # The first element is the word
                curr_word = line[0]
                # Add the word to the set of words
                words.add(curr_word)
                # The rest of the elements are the vector representation of the word
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
                
            return (words, word_to_vec_map)
        
        except Exception as e:
            logging.error("Error: "+str(e))
            raise CustomException(e, sys)

def load_config() -> dict:
    """
    This function will load the yaml configuration file.
    input: 
        None
    output:
        config: configuration file
    """
    try:
        with open(CONFIG_FILE_PATH) as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        logging.error("Error: "+str(e))
        raise CustomException(e, sys)