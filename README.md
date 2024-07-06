# Emojifier

## Overview
This project aims to create an AI model that predicts emojis based on input sentences. It utilizes GloVe vectors for word embeddings and an LSTM network model built with TensorFlow/Keras. The project includes a Kivy-based GUI for user interaction.

## Table of Contents
1. [Word Embedding](#word-embedding)
1. [Dataset](#dataset)
1. [Project Structure](#project-structure)
1. [Model Prediction](#model-prediction)
3. [Installation](#installation)

## Word Embedding
This project uses GloVe (Global Vectors for Word Representation) for word embedding. GloVe is an unsupervised learning algorithm for obtaining vector representations for words. The model reads GloVe vectors from the file `glove.6B.50d.txt`, which contains pre-trained word embeddings. These vectors are used to convert words in sentences to their corresponding embeddings, which are then fed into the model.</br>
The pre-trained word embeddings used in this project are obtained from the lab files inside the week 2 of the "Sequence Models" course in the "Deep Learning Specialization" on Coursera. You can find the `glove.6B.50d.txt` file used in this project at this [link](https://drive.google.com/file/d/11UPs7aFhyxkGfVGrMTqa2wdnFBFC-q4b/view?usp=sharing).

## Dataset
The dataset used in this project is sourced from `data/emojify_data.csv`, which contains pairs of sentences and corresponding emoji labels. Each sentence is associated with an emoji label representing the emotion or meaning conveyed by the sentence. The dataset is preprocessed and transformed into GloVe vectors for training the emoji prediction model. The dataset is obtained from the week 2 of the "Sequence Models" course in the "Deep Learning Specialization" on Coursera.

</br>

There are five classes of emojis in the dataset:

- :heart: (Class 0)
- :baseball: (Class 1)
- :smile: (Class 2)
- :disappointed: (Class 3)
- :fork_and_knife: (Class 4)

</br>

This is a sample from the dataset:
<div align="center">
<img src= "images/data_set.png" style="width:700px;height:700;">
</div> 


## Project Structure
- **src/data_transformation.py**: Handles the conversion of sentences into GloVe vectors and transforms the dataset.
- **src/predict_model.py**: Loads the trained model and predicts emojis for given sentences.
- **src/train_model.py**: Trains the neural network model using the transformed data and evaluates its performance. The model includes LSTM layers to process the sequence of word embeddings.
- **src/utils.py**: Contains utility functions for reading GloVe vectors and loading configuration settings.
- **app.py**: Creates a GUI using Kivy for users to input sentences and get emoji predictions.

##Model Prediction
To predict an emoji for a given sentence:

```python
from src.predict_model import ModelPredictor

# Initialize the predictor
predictor = ModelPredictor()

# Example sentence
sentence = "I am happy"

# Predict emoji index
prediction = predictor.predict(sentence)

# Output the predicted emoji index
print(f"Predicted emoji index: {prediction}")
```

## Installation
1. Create a new environment with a 3.9 Python version.
1. Create a directory on your device and navigate to it.
1. Clone the repository:
   ```
   git clone https://github.com/amromeshref/Emojifier.git
   ```
1. Navigate to the Emojifier directory.
   ```
   cd Emojifier
   ```
1. Download this [file](https://drive.google.com/file/d/11UPs7aFhyxkGfVGrMTqa2wdnFBFC-q4b/view?usp=sharing) and put it in the `data/` directory.
1. Type the following command to install the requirements file using pip:
    ```
    pip install -r requirements.txt
    ```
1. Type the following command to use the GUI app:
   ```
   python3 app.py
   ```


