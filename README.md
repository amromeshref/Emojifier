# Emojifier

## Overview
This project aims to create an AI model that predicts emojis based on input sentences. It utilizes GloVe vectors for word embeddings and an LSTM network model built with TensorFlow/Keras. The project includes a Kivy-based GUI for user interaction.

## Table of Contents
1. [Word Embedding](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
    - [Data Transformation](#data-transformation)
    - [Model Training](#model-training)
    - [Model Prediction](#model-prediction)
    - [GUI Application](#gui-application)
4. [Configuration](#configuration)
5. [Contributing](#contributing)
6. [License](#license)
7. [Authors](#authors)

## Word Embedding
This project uses GloVe (Global Vectors for Word Representation) for word embedding. GloVe is an unsupervised learning algorithm for obtaining vector representations for words. The model reads GloVe vectors from the file `glove.6B.50d.txt`, which contains pre-trained word embeddings. These vectors are used to convert words in sentences to their corresponding embeddings, which are then fed into the model.</br>
The pre-trained word embeddings used in this project are obtained from the lab files inside the week 2 of the "Sequence Models" course in the "Deep Learning Specialization" on Coursera. You can find the `glove.6B.50d.txt` file used in this project at [this link](link_to_the_file_if_available).

## Dataset
The dataset used in this project is sourced from `data/emojify_data.csv`, which contains pairs of sentences and corresponding emoji labels. Each sentence is associated with an emoji label representing the emotion or meaning conveyed by the sentence. The dataset is preprocessed and transformed into GloVe vectors for training the emoji prediction model.

</br>

There are five classes of emojis in the dataset:

- :heart: (Class 0)
- :baseball: (Class 1)
- :smile: (Class 2)
- :disappointed: (Class 3)
- :fork_and_knife: (Class 4)

</br>

The dataset is obtained from the week 2 of the "Sequence Models" course in the "Deep Learning Specialization" on Coursera. </br>

This is a sample from the dataset:
<div align="center">
<img src= "images/data_set.png" style="width:700px;height:700;">
</div> 


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
1. Type the following command to install the requirements file using pip:
    ```
    pip install -r requirements.txt
    ```
1. Type the following command to use the GUI app:
   ```
   python3 app.py
   ```


