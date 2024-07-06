# Emojifier

## Overview
This project aims to create an AI model that predicts emojis based on input sentences. It utilizes GloVe vectors for word embeddings and an LSTM network model built with TensorFlow/Keras. The project includes a Kivy-based GUI for user interaction.

## Table of Contents
1. [Project Structure](#project-structure)
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

## Project Structure
- **src/data_transformation.py**: Handles the conversion of sentences into GloVe vectors and transforms the dataset.
- **src/predict_model.py**: Loads the trained model and predicts emojis for given sentences.
- **src/train_model.py**: Trains the model using the transformed data and evaluates its performance. The model includes LSTM layers to process the sequence of word embeddings.
- **src/utils.py**: Contains utility functions for reading GloVe vectors and loading configuration settings.
- **app.py**: Creates a GUI using Kivy for users to input sentences and get emoji predictions.

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
    ```bash
    pip install -r requirements.txt
    ```

