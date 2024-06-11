# Digit Recognizer

![MNIST Digits](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Testing the Model](#testing-the-model)
- [Displaying Predictions](#displaying-predictions)
- [Image Augmentation](#image-augmentation)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Issues](#issues)
- [License](#license)
- [Contact](#contact)

## Overview

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits using the MNIST dataset. The goal is to accurately classify images of digits (0-9) by learning from a vast dataset of labeled handwritten digits.

This model achieves a high validation accuracy and is trained using various image preprocessing and augmentation techniques to enhance its performance. The repository includes scripts for training the model, testing its performance, and visualizing predictions.

## Project Structure

The repository is structured as follows:

```
Digit_Recognizer/
│
├── data/
│   ├── train.csv        # Training data
│   ├── test.csv         # Test data
│   └── sample_submission.csv  # Sample submission file
│
├── display_random.py    # Script to display random test images with predictions
├── model_architecture.json  # JSON file containing the model architecture
├── model_weights.h5     # HDF5 file containing the model weights
├── predictions.csv      # Output file for predictions on test data
├── test.py              # Script to generate predictions on test data
├── train.py             # Script to train the model
│
└── README.md            # Readme file
```

## Data

The project uses the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits, each labeled with a corresponding digit from 0 to 9. The images are grayscale and have a resolution of 28x28 pixels.

## Model Architecture

The model implemented is a Convolutional Neural Network (CNN), designed to handle the spatial structure of image data efficiently. The architecture includes:

- Convolutional layers for feature extraction.
- Max-pooling layers to reduce spatial dimensions.
- Fully connected layers to perform classification based on the extracted features.

This combination allows the model to capture the relevant patterns and characteristics of handwritten digits, leading to high classification accuracy.

## Training the Model

The model is trained using the `train.py` script, which:
- Loads the training data from `train.csv`.
- Preprocesses the data by normalizing pixel values.
- Augments the data to improve generalization.
- Compiles the model with an appropriate loss function and optimizer.
- Trains the model over a set number of epochs, saving the best model weights.

### Running the Training Script

To train the model, execute the following command in your terminal:

```bash
python train.py
```

## Testing the Model

Once the model is trained, you can evaluate its performance on the test set using the `test.py` script. This script:
- Loads the test data from `test.csv`.
- Preprocesses the data similarly to the training set.
- Uses the trained model to predict labels for the test images.
- Saves the predictions to `predictions.csv`.

### Running the Testing Script

To generate predictions on the test set, run:

```bash
python test.py
```

## Displaying Predictions

The `display_random.py` script allows you to visualize a random selection of test images along with their predicted labels. This can be useful for understanding how the model performs on individual samples.

### Running the Display Script

To display random predictions, use the command:

```bash
python display_random.py
```

## Image Augmentation

One of the key techniques employed in this project is image augmentation. This involves creating variations of the training images through transformations such as rotation, scaling, and shifting. Augmentation helps the model generalize better by exposing it to a wider variety of image variations during training.

## Requirements

The project requires Python 3 and several Python libraries. The key dependencies include:

- TensorFlow
- Keras
- NumPy
- pandas
- matplotlib

## Installation

To set up the environment for this project, you can follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/chandreyeeshome/Digit_Recognizer.git
   cd Digit_Recognizer
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After setting up the environment, you can train, test, and visualize the model using the provided scripts. Here’s a quick guide:

1. **Train the model:**

   ```bash
   python train.py
   ```

2. **Generate predictions on the test set:**

   ```bash
   python test.py
   ```

3. **Display random test images with predictions:**

   ```bash
   python display_random.py
   ```

## Contributing

Contributions are welcome! If you have any improvements or new features to propose, please fork the repository and create a pull request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For any questions or feedback, feel free to contact me through GitHub or at [my email](mailto:chandreyeeshome04@gmail.com).

---

This README provides a thorough guide to understanding and using the Digit Recognizer project. Whether you're looking to train a model, generate predictions, or just explore the code, you should find all the information you need right here.

If you have any suggestions for improving this README or the project itself, please let me know!

---

**References:**
- [Digit Recognizer - Kaggle](https://www.kaggle.com/c/digit-recognizer)
- [GitHub - CNN Digit Recognizer](https://github.com/andikarachman/CNN-Digit-Recognizer)
- [GitHub - Softmax Digit Recognizer](https://github.com/akshaybahadur21/Digit-Recognizer)
