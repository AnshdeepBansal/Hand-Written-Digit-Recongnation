# Handwritten Digit Recognition using MNIST Dataset

## Overview

This project focuses on building a machine learning model to recognize handwritten digits from the popular **MNIST dataset**. The dataset consists of **70,000 grayscale images** of handwritten digits (0-9), where 60,000 images are used for training, and 10,000 images are used for testing.

Our model was trained on **60,000 images** and tested on **10,000 images**, achieving an impressive **accuracy of 97.79%**.

## Dataset

The **MNIST dataset** contains 70,000 images of handwritten digits, each with dimensions **28x28 pixels**:
- **Training set**: 60,000 images
- **Testing set**: 10,000 images

Each image is labeled with the correct digit (0-9), and our goal is to predict the correct label for each image.

You can find the MNIST dataset at [OpenML - MNIST Dataset](https://www.openml.org/d/554).

## Model Architecture

To recognize the handwritten digits, we built a neural network model that processes the 28x28 pixel images. The general architecture of the model is as follows:
- **Input Layer**: Flatten the 28x28 pixel image into a vector of 784 values.
- **Hidden Layers**: Several fully connected layers with **ReLU activation**.
- **Output Layer**: A softmax layer with 10 neurons (for digits 0-9) to predict the probability distribution over each digit.

## Training and Testing

### Training:
- **Dataset**: 60,000 training images
- **Optimizer**: Adam optimizer
- **Loss Function**: Categorical cross-entropy
- **Batch Size**: 64
- **Epochs**: 10

### Testing:
- **Dataset**: 10,000 testing images
- **Accuracy Achieved**: 97.79%

## Results

The model was able to achieve an accuracy of **97.79%** on the test dataset, meaning that it correctly identified handwritten digits in most cases. This high accuracy demonstrates the effectiveness of the model for digit recognition tasks.

## Performance Metrics

- **Training Accuracy**: 97.79%
- **Loss**: Minimized to achieve the high accuracy on both training and testing datasets.
- **Confusion Matrix**: Displayed to show the performance of the model across each digit class (0-9).

## Future Improvements

- Improve the model by experimenting with more complex architectures (e.g., Convolutional Neural Networks).
- Implement data augmentation techniques to increase the size and variability of the training data.
- Explore different optimization techniques or learning rates to improve the performance further.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The **MNIST Dataset** provided by [OpenML](https://www.openml.org/d/554).
- Inspired by various digit recognition tutorials and implementations in the deep learning community.

---
