# 🐑 Sheep Breed Classification using CNN (PyTorch)

This project uses a Convolutional Neural Network (CNN) built with **PyTorch** to classify images of different sheep breeds. It was developed for the **Sheep Classification Challenge 2025** on Kaggle and supports 7 classes.


## 📦 Dataset

- **Source**: [Kaggle Sheep Classification Challenge 2025](https://www.kaggle.com/competitions/sheep-classification-challenge-2025)
- **Image Classes**:
  - Barbari
  - Goat
  - Harri
  - Naeimi
  - Najdi
  - Roman
  - Sawakni


## 🧠 Model Overview

The CNN model includes:

- 3 Convolutional layers with ReLU activation and MaxPooling
- Fully connected layers with Dropout for regularization
- Final Softmax layer for multi-class classification

All images were resized to `224x224` before training. The model was trained using CrossEntropyLoss and Adam optimizer in PyTorch.





## 🚀 How to Predict

To generate predictions using the trained model:

1. **Download the model** (`sheep_cnn.pth`) from the Google Drive link above.
2. Place it in the root directory of this project.
3. Make sure your test images are in a folder named `/test`.
4. Run the following command:

```bash
python predict.py
