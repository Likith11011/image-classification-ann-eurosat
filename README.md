# Image Classification using ANN (EuroSAT Dataset)

## Overview

This project implements an **Artificial Neural Network (ANN)** for image classification using the EuroSAT dataset. The goal is to classify satellite images into different land-use categories.

---

## Dataset

* **Dataset:** EuroSAT
* **Type:** Satellite Image Classification
* **Classes:** Multiple land-use categories
* Images resized to **64 × 64 pixels**

---

## Methodology

### Data Preprocessing

* Image resizing and normalization
* Standardization using mean and standard deviation
* Train-validation split (80:20)

### Model Architecture (ANN)

* Fully connected neural network
* Multiple Dense layers with ReLU activation
* Batch Normalization and Dropout for regularization
* Softmax output layer

---

## Training Details

* Optimizer: Adam (learning rate = 0.0003)
* Loss Function: Categorical Crossentropy (with label smoothing)
* Batch Size: 16
* Epochs: 60
* Callbacks:

  * Early Stopping
  * ReduceLROnPlateau
  * Model Checkpoint

---

## Results

* Model trained on EuroSAT dataset
* Validation accuracy improves with training
* ANN provides baseline performance for image classification

68% Accuracy

---

## Output

* Trained model: `final_ann_model.keras`
* Best model: `best_ann_model.keras`
* Training graphs (accuracy & loss)

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn

---

## How to Run

1. Clone the repository

```
git clone https://github.com/likith11011/image-classification-ann-eurosat.git
cd image-classification-ann-eurosat
```

2. Install dependencies

```
pip install -r requirements.txt
```

3. Run the training script

```
python src/train_ann.py
```

---

## Future Work

* Implement CNN model for performance comparison
* Improve accuracy using data augmentation
* Hyperparameter tuning

---

## Author

Likith B
AI & Machine Learning Student
