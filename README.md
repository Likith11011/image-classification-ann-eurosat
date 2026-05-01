# Image Classification using ANN & CNN (EuroSAT Dataset)

## Overview

This project implements and compares two deep learning approaches for image classification:

* **Artificial Neural Network (ANN)**
* **Convolutional Neural Network (CNN)**

The objective is to classify satellite images from the EuroSAT dataset into different land-use categories and analyze the performance difference between ANN and CNN models.

---

## Dataset

* **Dataset:** EuroSAT
* **Type:** Satellite Image Classification
* **Classes:** Multiple land-use categories
* **Image Size:** 64 × 64 pixels

---

## Methodology

### Data Preprocessing

* Image resizing to 64×64
* Normalization (pixel values scaled to [0,1])
* Standardization (for ANN)
* Train-validation split (80:20)

---

## Models Implemented

### 1. Artificial Neural Network (ANN)

* Fully connected dense layers
* Batch normalization and dropout used
* Serves as a **baseline model**

---

### 2. Convolutional Neural Network (CNN)

* Conv2D + MaxPooling layers
* Extracts spatial features from images
* More effective for image classification tasks

---

## Training Details

* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Batch Size: 16 (ANN), 32 (CNN)
* Epochs: 60 (ANN), 25 (CNN)
* Callbacks:

  * Early Stopping
  * Reduce Learning Rate (ANN)
  * Model Checkpoint

---

## Model Comparison

| Model | Description             | Performance          |
| ----- | ----------------------- | -------------------- |
| ANN   | Fully connected network | Baseline performance |
| CNN   | Convolutional network   | Higher accuracy      |

---

## Results

* **ANN Accuracy:** 68%
* **CNN Accuracy:** 95%

CNN outperformed ANN due to its ability to capture spatial features in images.

---
## Note on Model Files

Trained model files are not included in this repository due to size limitations.

To generate the models, run:

python src/train_ann.py  
python src/train_cnn.py


## Output

* Trained Models:

  * `models/final_ann_model.keras`
  * `models/final_cnn_model.keras`

* Best Models:

  * `models/best_ann_model.keras`
  * `models/best_cnn_model.keras`

* Training Graphs:

  * `results/ann_accuracy.png`
  * `results/cnn_accuracy.png`

---

## Project Structure

```
image-classification-ann-eurosat/
│── data/
│── models/
│── results/
│── src/
│   ├── train_ann.py
│   └── train_cnn.py
│── requirements.txt
│── README.md
```

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

3. Run ANN model

```
python src/train_ann.py
```

4. Run CNN model

```
python src/train_cnn.py
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn
* Pillow

---

## Key Insights

* CNN significantly improves performance over ANN for image classification
* Feature extraction plays a crucial role in deep learning models
* Model comparison helps in selecting the best architecture

---

## Future Work

* Apply data augmentation techniques
* Hyperparameter tuning
* Implement advanced CNN architectures (ResNet, VGG)

---

## Author

**Likith B**
AI & Machine Learning Student
