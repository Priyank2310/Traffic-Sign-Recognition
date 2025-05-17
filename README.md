# 🚦 Traffic Sign Recognition using Deep Learning (ANN, CNN, and Deep CNN)

## 📌 Problem Statement

The goal of this project is to classify traffic signs into one of 43 categories using deep learning techniques. Specifically, we aim to build and compare three neural network models:

1. **Artificial Neural Network (ANN)** — A simple dense neural network to serve as a baseline.
2. **Convolutional Neural Network (CNN)** — A model leveraging convolution layers to learn spatial hierarchies in image data.
3. **Deep Convolutional Neural Network (Deep CNN)** — A deeper architecture with more convolutional layers to improve feature extraction and accuracy.

We evaluate these models on the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset using metrics such as accuracy, confusion matrix, and classification report.

---

## 📚 Dataset

- **Dataset Name**: GTSRB (German Traffic Sign Recognition Benchmark)  
- **Link**: [GTSRB Dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?select=Train)
- **Description**: The GTSRB dataset contains over 50,000 images of 43 different traffic sign classes. Images vary in size and are preprocessed (resized, normalized) for input into deep learning models.

### Dataset Structure:
  
- **Train/**: Directory containing training images categorized into 43 traffic sign classes.

- **Test/**: Directory containing test images for model evaluation.

### Classes:
- 43 different classes of traffic signs, including:
  - **Speed Limit (20 km/h)**
  - **Speed Limit (30 km/h)**
  - **Speed Limit (50 km/h)**
  - **Yield**
  - **Stop**
  - **No Entry**
  - And more…

### Features Used:
- **Image Data**: RGB images, resized to 32x32 pixels
- **Preprocessing**:
  - Images have been normalized and resized to ensure consistency across the dataset.
  - Each image is labeled with one of the 43 traffic sign classes.

---

## 💡 Model Architectures and Performance

### 1. **Artificial Neural Network (ANN)**

- **Architecture**: 
  - Input → Flatten → Dense (ReLU) → Dropout → Dense (Softmax)
- **Test Accuracy**: **28.84%**
- **Remarks**: Serves as a baseline model. Performs poorly due to lack of spatial feature learning.

### 2. **Convolutional Neural Network (CNN)**

- **Architecture**:
  - Conv2D + ReLU → MaxPooling → BatchNormalization → Dropout → Dense Layers
- **Test Accuracy**: **99.67%**
- **Remarks**: Learns image features better and generalizes well.

### 3. **Deep Convolutional Neural Network (Deep CNN)**

- **Architecture**:
  - Additional Conv2D layers, deeper hierarchy, increased filters, BatchNorm, and Dropout
- **Test Accuracy**: **99.82%**
- **Remarks**: Best performing model, captures detailed and abstract features from the image.

---

## 📊 Evaluation

### Confusion Matrix & Classification Reports

- Confusion matrices and classification reports for all three models have been plotted and included in the `.ipynb` notebook files.
- These metrics provide insight into class-wise precision, recall, and F1-score.

---

## 🧪 Model Summary

| Model     | Test Accuracy | Remarks                                      |
|-----------|----------------|----------------------------------------------|
| ANN       | 28.84%         | Baseline model, struggles with image data    |
| CNN       | 99.67%         | Strong feature learning, generalizes well    |
| Deep CNN  | 99.82%         | Highest accuracy, robust and deep architecture |

---

## 🔧 Tools & Libraries Used

- **Programming Language**: Python
- **Libraries**:
  - TensorFlow / Keras
  - NumPy, Pandas
  - Matplotlib, Seaborn
  - Scikit-learn (metrics and visualization)

---

## ✅ Conclusion

- **ANNs** are not ideal for image classification as they do not preserve spatial relationships between pixels.
- **CNNs** significantly improve performance by learning local patterns through convolutional layers.
- **Deep CNNs** further enhance accuracy with deeper networks and better regularization techniques.
- The transition from ANN to Deep CNN results in an accuracy improvement of over **70%**, highlighting the importance of model architecture in image classification.
- Future improvements may include:
  - Transfer learning with pre-trained models (e.g., ResNet, EfficientNet)
  - Hyperparameter tuning using tools like Optuna or Keras Tuner

---

## 🔧 Setup Instructions

### 📦 Install Dependencies

To get started, install the required dependencies using the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
