Kidney Malignancy Classification using Machine Learning & Deep Learning

This project presents an end-to-end medical image classification system for detecting **Benign vs Malignant kidney cases** from CT images.

The system compares classical machine learning techniques with deep learning models and deploys the best-performing model using Streamlit.

---

## 📌 Project Overview

Kidney tumors can be life-threatening if not detected early. This project builds a predictive system that:

- Preprocesses renal CT images
- Extracts texture-based features (GLCM)
- Trains classical ML models (SVM, Logistic Regression, Random Forest)
- Trains a Deep Learning model (ResNet18 - Transfer Learning)
- Compares performance
- Deploys the best-performing model (Random Forest) via Streamlit UI

---

## 🧠 Models Implemented

### 🔹 Classical Machine Learning
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest (Best Performing Model)

### 🔹 Deep Learning
- ResNet18 (Transfer Learning using PyTorch)

---

## 📊 Final Model Performance

| Model | Accuracy |
|-------|----------|
| SVM | ~82% |
| Logistic Regression | ~82% |
| Random Forest | ~99% |
| ResNet18 | ~93% |

✅ Random Forest achieved the highest accuracy and tumor recall, and was therefore selected for deployment.

---

## ⚙️ Feature Engineering

Texture-based features extracted using:

- GLCM Contrast
- GLCM Correlation
- GLCM Energy
- GLCM Homogeneity
- Mean Intensity
- Standard Deviation

---

## 🚀 Streamlit Deployment

The deployed app allows users to:

1. Upload a kidney CT image
2. Automatically preprocess the image
3. Extract features
4. Predict:
   - ✅ Benign
   - ⚠️ Malignant