import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

# ===============================
# Load saved Random Forest model
# ===============================
model = joblib.load("best_random_forest.pkl")

IMG_SIZE = 224


# ===============================
# Preprocessing function
# ===============================
def preprocess_image(img):

    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = cv2.GaussianBlur(img, (3,3), 0)

    img = img / 255.0

    return img


# ===============================
# Feature Extraction (GLCM)
# ===============================
def extract_features(img):

    img_uint8 = (img * 255).astype(np.uint8)

    glcm = graycomatrix(
        img_uint8,
        distances=[1],
        angles=[0],
        symmetric=True,
        normed=True
    )

    features = [
        graycoprops(glcm, 'contrast')[0,0],
        graycoprops(glcm, 'correlation')[0,0],
        graycoprops(glcm, 'energy')[0,0],
        graycoprops(glcm, 'homogeneity')[0,0],
        np.mean(img),
        np.std(img)
    ]

    return np.array(features).reshape(1, -1)


# ===============================
# Streamlit UI
# ===============================

st.title("Kidney Malignancy Classification")
st.write("Upload a renal CT image to classify as Benign or Malignant.")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # preprocessing
    processed = preprocess_image(image)

    # feature extraction
    features = extract_features(processed)

    # prediction
    prediction = model.predict(features)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Malignant Kidney Case Detected")
    else:
        st.success("✅ Benign Kidney Case")
