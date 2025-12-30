import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import zipfile
import os

@st.cache_data
def load_data():
    zip_path = "EEG_Eye_State_Classification.csv.zip"
    extract_path = "data"

    # Extract zip if not already extracted
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    # Read CSV from extracted folder
    csv_path = os.path.join(extract_path, "EEG_Eye_State_Classification.csv")
    df = pd.read_csv(csv_path)
    return df

# ---------------------------------
# App Title
# ---------------------------------
st.set_page_config(page_title="EEG Eye State Classification", layout="centered")
st.title("🧠 EEG Eye State Classification")
st.write("Compare ML models to predict **Eyes Open / Closed** from EEG signals")

# ---------------------------------
# Load Dataset
# ---------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("EEG_Eye_State_Classification.csv")
    return df

df = load_data()

# ---------------------------------
# Show Dataset
# ---------------------------------
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())

# ---------------------------------
# Preprocessing
# ---------------------------------
X = df.drop("eyeDetection", axis=1)
y = df["eyeDetection"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ---------------------------------
# Model Selection
# ---------------------------------
model_choice = st.selectbox(
    "Select Machine Learning Model",
    ("Logistic Regression", "Naive Bayes", "SVM", "KNN")
)

# ---------------------------------
# Train Selected Model
# ---------------------------------
if model_choice == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)

elif model_choice == "Naive Bayes":
    model = GaussianNB()

elif model_choice == "SVM":
    model = SVC()

elif model_choice == "KNN":
    k = st.slider("Select number of neighbors (K)", 3, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)

# ---------------------------------
# Train & Evaluate
# ---------------------------------
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

st.subheader("📊 Model Performance")
st.success(f"Accuracy: {accuracy:.4f}")

# ---------------------------------
# Manual Prediction Section
# ---------------------------------
st.subheader("🧪 Test with Custom EEG Values")

user_input = []
for i in range(14):
    value = st.number_input(f"EEG Feature {i+1}", value=0.0)
    user_input.append(value)

if st.button("Predict Eye State"):
    input_scaled = scaler.transform([user_input])
    prediction = model.predict(input_scaled)[0]

    if prediction == 0:
        st.info("👀 Eyes Open")
    else:
        st.warning("😴 Eyes Closed")
