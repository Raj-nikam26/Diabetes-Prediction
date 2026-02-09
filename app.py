import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image

# ---------------- LOAD MODEL & SCALER ----------------

with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ---------------- LOAD DATASET (ONLY FOR DISPLAY) ----------------

diabetes_df = pd.read_csv('diabetes.csv')
diabetes_mean_df = diabetes_df.groupby('Outcome').mean()

# ---------------- STREAMLIT APP ----------------

def app():

    img = Image.open("img.jpeg")
    img = img.resize((200, 200))
    st.image(img, caption="Diabetes Image", width=200)

    st.title('Diabetes Prediction')

    # Sidebar inputs
    st.sidebar.title('Input Features')

    preg = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # ---------------- PREDICTION ----------------

    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    input_data_np = np.asarray(input_data).reshape(1, -1)

    # scale input
    input_data_scaled = scaler.transform(input_data_np)

    prediction = model.predict(input_data_scaled)

    st.write('Based on the input features, the model predicts:')

    if prediction[0] == 1:
        st.warning('This person has diabetes.')
    else:
        st.success('This person does not have diabetes.')

    # ---------------- DATA INSIGHTS ----------------

    st.header('Dataset Summary')
    st.write(diabetes_df.describe())

    st.header('Distribution by Outcome')
    st.write(diabetes_mean_df)


# ---------------- RUN APP ----------------

if __name__ == '__main__':
    app()
