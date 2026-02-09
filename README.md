# Diabetes Prediction System 

A Machine Learning based web application that predicts whether a person is diabetic or not using medical parameters.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn (SVM)
- Streamlit
- Pickle

## Dataset
- PIMA Indians Diabetes Dataset

## Machine Learning Model
- Algorithm: Support Vector Machine (SVM)
- Kernel: Linear
- Feature Scaling: StandardScaler

## Workflow
1. Data preprocessing and feature scaling
2. Model training using SVM
3. Saving trained model and scaler using pickle
4. Loading model in Streamlit app
5. Real-time prediction without retraining

## How to Run the Project

```bash
pip install -r requirements.txt
streamlit run app.py

## Project Structure

Diabetes_Prediction/
│
├── app.py
├── diabetes.csv
├── diabetes_model.pkl
├── scaler.pkl
├── img.jpeg
├── requirements.txt
├── README.md
└── Diabetes_Prediction.ipynb

Output

The app predicts whether a person has diabetes based on input medical parameters such as glucose level, BMI, age, etc.

Author

Raj Nikam
Github- @Raj-nikam26
