import pandas as pd
import numpy as np
import pickle
import streamlit as st
import base64

# Function to read and encode the image file
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Set the background image using CSS
def set_background(image_base64):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;
    }}
    .css-1g8v9l0 {{
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
    }}
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: white;
    }}
    .stButton > button {{
        background-color: #4C4C6D;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call the function with the uploaded background image
image_base64 = get_base64_image("image.jpg")
set_background(image_base64)

# Streamlit Frontend
st.markdown("<h1 style='text-align: center;'>Customer Retention Prediction</h1>", unsafe_allow_html=True)

# Define the models
models = {
    "Logistic Regression": 'logistic_regression_model.pkl',
    "Decision Tree Classifier": 'decision_tree_model.pkl',
    "Random Forest Classifier": 'random_forest_model.pkl',
    "AdaBoost Classifier": 'adaboost_model.pkl',
    "Gradient Boosting Classifier": 'gradient_boost_model.pkl',
    "XGBoost Classifier": 'xgboost_model.pkl',
    "Final Gradient Boosting Classifier": "final_gb_model.pkl",
    "Final XGBoosting Classifier": "final_xgb_model.pkl"
}

# Select the model
selected_model = st.selectbox("Choose a model", list(models.keys()))

# Input features for prediction
features = {
    "gender": st.selectbox("Gender", ['Male', 'Female']),
    "SeniorCitizen": st.selectbox("Senior Citizen", [0, 1]),
    "Partner": st.selectbox("Partner", ['Yes', 'No']),
    "Dependents": st.selectbox("Dependents", ['Yes', 'No']),
    "tenure": st.number_input("Tenure (months)", min_value=0, max_value=100),
    "PhoneService": st.selectbox("Phone Service", ['Yes', 'No']),
    "MultipleLines": st.selectbox("Multiple Lines", ['Yes', 'No phone service', 'Yes', 'No']),
    "InternetService": st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No']),
    "OnlineSecurity": st.selectbox("Online Security", ['Yes', 'No', 'No internet service']),
    "OnlineBackup": st.selectbox("Online Backup", ['Yes', 'No', 'No internet service']),
    "DeviceProtection": st.selectbox("Device Protection", ['Yes', 'No', 'No internet service']),
    "TechSupport": st.selectbox("Tech Support", ['Yes', 'No', 'No internet service']),
    "StreamingTV": st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service']),
    "StreamingMovies": st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service']),
    "Contract": st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year']),
    "PaperlessBilling": st.selectbox("Paperless Billing", ['Yes', 'No']),
    "PaymentMethod": st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']),
    "MonthlyCharges": st.number_input("Monthly Charges", min_value=0.0),
    "TotalCharges": st.number_input("Total Charges", min_value=0.0)
}

# Categorical encoding
def encode_inputs(inputs):
    inputs['gender'] = 1 if inputs['gender'] == 'Male' else 0
    inputs['Partner'] = 1 if inputs['Partner'] == 'Yes' else 0
    inputs['Dependents'] = 1 if inputs['Dependents'] == 'Yes' else 0
    inputs['PhoneService'] = 1 if inputs['PhoneService'] == 'Yes' else 0
    inputs['MultipleLines'] = 1 if inputs['MultipleLines'] == 'Yes' else (2 if inputs['MultipleLines'] == 'No phone service' else 0)
    inputs['InternetService'] = 1 if inputs['InternetService'] == 'DSL' else (2 if inputs['InternetService'] == 'Fiber optic' else 0)
    inputs['OnlineSecurity'] = 1 if inputs['OnlineSecurity'] == 'Yes' else (2 if inputs['OnlineSecurity'] == 'No internet service' else 0)
    inputs['OnlineBackup'] = 1 if inputs['OnlineBackup'] == 'Yes' else (2 if inputs['OnlineBackup'] == 'No internet service' else 0)
    inputs['DeviceProtection'] = 1 if inputs['DeviceProtection'] == 'Yes' else (2 if inputs['DeviceProtection'] == 'No internet service' else 0)
    inputs['TechSupport'] = 1 if inputs['TechSupport'] == 'Yes' else (2 if inputs['TechSupport'] == 'No internet service' else 0)
    inputs['StreamingTV'] = 1 if inputs['StreamingTV'] == 'Yes' else (2 if inputs['StreamingTV'] == 'No internet service' else 0)
    inputs['StreamingMovies'] = 1 if inputs['StreamingMovies'] == 'Yes' else (2 if inputs['StreamingMovies'] == 'No internet service' else 0)
    inputs['Contract'] = 0 if inputs['Contract'] == 'Month-to-month' else (1 if inputs['Contract'] == 'One year' else 2)
    inputs['PaperlessBilling'] = 1 if inputs['PaperlessBilling'] == 'Yes' else 0
    inputs['PaymentMethod'] = 0 if inputs['PaymentMethod'] == 'Electronic check' else (1 if inputs['PaymentMethod'] == 'Mailed check' else (2 if inputs['PaymentMethod'] == 'Bank transfer (automatic)' else 3))
    return inputs

# Predict button
if st.button("Predict"):
    # Convert input features to match model format
    input_data = pd.DataFrame([features])
    input_data = encode_inputs(input_data.iloc[0].to_dict())

    # Convert dictionary to 2D array (reshape to match model input)
    input_data_array = np.array(list(input_data.values())).reshape(1, -1)

    # Load the selected model
    model_file = models[selected_model]
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    # Predict churn
    prediction = model.predict(input_data_array)
    
    # Display prediction
    if prediction[0] == "Yes":
        st.success("The customer is likely to leave.")
    else:
        st.success("The customer is likely to stay.")
