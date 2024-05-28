import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load the trained model
with open('random_forest.pickle', 'rb') as f:
    model = pickle.load(f)

# Load the scaler used during training
scaler = StandardScaler()

# Define the feature columns used during training
X_train_columns = [
    'pressure_first_finger', 'size_first_finger',
    'x_coordinate_first_finger', 'y_coordinate_first_finger', 'distance',
    'tap_LongPress', 'tap_Scroll', 'tap_Swipe', 'tap_Tap', 'tap_TwoFinger',
    'tap_Unidentified', 'app_name_Other', 'app_name_com.android.chrome',
    'app_name_com.android.documentsui', 'app_name_com.android.systemui',
    'app_name_com.android.vending', 'app_name_com.appvv.os9launcherhd',
    'app_name_com.google.android.GoogleCamera',
    'app_name_com.google.android.apps.magazines',
    'app_name_com.google.android.apps.photos',
    'app_name_com.google.android.calendar',
    'app_name_com.google.android.gms',
    'app_name_com.google.android.googlequicksearchbox',
    'app_name_com.google.android.play.games',
    'app_name_com.google.android.youtube',
    'app_name_com.isispoly.DemoDrawAPIN',
    'app_name_com.mehequanna.gestureplayground',
    'app_name_com.talkatone.android', 'app_name_com.touchlogger',
    'origin_first_finger_False', 'origin_first_finger_True'
]

# Function to preprocess the input data
def preprocess_input(input_data):
    # Fill missing values in app_name as done during training
    input_data['app_name'].fillna('Other', inplace=True)
    
    # Apply dummy encoding for categorical variables
    df_encoded = pd.get_dummies(input_data, columns=['tap', 'app_name', 'origin_first_finger'])
    
    # Reorder columns to match training data
    df_encoded = df_encoded.reindex(columns=X_train_columns, fill_value=0)
    
    # Standardize numerical features
    numerical_features = ['pressure_first_finger', 'size_first_finger', 'x_coordinate_first_finger',
                          'y_coordinate_first_finger', 'distance']
    df_encoded[numerical_features] = scaler.fit_transform(df_encoded[numerical_features])
    
    # Convert the DataFrame to a NumPy array
    X_scaled = df_encoded.values
    return X_scaled

# Function to make predictions
def make_predictions(model, X):
    return model.predict(X)

# Custom CSS for styling
st.markdown("""
    <style>
        .main-title {
            color: #4CAF50;
            font-family: 'Courier New', Courier, monospace;
            text-align: center;
            font-size: 40px;
            margin-bottom: 20px;
        }
        .sub-title {
            color: #555555;
            font-family: 'Arial', sans-serif;
            text-align: center;
            font-size: 20px;
            margin-bottom: 20px;
        }
        .input-container {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 10px;
        }
        .result-message {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
        }
        .result-message.kid {
            color: #FF4500;
        }
        .result-message.parent {
            color: #1E90FF;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.markdown('<h1 class="main-title">Gesture Recognition App</h1>', unsafe_allow_html=True)

# Subtitle
st.markdown('<h2 class="sub-title">Enter Data</h2>', unsafe_allow_html=True)

# Input fields for user to input data
st.markdown('<div class="input-container">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    tap = st.selectbox('Tap', ['LongPress', 'Scroll', 'Swipe', 'Tap', 'TwoFinger', 'Unidentified'])
    origin_first_finger = st.selectbox('Origin First Finger', [True, False])
    pressure_first_finger = st.number_input('Pressure First Finger')
    size_first_finger = st.number_input('Size First Finger')
with col2:
    app_name = st.text_input('App Name')
    x_coordinate_first_finger = st.number_input('X Coordinate First Finger')
    y_coordinate_first_finger = st.number_input('Y Coordinate First Finger')
    distance = st.number_input('Distance')
st.markdown('</div>', unsafe_allow_html=True)

# Create a DataFrame from the input data
input_data = {
    'tap': [tap],
    'app_name': [app_name],
    'origin_first_finger': [origin_first_finger],
    'pressure_first_finger': [pressure_first_finger],
    'size_first_finger': [size_first_finger],
    'x_coordinate_first_finger': [x_coordinate_first_finger],
    'y_coordinate_first_finger': [y_coordinate_first_finger],
    'distance': [distance]
}
input_df = pd.DataFrame(input_data)

# Preprocess the input data
X_input = preprocess_input(input_df)

# Button to make predictions
if st.button('Make Prediction'):
    # Make predictions
    prediction = make_predictions(model, X_input)
    
    # Display prediction result with color
    if prediction[0] == 1:
        st.markdown("<div class='result-message kid'>A kid is using the phone.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-message parent'>The parent is using the phone.</div>", unsafe_allow_html=True)
