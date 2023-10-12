import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load data and models (you can put this in a separate file and import it)
df = pickle.load(open('watchesdf.pkl', 'rb'))
model = pickle.load(open('watchesmodel.pkl', 'rb'))
knn_imputer1 = pickle.load(open('knn_imputer1.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
ohe = pickle.load(open('watchesohe.pkl', 'rb'))

# Set page title and background color
st.set_page_config(
    page_title="Watch Price Predictor",
    page_icon="⌚",
    layout="wide",
)

# Custom CSS for styling
st.markdown(
    
    """<style>
    .stApp {
        background-color: #f5f5f5;
    }
    .st-eb {
        padding: 1rem;
        background-color: #fff;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    }
    .st-bm {
        margin-top: 2rem;
    }
    .st-fz {
        font-size: 18px;
    }
    </style>"""
    ,
    unsafe_allow_html=True,
)

# Add a title with custom styling
st.markdown("<h1 style='text-align: center; color: Black;'>Watch Price Predictor</h1>", unsafe_allow_html=True)

# Create a sidebar with user input widgets and custom styling
st.sidebar.markdown("<h2 style='text-align: center; color: Red;'>Enter Watch Details</h2>", unsafe_allow_html=True)

name = st.sidebar.selectbox("Name", df['model'].unique())
mvmt = st.sidebar.selectbox("Movement of Watch", df['mvmt'].unique())
casem = st.sidebar.selectbox("Material of Watch", df['casem'].unique())
bracem = st.sidebar.selectbox("Material of Bracelet", df['bracem'].unique())
year_of_production = st.sidebar.selectbox("Year of Production", df['yop'].unique())
cond = st.sidebar.selectbox("Condition of Watch", df['condition_of_watch'].unique())
sex = st.sidebar.selectbox("Sex of Watch", df['sex'].unique())
diameter = st.sidebar.number_input("Diameter of Watch")

# Create a function to predict the price
def predict_price():
    query = pd.DataFrame(
        {
            'model': [name],
            'mvmt': [mvmt],
            'casem': [casem],
            'bracem': [bracem],
            'yop': [year_of_production],
            'condition_of_watch': [cond],
            'sex': [sex],
            'diameter_mm': [diameter]
        }
    )
    categorical_columns = ['model', 'mvmt', 'casem', 'bracem', 'sex', 'condition_of_watch', 'yop']
    preprocessing = ohe.transform(query[categorical_columns]).toarray()

    a = knn_imputer1.transform(query[['diameter_mm']])
    new_df = np.concatenate((preprocessing, a), axis=1)
    new_df1 = pd.DataFrame(new_df)
    scaled_df = scaler.transform(new_df1)

    prediction = np.exp(model.predict(scaled_df))
    return int(prediction)

# Create a section for displaying the prediction with custom styling
st.sidebar.markdown("<h3 class='st-fz'>Predicted Price</h3>", unsafe_allow_html=True)
if st.sidebar.button("Predict Price"):
    predicted_price = predict_price()
    st.sidebar.markdown(f"<p style='text-align: center; font-size: 24px;'>The predicted price of the watch is: <strong>${predicted_price}</strong></p>", unsafe_allow_html=True)

# Improve the main content layout
st.markdown("<h2 style=color: Black>About Watch Price Predictor</h2>", unsafe_allow_html=True)
st.write("Welcome to the Watch Price Predictor! You can use this tool to estimate the price of a watch based on various attributes.")

st.markdown("<h3 style= color: Black>How it works</h3>", unsafe_allow_html=True)
st.write("1. Enter the details of the watch on the left sidebar.")
st.write("2. Click the 'Predict Price' button to get the estimated price on the sidebar.")

st.markdown("<h3 style= color:Black>Note</h3>", unsafe_allow_html=True)
st.write("The prediction is based on a machine learning model and may not be entirely accurate.")

# Add a footer with custom styling
st.sidebar.markdown("<hr class='st-bm'>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: Red;'>Made with ❤️ by Your Name</p>", unsafe_allow_html=True)
