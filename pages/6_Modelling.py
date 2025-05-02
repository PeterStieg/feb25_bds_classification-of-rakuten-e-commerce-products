# Modelling
import streamlit as st

st.set_page_config(
    page_title="Modelling",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.title("Modelling")
st.sidebar.header("Modelling")

# Add your modelling code here
st.write("Welcome to the Modelling page!")

# Example section
st.header("Section 1")
st.write("This is a sample section for our modelling content.")

# Example model section
st.header("Model Selection")
model_option = st.selectbox(
    "Select a model:",
    ("Linear Regression", "Random Forest", "XGBoost", "Neural Network"),
)

st.write("You selected:", model_option)

# Model parameters section
st.header("Model Parameters")
st.write("Add model parameter sliders and inputs here")
