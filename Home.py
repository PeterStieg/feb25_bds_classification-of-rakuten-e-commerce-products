# Home.py
import streamlit as st

st.set_page_config(
    page_title="FEB25 BDS // Rakuten: Classification of eCommmerce products",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.title("FEB25 BDS // Rakuten")
st.sidebar.image("images/logos/rakuten-logo-red-wide.svg", use_container_width=True)

# Home page content
st.write("## eCommerce Products Classification Project")
st.write("Use the sidebar to navigate through the steps of our project.")
