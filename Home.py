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
st.markdown(
    "This is a Streamlit app for the FEB25 BDS project on **eCommerce product classification**.  \nThe goal of this project is to classify products from the **Rakuten dataset** into different categories."
)

# Create navigation dropdown
page_selected = st.selectbox(
    "Use the sidebar or drowdown menu to navigate through the steps of our project:",
    [
        "Home",
        "1. Project Overview",
        "2. Team Presentation",
        "3. Data Exploration",
        "4. Data Visualization",
        "5. Preprocessing",
        "6. Modelling",
        "7. Prediction",
        "8. Thank you",
    ],
    key="navigation_dropdown",
)

# Handle navigation
if page_selected == "1. Project Overview":
    st.switch_page("pages/1_Project_Overview.py")
elif page_selected == "2. Team Presentation":
    st.switch_page("pages/2_Team_Presentation.py")
elif page_selected == "3. Data Exploration":
    st.switch_page("pages/3_Data_Exploration.py")
elif page_selected == "4. Data Visualization":
    st.switch_page("pages/4_Data_Visualization.py")
elif page_selected == "5. Preprocessing":
    st.switch_page("pages/5_Preprocessing.py")
elif page_selected == "6. Modelling":
    st.switch_page("pages/6_Modelling.py")
elif page_selected == "7. Prediction":
    st.switch_page("pages/7_Prediction.py")
elif page_selected == "8. Thank you":
    st.switch_page("pages/8_Thank_you.py")
