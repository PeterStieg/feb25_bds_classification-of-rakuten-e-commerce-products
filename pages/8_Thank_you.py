# Thank you
import streamlit as st

st.set_page_config(
    page_title="Thank You",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.title("Thank You")
st.sidebar.header("Thank You")

# Main content
st.markdown(
    """
## Thank You for Exploring Our Project!

We appreciate you taking the time to review our work on the classification of Rakuten e-commerce products. 
This project represents significant effort and collaboration from our entire team.

### Key Contributions

Our team has worked diligently to:
* Collect and process complex e-commerce data
* Develop robust classification models
* Create this interactive Streamlit application to showcase our work

### Feedback and Contact

Your feedback is valuable to us! If you have any questions, suggestions, or would like to discuss this project further,
please don't hesitate to reach out to our team members.

### Acknowledgements

We would like to extend our sincere gratitude to:
* Our instructors and mentors who guided us throughout this project
* Rakuten for providing the dataset that made this research possible
* The open-source community for the wonderful tools and libraries that powered our analysis
"""
)

# Add a decorative element - a simple progress completion bar
st.subheader("Project Exploration Progress")
st.progress(100)
st.success("You've completed the entire project tour! Thank you for your interest!")

# Optional: Add team contact information
st.markdown("---")
st.subheader("Connect With Us")

# You can customize this part with actual team information
col1, col2, col3 = st.columns(3)
with col1:
    st.info("Project Lead: [Team Member Name](mailto:email@example.com)")
with col2:
    st.info("Data Science: [Team Member Name](mailto:email@example.com)")
with col3:
    st.info("Development: [Team Member Name](mailto:email@example.com)")

# Final touch - a simple animation
st.balloons()
