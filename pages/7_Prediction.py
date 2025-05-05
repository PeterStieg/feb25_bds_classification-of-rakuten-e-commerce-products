# Prepare image for prediction using identical preprocessing pipeline of training database


transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


cat_10 = "image_36900930_product_1178410_vgg16.jpg"
cat_40 = "image_857276_product_1170843_vgg16.jpg"
cat_50 = "image_862362663_product_102075783_vgg16.jpg"
cat_60 = "image_1100250764_product_247653336_vgg16.jpg"
cat_1140 = "image_1054649717_product_1046145679_vgg16.jpg"


# Load and preprocess an image
image = Image.open(PROJECT_FOLDER + "images/image_prediction/" + cat_1140)
plt.imshow(image)

image_tensor = transform(image).unsqueeze(0)  # Add batch dimension


# Make prediction
with torch.no_grad():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_ft.to(device)
    image_tensor = image_tensor.to(device)

    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)

print(f"Predicted class: {predicted.item()}")


# Prediction
import streamlit as st

st.set_page_config(
    page_title="Prediction",
    page_icon="images/logos/rakuten-favicon.ico",
    layout="wide",
)

st.title("Prediction")
st.sidebar.header("Prediction")

st.write("Welcome to the Prediction page!")

# Example section
st.header("Section 1")
st.write("This is a sample section for our prediction content.")

# Example prediction input section
st.header("Make Predictions")
st.write("Add input fields for your model predictions here")

# Example prediction output
st.header("Prediction Results")
if st.button("Generate Prediction"):
    st.write("Your prediction results will appear here")
