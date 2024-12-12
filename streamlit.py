%%writefile app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

# Load the fine-tuned ResNet model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Check if CUDA is available
    model = models.resnet18(pretrained=False)  # Initialize ResNet18
    model.fc = torch.nn.Linear(model.fc.in_features, 4)  # Adjust for 4 classes
    model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=device))  # Map the model to the appropriate device
    model.to(device)  # Send the model to the correct device
    model.eval()  # Set the model to evaluation mode
    return model

# Initialize the model
model = load_model()

# Define image transformations (same as used in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



# Define class labels
class_names = ['glioma', 'healthy', 'meningioma', 'pituitary']

# Streamlit interface
st.title("Brain Tumor Classification with ResNet")
st.write("Upload an MRI image to classify the type of brain tumor.")


# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Transform the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)

    # Display the prediction
    st.write(f"Prediction: **{class_names[predicted_class.item()]}**")
