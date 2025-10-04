import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Page config
st.set_page_config(page_title="Brain Tumor Classification", page_icon="🧠")
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI image to classify brain tumor type")

# Load model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 4)
    )
    model.load_state_dict(torch.load("../models/best_brain_tumor_resnet18_finetuned.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Labels
labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI Image', use_container_width=True)
    
    # Make prediction
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Display results
    st.subheader("Prediction Results")
    st.success(f"**Prediction:** {labels[predicted_class]}")
    st.info(f"**Confidence:** {confidence:.2%}")
    
    # Show all probabilities
    st.subheader("All Class Probabilities")
    for i, label in enumerate(labels):
        prob = probabilities[0][i].item()
        st.progress(prob, text=f"{label}: {prob:.2%}")