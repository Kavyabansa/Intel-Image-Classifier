import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

st.set_page_config(page_title="Intel Image Classifier", page_icon="🌄")

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MODEL_PATH = Path("intel_efficientnet_v2.pth")
DEVICE = torch.device("cpu")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 6)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

st.title("🌄 Intel Image Classifier")
st.write(
    "Upload an image and the EfficientNet-B0 model will predict whether "
    "it is a buildings, forest, glacier, mountain, sea, or street scene."
)

model = load_model()

if model is None:
    st.error(
        "Model file not found. Add `intel_efficientnet_v2.pth` to the root of "
        "the GitHub repository and redeploy."
    )
    st.stop()

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    pred = int(output.argmax(dim=1).item())

    st.subheader(f"Prediction: {CLASSES[pred]}")
    st.write(f"Confidence: {float(probs[pred]) * 100:.2f}%")

    results = {
        CLASSES[i]: float(probs[i])
        for i in range(len(CLASSES))
    }
    st.bar_chart(results)
