"""
app/app.py
==========
Streamlit web application for Cats vs Dogs inference.

Run with:
    streamlit run app/app.py

Features:
    - Upload any image (jpg, jpeg, png)
    - Preview the uploaded image
    - Predict Cat or Dog with confidence score
    - Display a probability bar chart
    - Clean, professional UI
"""

from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent


def build_model(num_classes: int = 2) -> nn.Module:
    """Build a ResNet-18 model with a replaced final FC layer."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASS_NAMES = ["Cat", "Dog"]
IMAGE_SIZE  = 224
MODELS_DIR  = ROOT / "models"

# Paths to look for trained model (in order of preference)
MODEL_CANDIDATES = [
    MODELS_DIR / "cats_vs_dogs_resnet18",   # torch.save folder format
    ROOT / "best_model.pth",                 # flat .pth checkpoint
    MODELS_DIR / "best_model.pth",
]

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cats vs Dogs Classifier",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Load model (cached so it is only loaded once)
# ---------------------------------------------------------------------------

def _load_folder_checkpoint(folder: Path, device) -> dict:
    """
    Load a state_dict from PyTorch's folder-based checkpoint format.
    This format is produced when torch.save(state_dict, <directory>) is
    called on PyTorch >= 2.x and the path is a directory.
    """
    import pickle

    def persistent_load(saved_id):
        _, dtype, key, _, numel = saved_id
        dtype_obj = getattr(torch, dtype) if isinstance(dtype, str) else dtype.dtype
        nbytes = numel * torch.empty(0, dtype=dtype_obj).element_size()
        storage = torch.UntypedStorage.from_file(
            str(folder / "data" / str(key)), False, nbytes
        )
        return torch.storage.TypedStorage(
            wrap_storage=storage, dtype=dtype_obj, _internal=True
        )

    with open(folder / "data.pkl", "rb") as f:
        unpickler = pickle.Unpickler(f)
        unpickler.persistent_load = persistent_load
        return unpickler.load()


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the best available trained model and cache it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find first available checkpoint
    model_path = None
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        return None, None, device

    model = build_model(num_classes=2)

    # Handle both folder-format (directory) and plain .pth checkpoints
    if model_path.is_dir():
        sd = _load_folder_checkpoint(model_path, device)
    else:
        state = torch.load(model_path, map_location=device, weights_only=False)
        sd = (
            state.get("model_state_dict")
            or state.get("state_dict")
            or state
            if isinstance(state, dict)
            else state.state_dict()
        )

    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, model_path, device


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Configuration")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model:** ResNet-18 (Transfer Learning)\n\n"
    "**About**\n\n"
    "This app uses a PyTorch deep learning model to classify "
    "images as either a **Cat** or a **Dog**.\n\n"
    "Built by Prasanna Sunuwar"
)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

model, model_path, device = load_model()

if model is None:
    st.error(
        "No trained model checkpoint found.\n\n"
        "Expected one of:\n"
        "- `models/cats_vs_dogs_resnet18/`\n"
        "- `best_model.pth`\n\n"
        "Run the training notebook (`main.ipynb`) first to produce a checkpoint."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

st.title("Cats vs Dogs Image Classifier")
st.markdown(
    "Upload a photo of a cat or a dog. "
    "The model will predict the class and show the confidence score."
)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Supported formats: JPG, JPEG, PNG, BMP, WEBP",
)

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Uploaded Image")
        st.image(img_pil, use_column_width=True)

    # Run inference
    with st.spinner("Classifying..."):
        img_tensor = INFER_TRANSFORM(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor)
            proba  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx        = int(np.argmax(proba))
        predicted_class = CLASS_NAMES[pred_idx]
        confidence      = float(proba[pred_idx])

    with col2:
        st.subheader("Prediction")

        badge_color = "#FF5722" if predicted_class == "Dog" else "#2196F3"

        st.markdown(
            f"""
            <div style="
                background-color:{badge_color};
                color:white;
                padding:20px;
                border-radius:12px;
                text-align:center;
                font-size:28px;
                font-weight:bold;
            ">
                {predicted_class}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**Confidence:** `{confidence:.1%}`")
        st.progress(float(confidence))

        st.markdown("---")
        st.subheader("Class Probabilities")
        for cls, prob in zip(CLASS_NAMES, proba):
            st.markdown(f"**{cls}**")
            st.progress(float(prob))
            st.caption(f"{prob:.1%}")

    st.markdown("---")

    with st.expander("Model Details"):
        st.markdown(
            f"- **Model:** ResNet-18 (Transfer Learning)\n"
            f"- **Input Size:** {IMAGE_SIZE} × {IMAGE_SIZE} px\n"
            f"- **Device:** `{device}`\n"
            f"- **Checkpoint:** `{model_path.name}`"
        )

else:
    st.info(
        "Upload an image above to get started. "
        "The classifier works best with clear, well-lit photos."
    )
    st.markdown(
        "**Tips for best results:**\n"
        "- Use a clear, well-lit photo\n"
        "- The animal should be the main subject\n"
        "- Works with any common image format (JPG, PNG, etc.)"
    )
