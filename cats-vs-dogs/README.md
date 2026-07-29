# Cats vs Dogs Image Classification

A complete, beginner-friendly, end-to-end Deep Learning project using PyTorch.
Classifies images of cats and dogs using a Custom CNN and Transfer Learning
(ResNet-18, EfficientNet-B0). Includes full evaluation, visualisations,
and a Streamlit deployment app.

---

## Business Problem

Automated animal recognition is a foundational Computer Vision task used in
pet management apps, shelter systems, and wildlife monitoring. This project
builds and evaluates multiple classification models, demonstrating how
Transfer Learning dramatically outperforms a model trained from scratch on
a limited dataset.

---

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle — `bhavikjikadara/dog-and-cat-classification-dataset` |
| Classes | Cat, Dog |
| Total Images | ~25,000 |
| Split | 70% Train / 15% Val / 15% Test |

The dataset is downloaded automatically via the Kaggle API. No manual
download is required.

---

## Project Structure

```
cats-vs-dogs/
├── app/
│   └── app.py              # Streamlit web application
├── data/
│   ├── train/              # Training images (Cat/, Dog/)
│   ├── val/                # Validation images
│   └── test/               # Test images
├── models/
│   ├── custom_cnn_best.pth
│   ├── resnet18_best.pth
│   └── efficientnet_b0_best.pth
├── notebooks/
├── reports/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── src/
│   ├── __init__.py
│   ├── config.py           # All hyperparameters and paths
│   ├── dataset.py          # Download, cleaning, split, Dataset, DataLoaders
│   ├── model.py            # CustomCNN, ResNet18, EfficientNet-B0
│   ├── train.py            # Training loop, early stopping, mixed precision
│   ├── evaluate.py         # Metrics, confusion matrix, ROC, comparison table
│   ├── predict.py          # Single-image and batch prediction
│   └── utils.py            # Seed, device, save/load, plots
├── images/
├── main.ipynb              # Full end-to-end notebook (28 steps)
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/prasannasunuwar/cats-vs-dogs.git
cd cats-vs-dogs
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Kaggle API Setup

The dataset is downloaded automatically. You only need to provide your
Kaggle API credentials once.

### Step 1: Generate your API token

1. Log in to [kaggle.com](https://www.kaggle.com)
2. Go to **Account → API → Create New Token**
3. A file named `kaggle.json` is downloaded

### Step 2: Place the token

```bash
# Linux / macOS
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows (PowerShell)
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle"
Copy-Item kaggle.json "$env:USERPROFILE\.kaggle\kaggle.json"
```

### Step 3: Verify

```bash
kaggle datasets list
```

The dataset is then downloaded automatically when you run the notebook or
call `download_dataset()` from `src/dataset.py`.

---

## Usage

### Option A: Run the full notebook

```bash
jupyter notebook main.ipynb
```

The notebook walks through all 28 steps from data download to deployment.

### Option B: Run scripts directly

```python
from src.dataset import download_dataset, clean_corrupt_images, split_dataset, get_dataloaders
from src.model import get_model, print_model_summary
from src.train import train_model
from src.evaluate import evaluate_model, print_comparison_table
from src.utils import set_seed

set_seed(42)

# Download and prepare data
raw_path = download_dataset()
clean_corrupt_images(raw_path)
split_dataset(raw_path)
train_loader, val_loader, test_loader = get_dataloaders()

# Train all three models
results = {}
for model_name in ["custom_cnn", "resnet18", "efficientnet_b0"]:
    model   = get_model(model_name)
    history = train_model(model, model_name, train_loader, val_loader)
    results[model_name] = evaluate_model(model, model_name, test_loader)

print_comparison_table(results)
```

### Option C: Predict a single image

```python
from src.predict import load_and_predict

label, confidence = load_and_predict(
    image_path="path/to/your/image.jpg",
    model_name="resnet18",
)
print(f"Prediction: {label}  ({confidence:.1%})")
```

---

## Run the Streamlit App

```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501`.

---

## Model Architecture

### Custom CNN

```
Input (3, 224, 224)
  -> Conv2d(3, 32)  -> BN -> ReLU -> MaxPool -> Dropout
  -> Conv2d(32, 64) -> BN -> ReLU -> MaxPool -> Dropout
  -> Conv2d(64,128) -> BN -> ReLU -> MaxPool -> Dropout
  -> Conv2d(128,256)-> BN -> ReLU -> MaxPool -> Dropout
  -> GlobalAvgPool
  -> Linear(256, 512) -> ReLU -> Dropout(0.5)
  -> Linear(512, 2)
```

### ResNet-18 (Transfer Learning)

Pre-trained on ImageNet. Final `fc` layer replaced with:
`Dropout(0.3) -> Linear(512, 2)`

### EfficientNet-B0 (Transfer Learning)

Pre-trained on ImageNet. Classifier replaced with:
`Dropout(0.3) -> Linear(1280, 2)`

---

## Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 224 x 224 |
| Batch Size | 32 |
| Epochs (max) | 30 |
| Learning Rate | 1e-3 |
| Optimiser | Adam (weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Early Stopping | patience=7 |
| Mixed Precision | FP16 (when CUDA GPU is available) |
| Random Seed | 42 |

---

## Results

> Results are populated after training. Run `main.ipynb` to fill these in.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Custom CNN | TODO | TODO | TODO | TODO | TODO |
| ResNet-18 | TODO | TODO | TODO | TODO | TODO |
| EfficientNet-B0 | TODO | TODO | TODO | TODO | TODO |

---

## Data Augmentation

| Transform | Reason |
|---|---|
| Resize (224x224) | Standardise input dimensions |
| RandomHorizontalFlip | Animals face either direction |
| RandomRotation (15°) | Photos are not always perfectly level |
| RandomResizedCrop | Simulates zoom and different framing |
| ColorJitter | Handles varying lighting and camera settings |
| Normalize (ImageNet) | Required for pre-trained model compatibility |

---

## Technologies Used

| Library | Purpose |
|---|---|
| PyTorch | Model building and training |
| torchvision | Pre-trained models and transforms |
| kagglehub | Automated dataset download |
| scikit-learn | Metrics (accuracy, F1, ROC-AUC) |
| Matplotlib / Seaborn | Visualisations |
| Streamlit | Web deployment |
| tqdm | Progress bars |
| torchinfo | Model summary |
| Pillow | Image loading |

---

## Future Improvements

- Add cross-validation for more robust evaluation
- Experiment with data-efficient architectures (MobileNetV3, ViT)
- Implement Grad-CAM for model explainability
- Add REST API endpoint using FastAPI
- Deploy to Hugging Face Spaces or Streamlit Cloud
- Extend to multi-class pet classification

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Author

**Prasanna Sunuwar**
BIT Graduate | AI/ML Intern at EPF, Nepal
Aspiring Data Scientist and Machine Learning Engineer

- GitHub: [github.com/prasannasunuwar](https://github.com/prasannasunuwar)
- LinkedIn: [linkedin.com/in/prasannasunuwar](https://linkedin.com/in/prasannasunuwar)
