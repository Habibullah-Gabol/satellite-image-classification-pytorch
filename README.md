# TerraVision AI — Satellite Image Classification

An AI-powered satellite image classification application built with **PyTorch, ResNet18, and Streamlit**. Upload a satellite image to predict its land-use category and view the model's top predictions with confidence scores.

## Features

- **Deep Learning:** Uses a ResNet18 convolutional neural network.
- **Transfer Learning:** Fine-tuned model for satellite image classification.
- **10 Land-Use Classes:** Classifies satellite images into ten categories.
- **Top-3 Predictions:** Displays the three most likely classes with confidence scores.
- **Interactive Web Interface:** Upload and classify images through a Streamlit application.
- **Image Preprocessing:** Resizes images and applies normalization before inference.

## Technologies Used

- Python
- PyTorch
- Torchvision
- ResNet18
- Streamlit
- Pillow
- EuroSAT RGB Dataset

## Supported Classes

The model predicts one of the following categories:

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

## Project Structure

```text
satellite-image-classification-pytorch/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── models/
    └── best_resnet18_finetuned.pth
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Habibullah-Gabol/satellite-image-classification-pytorch.git
cd satellite-image-classification-pytorch
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Application

```bash
python -m streamlit run app.py
```

Streamlit will open the application in your browser.

## How to Use

1. Launch the TerraVision AI application.
2. Upload a satellite image in a supported format, such as JPG, JPEG, PNG, or WEBP.
3. Let the model process the image.
4. View the predicted land-use category and its confidence score.
5. Explore the top-three predictions and other available application sections.

## Model Information

- **Architecture:** ResNet18
- **Framework:** PyTorch
- **Task:** Multiclass image classification
- **Number of Classes:** 10
- **Input Size:** 224 × 224 pixels
- **Preprocessing:** ImageNet mean and standard deviation normalization
- **Dataset:** EuroSAT RGB

The final classification layer is adapted to predict ten satellite image categories. The trained model weights are loaded from `models/best_resnet18_finetuned.pth`.

## Model Performance

Add your final evaluation results here after confirming them from your evaluation notebook.

| Metric | Result |
|---|---|
| Test Accuracy | 97% |
| Macro F1-Score | 97% |

Use results from your final test evaluation rather than training accuracy.

## Important Notes

- The trained model checkpoint must be present at `models/best_resnet18_finetuned.pth`.
- The EuroSAT dataset is not required to run inference if the trained checkpoint is already available.
- The repository should include the model checkpoint or provide instructions for obtaining it if the file is too large for GitHub.
- Predictions are model estimates and may be incorrect for unfamiliar images or images outside the training distribution.

## Future Improvements

- Deploy the application online.
- Add Grad-CAM visualizations to explain model predictions.
- Evaluate performance on satellite images from different sources.
- Optimize inference speed and model size.
- Add more land-use categories and datasets.

## Learning Outcomes

This project demonstrates practical experience with:

- Deep learning and convolutional neural networks
- PyTorch model loading and inference
- Transfer learning with ResNet18
- Image preprocessing and classification
- Building an interactive AI application with Streamlit
- Organizing a deployable machine learning project

## License

Choose an appropriate open-source license before distributing this project. Also verify the dataset's licensing and attribution requirements before redistribution.

---

**TerraVision AI** — Turning satellite imagery into meaningful predictions with deep learning.
