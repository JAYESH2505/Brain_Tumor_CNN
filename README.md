# Brain Tumor Detection Using CNN and PyTorch

This repository contains the implementation of a deep learning-based approach for detecting brain tumors from MRI scans. Using Convolutional Neural Networks (CNNs) and PyTorch, the model classifies MRI images into one of four categories: `meningioma`, `pituitary`, `healthy`, or `glioma`.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
Brain tumors can be life-threatening and require early detection for effective treatment. This project leverages a CNN-based architecture to identify brain tumors from MRI images. The model was trained on the Kaggle dataset *Brain Tumor MRI Scans*, which includes four categories of scans.

## Dataset
The dataset used for this project is available on Kaggle: [Brain Tumor MRI Scans]([https://www.kaggle.com/datasets](https://www.kaggle.com/datasets/rm1000/brain-tumor-mri-scans)). 

**Categories:**
- Meningioma
- Pituitary
- Healthy
- Glioma

**Structure:** Ensure your dataset is structured as follows for training:
```
/dataset/
    /train/
        /meningioma/
        /pituitary/
        /healthy/
        /glioma/
    /val/
        /meningioma/
        /pituitary/
        /healthy/
        /glioma/
    /test/
        /meningioma/
        /pituitary/
        /healthy/
        /glioma/
```

---

## Features
- **Preprocessing:** Image normalization and augmentation.
- **Model:** Custom CNN architecture using PyTorch.
- **Metrics:** Accuracy, Precision, Recall, and F1-Score.
- **Visualization:** Training/Validation loss curves and predictions.

---

## Installation

### Clone Repository
```bash
git clone https://github.com/<your-username>/brain-tumor-detection.git
cd brain-tumor-detection
```

### Install Dependencies
Create a virtual environment and install required packages:
```bash
python -m venv venv
source venv/bin/activate # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Required Packages
All required Python packages are listed in `requirements.txt`. Here are the key dependencies:
```
numpy
pandas
matplotlib
seaborn
scikit-learn
torch
torchvision
tqdm
```

---

## Usage

### Training the Model
Train the model using the following command:
```bash
python train.py --data_dir ./dataset --epochs 20 --batch_size 32 --lr 0.001
```

### Evaluating the Model
Evaluate the trained model on the test dataset:
```bash
python evaluate.py --data_dir ./dataset --model_path ./saved_model.pth
```

### Visualizing Results
Generate confusion matrices and training curves using:
```bash
python visualize.py --model_path ./saved_model.pth --data_dir ./dataset
```

---

## Project Structure
```
brain-tumor-detection/
│
├── dataset/                 # Dataset folder (not included in the repo)
├── models/                  # Saved models
├── notebooks/               # Jupyter Notebooks for experiments
├── scripts/                 # Python scripts for training and evaluation
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   ├── visualize.py         # Visualization script
├── utils/                   # Utility functions
├── README.md                # Project README
├── requirements.txt         # Required dependencies
├── LICENSE                  # License file
└── .gitignore               # Git ignore file
```

---

## Results
Include a summary of your model's performance:
- Training Accuracy: **97.2%**
- Validation Accuracy: **98.1**
- Test Accuracy: **97.6%**

### Sample Predictions
Provide some visual examples of the model's predictions.

---

## Contributing
Contributions are welcome! If you find a bug or have a feature request, please open an issue or submit a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
