Breast Cancer Classification 

Overview:

This project implements breast cancer classification using VGG16, DenseNet121, and a Custom CNN. The models analyze mammogram images to classify them as Benign or Malignant.

Getting Started:

## Installation & Setup  

### 1️	**Download and Unzip**  
You can download the repository as a ZIP file and extract it:  
- Click on the **"Code"** button (above the file list).  
- Select **"Download ZIP"**.  
- Extract the ZIP file on your local machine.  

### 2️	**Clone via Git**  
Alternatively, you can clone the repository using Git:  

```bash
git clone https://github.com/mehrshirzad/breast-cancer-classification.git


```bash
cd breast-cancer-classification

###3	**Download and Unzip the Dataset**  
- Download the dataset from: **https://www.kaggle.com/datasets/kmader/mias-mammography**.  
- Extract the dataset and place it in the `data/` directory: 

Usage:

1️. Install Dependencies
Ensure you have all necessary packages installed:

```bash
pip install -r requirements.txt

2️. Train a Model

To train a model, run:

```bash
python train.py --model vgg16

or for other models:

```bash
python train.py --model densenet

python train.py --model custom_cnn

3️. Evaluate a Model

To evaluate a trained model:

```bash
python evaluate.py --model vgg16

Supported models: vgg16, densenet, custom_cnn


Project Structure:

│-- models/                   # Model architectures
│   │-- vgg16_model.py         # VGG16 Model
│   │-- densenet_model.py      # DenseNet Model
│   │-- custom_cnn.py          # Custom CNN Model
│
│-- utils/                     # Utility functions
│   │-- data_preprocessing.py   # Image loading & preprocessing
│
│-- data/                      # Dataset 
│
│-- train.py                   # Training script
│-- evaluate.py                # Evaluation script
│-- requirements.txt            # Required dependencies
│-- README.txt                  # Project documentation

Technical Details:

🔹 Data Preprocessing
Parse Info.txt for image metadata.
Load mammogram images and extract Regions of Interest (ROIs).
Resize images to 128x128 and normalize pixel values.
Convert grayscale images to RGB (3-channel).

🔹 Models
✅ 1. VGG16
Uses pretrained VGG16 as the base model.
Fine-tuned on the breast cancer dataset.
✅ 2. DenseNet121
Deeper CNN with dense connections.
Suitable for feature-rich datasets.
✅ 3. Custom CNN
Built from scratch with:
Conv2D layers for feature extraction.
MaxPooling for downsampling.
Fully connected layers for classification.

Results:

Model	Accuracy	F1-Score
VGG16	X%	X.XX
DenseNet	X%	X.XX
Custom CNN	X%	X.XX


References
MIAS Dataset  (Mammographic Image Analysis Society)
TensorFlow & Keras for Deep Learning
Scikit-learn for evaluation metrics


