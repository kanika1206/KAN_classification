# 🌍 KAN-Based Satellite Image Classification  
Hybrid CNN + Kolmogorov–Arnold Networks (KAN) for EuroSAT Dataset

## 📌 Overview  
This project implements a hybrid deep learning architecture combining:

- Pre-trained CNN / Transformer backbones (VGG16, ConvNeXt, ViT)
- Kolmogorov–Arnold Network (KAN) layers as the classification head

KAN layers use spline-based functional mappings that outperform traditional linear layers in modeling nonlinear patterns.  
The goal is high-accuracy land use / land cover classification using the EuroSAT dataset.

---

## 📂 Project Structure

KAN_classification/
│── kan.py
│── kcn.py
│── main.py
│── models.py
│── train_and_test.py
│── train_test_split.py
│── results.txt
│── kcn.txt
│── PatchBasedClassification/


---

## 🛰 Dataset — EuroSAT  
Based on Sentinel-2 satellite images:

- 27,000 RGB images
- 64×64 resolution
- 10 land-use classes  
  (Forest, River, Residential, Industrial, Highway, etc.)

Folder format:
data/
├── train/
└── test/

---

## 🧠 Methodology

### **1️⃣ Feature Extraction Backbone**
Pre-trained models with frozen weights:

- VGG16  
- ConvNeXt Tiny  
- Vision Transformer (ViT)

These networks extract high-level spatial features.

---

### **2️⃣ KAN Classification Head**
KAN replaces the dense/linear layer with a learnable **functional mapping** using B-splines:

\[
y = \sum_i \phi_i(x_i)
\]

Advantages:
- Better nonlinear modeling  
- Strong generalization  
- Improved performance on satellite images  

---

### **3️⃣ Training**
- Loss: Cross Entropy Loss  
- Optimizer: Adam (lr=0.001)  
- Metrics:
  - Accuracy  
  - Precision, Recall, F1  
  - Confusion Matrix  

Data Augmentation:
- RandomResizedCrop  
- Horizontal flip  
- ImageNet normalization  

---

## 📊 Results

| Model | Backbone | Classifier | Accuracy |
|------|----------|------------|----------|
| VGG16 | CNN | Linear | 93.26% |
| VGG16 + KAN | CNN | KAN | 94.12% |
| ViT | Transformer | Linear | ~95% |
| ViT + KAN | Transformer | KAN | 96.27% |
| ConvNeXt | CNN | Linear | 96.21% |
| ConvNeXt + KAN | CNN | KAN | 96.62% |

KAN layers consistently improve model performance.

---

## 🔍 Confusion Matrix  
Used for analyzing misclassifications and class-wise performance.

---

## 🧪 How to Run

### **1️⃣ Install requirements**
pip install -r requirements.txt

### **2️⃣ Train the model**
python train_and_test.py
### **3️⃣ Evaluate**
python main.py

Outputs include accuracy, F1 score, confusion matrix, and saved models.

---

## 🚀 Future Work
- Add Swin-KAN models  
- Reduce GPU memory usage  
- Add pixelwise segmentation  
- Deploy as an inference API  

---
## 📚 References  
- Liu et al. (2024) — Kolmogorov–Arnold Networks  
- Helber et al. (2019) — EuroSAT Dataset  

---
