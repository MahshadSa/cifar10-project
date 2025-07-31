# CIFAR-10 Image Classification with PyTorch

This project explores image classification using the CIFAR-10 dataset with PyTorch.  
It includes exploratory data analysis (EDA), a baseline convolutional neural network (CNN),  
improvements using regularization and data augmentation, and a transfer learning approach with a pretrained model.

---

## 📁 Dataset

- **CIFAR-10**: 60,000 color images (32×32) across 10 classes:  
  `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`  
- **Split**:  
  - **Train:** 50,000 images  
  - **Test:** 10,000 images  
- **Source:** [CIFAR-10 Official Website](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## 📊 Evaluation Metrics & Visualizations

- **Metrics**: Accuracy, Precision, Recall, F1-score (macro-averaged)  
- **Visualizations**:
  - Training and validation loss curves
  - Validation accuracy curve
  - Confusion matrix (class-wise performance)

---

## ✨ Features Implemented

1. **Training Pipeline**
   - Custom CNN model
   - CrossEntropyLoss + Adam optimizer (with weight decay)
   - Per-epoch loss & accuracy logging
   - Best model checkpointing (`best_model.pth`)

2. **Validation & Evaluation**
   - Macro-averaged Precision, Recall, and Accuracy
   - Class-wise precision, recall, and F1-score
   - Confusion matrix visualization

3. **Visualizations**
   - Epoch-based loss and accuracy plots
   - Confusion matrix (normalized or absolute counts)

4. **Extensibility**
   - Easy integration of data augmentation
   - Easy switching to pretrained models (e.g., ResNet18)

---

## 🧠 Project Structure

```bash
cifar10-project/
│
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory Data Analysis
│   ├── 02_base_model.ipynb       # Baseline CNN training and evaluation (GPU-enabled)
│   └── 03_improved_model.ipynb   # Enhanced CNN with improved logging, checkpointing, and visualizations
│
├── train.py                      # Training script
├── evaluate.py                   # Evaluation script
├── visualize.py                  # Visualization utilities
├── models.py                     # Model definitions
├── utils.py                      # Helper functions (e.g., metrics, transforms)
└── requirements.txt              # Dependencies


## 🛠 Usage
```bash
git clone <your_repo_url>
cd cifar10-project
pip install -r requirements.txt
python train.py
python evaluate.py
