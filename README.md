# CIFAR-10 Image Classification with PyTorch

This project explores image classification using the CIFAR-10 dataset with PyTorch. It includes exploratory data analysis (EDA), a baseline convolutional neural network (CNN), improvements using regularization and data augmentation, and a transfer learning approach with a pretrained model.

## 📁 Dataset

- **CIFAR-10**: 60,000 color images of size 32×32 across 10 classes
- **Train:** 50,000 images
- **Test:** 10,000 images
- Classes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`  
- Source: [CIFAR-10 Official Website](https://www.cs.toronto.edu/~kriz/cifar.html)

## 🚀 GPU Support
This project now supports NVIDIA GPU acceleration via PyTorch.  
- Verified on **NVIDIA GeForce GTX 1650** with CUDA 12.1
- Automatically detects GPU if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## 🧠 Project Structure

- `notebooks/01_EDA.ipynb` : Exploratory Data Analysis
-`notebooks/02_base_model.ipynb` : Baseline CNN model training and evaluation using PyTorch (GPU-enabled).

