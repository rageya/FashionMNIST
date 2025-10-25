# 👗 FashionMNIST - Image Classification with PyTorch

A deep learning project that uses machine learning models (MLPs and CNNs) to classify clothing items from the FashionMNIST dataset. This project demonstrates baseline, non-linear, and convolutional architectures for robust grayscale image classification.

## 📋 Project Summary
End-to-end deep learning system for recognizing clothing types in the FashionMNIST 10-class dataset. Three models are built: a two-layer baseline, a non-linear ReLU variant, and a multi-layer CNN, achieving strong performance using PyTorch.

## 🏗️ Model Architectures
**Model 0 (Baseline MLP):**
- Flatten input (28x28=784)
- Two Linear Layers: [784 → 10 → 10]
- Accuracy: **83.43%**

**Model 1 (MLP with ReLU):**
- Flatten input (28x28=784)
- Linear → ReLU → Linear → ReLU
- Accuracy: **75.02%**

**Model 2 (Convolutional Neural Network):**
- Block 1: Conv2d (1→10), ReLU, Conv2d (10→10), ReLU, MaxPool2d
- Block 2: Conv2d (10→10), ReLU, Conv2d (10→10), ReLU, MaxPool2d
- Classifier: Flatten, Linear (490→10)
- Accuracy: **88.08%**

## 📊 Dataset
- Source: FashionMNIST (from torchvision.datasets)
- Training images: 60,000
- Test images: 10,000
- Image size: 28x28, grayscale
- Classes (10): T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

## 🎯 Results & Performance
| Model Name            | Test Accuracy | Loss     | Training Time (sec) |
|----------------------|--------------|----------|---------------------|
| FashionMNISTModelV0  |   83.43%     | 0.48     | 27.6                |
| FashionMNISTModelV1  |   75.02%     | 0.69     | 31.3                |
| FashionMNISTModelV2  |   88.08%     | 0.33     | 36.9                |

CNN model (Model 2) achieves highest accuracy.

## 🚀 Usage Steps
1. **Clone Repo & Install Dependencies**
   ```bash
   git clone <your-github-url>
   cd <project-folder>
   pip install torch torchvision matplotlib tqdm numpy pandas
   ```
2. **Prepare Data**
   ```python
   from torchvision import datasets, transforms
   train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
   test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
   ```
3. **Set up DataLoader**
   ```python
   from torch.utils.data import DataLoader
   train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
   test_loader  = DataLoader(test_data, batch_size=32, shuffle=False)
   ```

4. **Train Models**
   - For each architecture, set device, optimizer, loss, and run training loops (examples in notebook).

5. **Evaluate Models**
   - Use accuracy function from helper_functions.py to calculate performance on test set.

6. **Make Predictions**
   - Forward a test image batch through `.eval()` model and use argmax for predicted class.

## 📦 Requirements
- torch
- torchvision
- matplotlib
- tqdm
- numpy
- pandas

## 📁 Project Structure
```
fashion-mnist-pytorch/
├── data/                # FashionMNIST dataset files
├── helper_functions.py  # Metrics and utilities
├── notebook.ipynb       # Colab notebook
├── README.md            # Project overview
```

## 🔑 Key Features
- Three model architectures (MLP, ReLU, CNN)
- Fast training, easy reproducibility
- Results and comparisons table for all models
- Visualizations and confusion matrix included
- Modular code for extensibility

## 📈 Training Details
- Epochs: 3
- Batch size: 32
- Optimizer: SGD (lr=0.1)
- Loss: CrossEntropyLoss
- Baseline and CNN architectures compared

## 📄 License
MIT License – Free for research and educational use.

## 👤 Author
rageya

## 🙏 Acknowledgments
- FashionMNIST dataset creators
- PyTorch & torchvision
- Daniel Bourke ([helper_functions.py](https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py))

## 📝 Notes
- Models can be improved with more epochs, data augmentation, and deeper architectures.
- CNNs achieve the best accuracy on FashionMNIST.
- Notebook includes confusion matrix and prediction plots.
