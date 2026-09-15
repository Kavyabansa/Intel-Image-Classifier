# 🏞️ Intel Image Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://intel-image-classifier-jknz4vzwzemcz2set63smb.streamlit.app/)

A deep learning image classification application built using **PyTorch, Torchvision, EfficientNet-B0, and Streamlit**.

The application classifies uploaded images into six different scene categories:

**Buildings • Forest • Glacier • Mountain • Sea • Street**

---

## 🚀 Live Demo

👉 **[Try the Intel Image Classifier](https://intel-image-classifier-jknz4vzwzemcz2set63smb.streamlit.app/)**

Upload an image and the trained deep learning model will predict the most likely scene along with prediction probabilities.

---

## ✨ Features

- 🖼️ Upload an image directly through the web interface
- 🤖 Deep learning based image classification
- 🧠 EfficientNet-B0 architecture
- 📊 Prediction probabilities for all six classes
- ⚡ Fast image inference
- 🌐 Deployed using Streamlit Community Cloud
- 🐍 Built using Python and PyTorch

---

## 🎯 Supported Classes

| Class | Description |
|---|---|
| 🏢 Buildings | Buildings and urban structures |
| 🌲 Forest | Forest and wooded landscapes |
| 🧊 Glacier | Glacier and snowy landscapes |
| ⛰️ Mountain | Mountain environments |
| 🌊 Sea | Sea, ocean and coastal scenes |
| 🛣️ Street | Roads and street environments |

---

## 🧠 Model

This project uses **EfficientNet-B0** with pretrained ImageNet weights.

The original classification layer was replaced with a custom classification layer containing **6 output classes**.

### Model Architecture

```text
Input Image
     ↓
Image Preprocessing
     ↓
EfficientNet-B0
     ↓
Feature Extraction
     ↓
Custom Linear Classifier
     ↓
6-Class Prediction
     ↓
Softmax Probabilities
