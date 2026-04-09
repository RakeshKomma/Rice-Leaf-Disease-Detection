# 🌾 Rice Leaf Disease Detection using Deep Learning

## 📌 Project Overview

This project is a **deep learning-based web application** that detects diseases in rice leaves using image classification. It identifies three major diseases:

* Bacterial Leaf Blight
* Brown Spot
* Leaf Smut

The model is built using **Transfer Learning (MobileNetV2)** and deployed using **Streamlit**.

---

## 🎯 Objectives

* Perform data analysis on rice leaf images
* Build a robust image classification model
* Apply data augmentation to handle small dataset
* Evaluate model performance using multiple metrics
* Deploy the model as a web application

---

## 📊 Dataset Details

* Total Images: 120
* Classes: 3
* Images per class: ~40

### Classes:

* Leaf smut
* Brown spot
* Bacterial leaf blight

---

## 🧠 Model Architecture

* Base Model: MobileNetV2 (pretrained on ImageNet)
* Transfer Learning: Frozen base layers
* Custom Layers:

  * Global Average Pooling
  * Dense (128 units, ReLU)
  * Dropout (0.5)
  * Output Layer (Softmax)

---

## 🔄 Techniques Used

* Data Augmentation:

  * Rotation
  * Zoom
  * Horizontal Flip
  * Brightness Adjustment

* Regularization:

  * Dropout
  * Batch Normalization

---

## 📈 Model Performance

* Accuracy: ~87%
* Balanced Precision, Recall, F1-score
* Strong generalization despite small dataset

---

## 🔍 Explainability (Grad-CAM)

Grad-CAM is used to visualize which parts of the image the model focuses on.

👉 The model correctly highlights infected regions of the leaf, ensuring reliable predictions.

---

## 🌐 Web Application (Streamlit)

Features:

* Upload rice leaf image
* Get disease prediction
* View confidence score
* Visualize Grad-CAM heatmap

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## Run the App

```bash
python -m streamlit run app.py
```

---

## 📁 Project Structure

```
rice-leaf-disease-app/
│
├── app.py
├── rice_disease_model.h5
├── requirements.txt
├── README.md
```

---

# Future Improvements

* Increase dataset size
* Use advanced models (EfficientNet, Vision Transformers)
* Improve accuracy with hyperparameter tuning
* Deploy mobile application

---

#Resume Highlight

> Developed a deep learning-based rice leaf disease detection system using MobileNetV2 with ~87% accuracy and deployed it as an interactive web application with Grad-CAM explainability.

---

## 📌 Author

Rakesh Komma
