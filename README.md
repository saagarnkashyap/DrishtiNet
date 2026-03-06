# Concept-Aware Retinal Disease Classification using Explainable AI

A deep learning framework for **retinal disease classification** using fundus images with integrated **Explainable AI visualizations**.
The project focuses on building an interpretable medical imaging system that can assist in identifying ocular diseases while showing **where the model is focusing inside the retina**.

---

# Project Overview

Retinal diseases are a major cause of vision impairment worldwide. Automated detection systems using **deep learning** can assist ophthalmologists by providing faster preliminary diagnosis.

However, most deep learning models behave like **black boxes**.
To address this problem, this project integrates **Explainable AI techniques** to visualize which regions of the retinal image influence the prediction.

This project implements:

* Retinal image classification
* Model interpretability using Grad-CAM
* Visual overlays for clinical insight
* A pipeline suitable for research experimentation

---

# Dataset

This project uses the **ODIR-5K (Ocular Disease Intelligent Recognition)** dataset.

Dataset contains:

* ~5000 retinal fundus images
* Left and right eye images
* Multiple ocular disease labels

### Disease Categories

| Label | Disease                          |
| ----- | -------------------------------- |
| N     | Normal                           |
| D     | Diabetic Retinopathy             |
| G     | Glaucoma                         |
| C     | Cataract                         |
| A     | Age-related Macular Degeneration |
| H     | Hypertension                     |
| M     | Myopia                           |
| O     | Other abnormalities              |

Dataset link:

https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k

---

# Project Architecture

```
Retinal Image
      │
      ▼
Image Preprocessing
(resize, normalize)
      │
      ▼
Deep Learning Model
(CNN classifier)
      │
      ▼
Prediction
(retinal disease class)
      │
      ▼
Explainability Module
(Grad-CAM)
      │
      ▼
Visual Interpretation
(heatmap + overlay)
```

---

# Explainable AI

Medical AI systems require transparency for clinical adoption.
This project integrates **Grad-CAM** to visualize model attention.

Grad-CAM highlights regions that contribute the most to a prediction.

Example pipeline:

```
Input Retina Image
      ↓
Forward Pass
      ↓
Gradient Computation
      ↓
Feature Map Weighting
      ↓
Heatmap Generation
      ↓
Overlay on Original Image
```

Output visualization:

* Original retinal image
* Grad-CAM heatmap
* Overlay highlighting model focus

---

# Current Results

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | ~95%  |
| Validation Accuracy | ~50%  |

The model currently shows **overfitting**, which is common in medical imaging tasks due to limited dataset size. Future improvements will address this.

---

# Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Kaggle Notebooks

---

# Project Structure

```
retinal-disease-classification/

│
├── dataset/
│   └── ODIR-5K images
│
├── notebooks/
│   └── training_pipeline.ipynb
│
├── models/
│   └── trained_model.h5
│
├── outputs/
│   ├── predictions
│   └── gradcam_visualizations
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── gradcam.py
│
└── README.md
```

---

# Installation

Clone the repository:

```
git clone [https://github.com/yourusername/retinal-disease-classification](https://github.com/saagarnkashyap/DrishtiNet).git
```

Install dependencies:

```
pip install tensorflow
pip install opencv-python
pip install numpy pandas matplotlib
```

---

# Running the Project

1. Download the dataset from Kaggle.
2. Place the dataset in the `dataset/` directory.
3. Run the training notebook:

```
notebooks/training_pipeline.ipynb
```

The notebook performs:

* Data preprocessing
* Model training
* Evaluation
* Grad-CAM visualization

---

# Grad-CAM Visualization Example

The system generates interpretability visualizations showing the retinal regions responsible for predictions.

Output includes:

* Original image
* Heatmap
* Overlay visualization

These visualizations help verify that the model focuses on clinically relevant retinal structures.

---

# Research Contribution

This project explores **interpretable deep learning for medical imaging** by combining:

* Image classification
* Explainable AI
* Retinal disease detection

The system can serve as a research baseline for:

* medical AI interpretability
* cross-dataset generalization
* clinical decision support systems

---

# Future Work

Planned improvements include:

### Model Improvements

* Transfer learning
* Data augmentation
* Regularization techniques

### Explainability Enhancements

* SHAP explanations
* Feature importance visualization

### Novel Research Extensions

* Liquid Neural Networks for adaptive learning
* LLM-based medical explanation generation
* Cross-dataset domain adaptation

---

# Authors

Saagar N Kashyap,
B.Tech Computer Science Engineering

Rishi Jha,
B.Tech Computer Science Engineering

Vijageesh,
B.Tech Computer Science Engineering

---

# License

This project is intended for **academic research and educational purposes**.
