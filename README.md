# Multi-Label Defect Prediction

This project predicts multiple defect types from software defect reports using multi-label classification, with a Streamlit app for real-time predictions.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Training](#training)
- [References](#references)

## Overview
Using **TF-IDF** features, we train **Logistic Regression**, **SVM**, **Perceptron**, and **DNN** models on a dataset (`defect_prediction_dataset.csv`). The DNN excels with a Micro-F1 of 0.85.

## Setup
1. **Clone the repo**:
   ```bash
   git clone https://github.com/Anas-Altaf/ML-Software-Quality-Checking.git
   cd ML-Software-Quality-Checking
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Add dataset**:
   Place `defect_prediction_dataset.csv` in the project directory.

## Usage
1. **Run the notebook**:
   Open `Multi_Label_Defect_Prediction.ipynb` in Jupyter for preprocessing and evaluation.
2. **Launch the app**:
   ```bash
   streamlit run app.py
   ```
3. **Interact**:
   - Input defect report text, select a model, and click "Predict Defects."

## Training
Models train automatically if weights are missing. Manually train via the notebook or app code.

## References
- [Scikit-learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/docs/)
- [Streamlit](https://docs.streamlit.io/)
- [Pandas](https://pandas.pydata.org/docs/)
