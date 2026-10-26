# GAN-Based Synthetic Data Generation for Imbalanced Datasets

A Web-Based Tool for Class Distribution Analysis and GAN-Assisted Tabular Data Balancing.

**Project Proposal Team:**
- DJOHAN CHERUSSERIL (URK24CS7147)
- DEVANAND M V (URK24CS7111)
- LAYO BAIJU (URK24CS7128)

---

## 🚀 Overview

Class imbalance in classification problems causes supervised machine learning algorithms to heavily bias toward the majority class, leading to severe failure in identifying rare minority events (e.g. fraudulent transactions, medical anomalies, disease diagnosis).

This project implements a web-based application using **Streamlit** and a **PyTorch Generative Adversarial Network (GAN)** specifically engineered for tabular data. It allows users to:
1. **Upload any tabular dataset (CSV)** or test immediately with built-in benchmarks.
2. Select from **3 primary user actions**:
   - **Option 1: Check if the dataset is balanced**: Computes Imbalance Ratio ($IR = N_{maj} / N_{min}$), minority/majority percentages, and clearly renders a diagnostic verdict (Balanced vs Imbalanced).
   - **Option 2: Balance using PyTorch GAN**: If imbalanced, trains a custom Generator-Discriminator network on minority-class feature distributions to generate realistic synthetic samples, bringing the dataset to full 50:50 parity or custom target.
   - **Option 3: Leave As-Is**: Preserves the original natural distribution without modification, providing baseline classifier metrics.
3. **Rigorous Evaluation & Fair Validation**:
   - Compares downstream ML classifier performance (**Random Forest / Logistic Regression**) trained on *Original Imbalanced Data* vs. *GAN-Balanced Data* evaluated on the **exact same untouched test set** (preventing data leakage).
   - Shows metrics: Minority Recall, Minority F1-Score, Minority Precision, Accuracy, and Confusion Matrices.
   - Diversity and validity checks (calculating exact overlap with real training data to ensure generalization without memorization).
4. **Data Export**:
   - Download the full balanced dataset as CSV.
   - Download only the generated synthetic minority instances.

---

## 📁 Repository Structure

```
FGAN IA3 Project/
├── app.py                     # Interactive Streamlit Web Application
├── gan_model.py               # PyTorch Tabular GAN (Generator, Discriminator, Loss Tracking)
├── data_processor.py          # Data ingestion, balance diagnostics, scaling, inverse reconstruction
├── evaluator.py               # Downstream classifier comparison & synthetic diversity analysis
├── create_sample_datasets.py  # Benchmark dataset generator
├── sample_data/               # Pre-packaged test datasets
│   ├── credit_default_imbalanced.csv   (12% minority)
│   ├── medical_diagnosis_imbalanced.csv (18% minority)
│   └── balanced_dataset_sample.csv     (50:50 parity)
├── test_pipeline.py           # Integration unit test script
├── run_app.bat                # 1-click launcher for Windows
└── requirements.txt           # Python dependency specifications
```

---

## ⚡ How to Run

### 1. Launch with double-click:
Double click `run_app.bat`

### 2. Or run via command line:
```bash
py -3.11 -m streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.
