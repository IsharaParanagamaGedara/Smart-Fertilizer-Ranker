# 🌾 Smart-Fertilizer-Ranker

A machine learning solution for predicting the top-3 most suitable fertilizers based on environmental and soil conditions using **XGBoost**, **Optuna**, and **ranking optimization**.  
Developed as part of the **Kaggle Playground Series – Season 5, Episode 6 (2025)**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/Model-XGBoost-yellowgreen?logo=xgboost)
![MAP@3](https://img.shields.io/badge/Eval-MAP@3-orange)
[![View on Kaggle](https://img.shields.io/badge/Kaggle-View%20Notebook-blue?logo=kaggle)](https://www.kaggle.com/code/sulaniishara/smart-fertilizer-ranker-map-3-xgboost)


---

## 📂 Dataset Overview

This project uses the **Fertilizer Prediction** dataset from the **Kaggle Playground Series – Season 5, Episode 6 (2025)**.  
It simulates real-world agricultural decision-making by predicting fertilizer recommendations based on:

- Climatic factors
- Soil composition
- Crop type

### 🧾 Key Features

- **Climatic Attributes**  
  `Temperature (°C)`, `Humidity (%)`, `Moisture (%)`

- **Soil & Crop Information**  
  `Soil Type`, `Crop Type`

- **Soil Nutrients**  
  `Nitrogen`, `Phosphorous`, `Potassium`

- **Target Variable**  
  `Fertilizer Name` — the recommended fertilizer (e.g., *Urea*, *14‑35‑14*, etc.)

### 📁 Files Provided

| File                      | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `train.csv`               | 750,000 samples — includes all features and the target fertilizer label     |
| `test.csv`                | 250,000 samples — requires top-3 fertilizer predictions per row             |
| `Fertilizer Prediction.csv` | 100,000 samples — the original dataset used to generate the synthetic data ([IEEE DataPort](https://ieee-dataport.org/documents/soil-fertility-data-fertilizer-recommendation), [Kaggle Dataset](https://www.kaggle.com/datasets/irakozekelly/fertilizer-prediction/data)) |

This dataset is well-suited for **multi-label classification**, **ranking models**, and **top‑K evaluation** using metrics like **MAP@3**.

---

## 🎯 Project Objective

The goal is to develop a **robust machine learning model** that predicts the **top-3 most appropriate fertilizers** for a given set of environmental and soil attributes.

### ✅ Key Pipeline Components

- 🔍 **Exploratory Data Analysis (EDA)**  
  Understand feature distributions, relationships, and handle class imbalance.

- ⚙️ **Feature Engineering**  
  Encode categorical variables and ensure consistency across training and test datasets.

- 📈 **Modeling with XGBoost**  
  Train a gradient-boosted tree model with hyperparameters optimized using **Optuna**.

- 🔢 **Multi-Label Ranking Strategy**  
  Generate ranked fertilizer predictions for each test sample.

- 🧪 **Evaluation Framework**
  - Cross-validated **Out-of-Fold (OOF)** predictions
  - Metrics:  
    - **Log Loss**  
    - **Mean Average Precision @3 (MAP@3)**

---

## 🌱 Outcome

By leveraging structured learning and ranking optimization, this project generates data-driven fertilizer recommendations that:

- Enhance **crop yield**
- Promote **sustainable agriculture**
- Support **precision farming decisions**

---

## 🧠 Tech Stack

- Python 🐍
- Pandas, NumPy, Matplotlib, Seaborn
- XGBoost, Optuna, Scikit-learn
- Jupyter / Kaggle Notebook

---

### 📜 License

This project is for research and educational purposes under standard open use guidelines.  
All dataset credits to Kaggle and original authors.

---

### 🤝 Acknowledgements

Special thanks to:
- Kaggle Playground Series team for hosting the competition
- Dataset contributors from [IEEE DataPort](https://ieee-dataport.org/)
- Open-source ML libraries and contributors

---

> 🌱 _Empowering smarter agriculture with data science._

