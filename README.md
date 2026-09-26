# 🏦 Bank Customer Churn Prediction

A PyTorch deep learning project that predicts whether a bank customer is likely to churn (close their account), paired with an interactive Streamlit app for real-time and batch predictions.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Overview

Customer churn is one of the most expensive problems for retail banks — acquiring a new customer costs far more than retaining an existing one. This project trains a fully-connected neural network on 10,000 bank customer records to estimate each customer's probability of churning, and wraps the model in a Streamlit dashboard for exploration and inference.

**Key features:**
- 🧠 Custom PyTorch neural network trained from scratch
- ⚖️ Class imbalance handled with SMOTE oversampling
- 📊 Interactive Streamlit app with single-customer and batch (CSV) prediction
- 📈 Built-in data insights: churn rate by geography, feature correlations, feature distributions
- 🎛️ Adjustable decision threshold for the churn/stay classification

---

## 🗂️ Project Structure

```
Bank_Churn-main/
├── app.py                          # Streamlit web application
├── Notebook_Project.ipynb          # Model training & evaluation notebook
├── requirements.txt                # Python dependencies
├── dataset/
│   └── Customer-Churn-Records.csv  # Training dataset (10,000 records)
├── artifacts/
│   ├── model_bank_costumers.pt     # Trained PyTorch model weights
│   ├── OneHotEncoder.pkl           # Fitted encoder for categorical features
│   └── StandardScaler.pkl          # Fitted scaler for numerical features
└── LICENSE
```

---

## 📊 Dataset

The model is trained on the **Bank Customer Churn Records** dataset (10,000 rows). Each row represents a customer, described by:

| Feature | Description |
|---|---|
| `CreditScore` | Customer's credit score |
| `Geography` | Country (France, Germany, Spain) |
| `Gender` | Male / Female |
| `Age` | Customer age |
| `Tenure` | Years as a bank customer |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products used |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is an active member |
| `EstimatedSalary` | Estimated annual salary |
| `Exited` | **Target** — 1 if the customer churned, 0 otherwise |

---

## 🧠 Model & Training

**Architecture** — a fully-connected feed-forward network:

```
Input (13 features) → 256 → 128 → 64 → 32 → 1 (Sigmoid)
                       ReLU  ReLU  ReLU  ReLU + Dropout(0.8)
```

**Pipeline:**
1. Numerical features are standardized with `StandardScaler`; categorical features (`Geography`, `Gender`) are one-hot encoded.
2. The dataset is imbalanced (~20% churn rate), so **SMOTE** is used to oversample the minority class before training.
3. The network is trained with **binary cross-entropy loss** and the **Adam optimizer**, using early stopping and a `ReduceLROnPlateau` learning-rate scheduler.

**Test set performance:**

| Metric | Score |
|---|---|
| Accuracy | 0.896 |
| ROC AUC | 0.896 |
| Precision | 0.846 |
| Recall | 0.965 |

> ⚠️ Because training data was rebalanced with SMOTE, the model's output probabilities are not calibrated to the real-world churn rate (~20%) and should be interpreted as relative risk scores rather than true probabilities.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+

### Installation

```bash
git clone <your-repo-url>
cd Bank_Churn-main
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Retrain the model

Open `Notebook_Project.ipynb` to explore the full preprocessing, training, and evaluation pipeline, or to retrain the model on updated data.

---

## 💻 App Walkthrough

The Streamlit app has four tabs:

- **🔮 Prediction** — Enter a customer's profile in the sidebar and get an instant churn probability, a risk gauge, and a comparison to the average customer.
- **📁 Batch** — Upload a CSV of multiple customers and download predictions for all of them at once.
- **📊 Data Insights** — Explore churn rate by geography, feature correlations, and feature distributions split by churn outcome.
- **ℹ️ About** — Model architecture, preprocessing steps, and known limitations.

---

## 🛠️ Tech Stack

- **PyTorch** — model definition and training
- **scikit-learn** — preprocessing (`StandardScaler`, `OneHotEncoder`) and metrics
- **imbalanced-learn (SMOTE)** — class balancing
- **Streamlit** — web application
- **Plotly** — interactive charts

---

## ⚠️ Limitations

- Probabilities are affected by SMOTE rebalancing and are not perfectly calibrated to the real churn rate.
- Dropout is set high (0.8), which can make single-prediction outputs slightly noisy.
- This is an educational/portfolio project — not a substitute for a full customer retention or CLV analysis.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
