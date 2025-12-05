# 🧠 Stroke Prediction Project

## 📌 Overview

According to the World Health Organization (WHO), stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths. This project aims to predict whether a patient is likely to experience a stroke based on input parameters like gender, age, various diseases, and smoking status.

## 📂 Dataset

The dataset contains 11 clinical features for predicting stroke events.
**Source:** [Link to Kaggle/Source]
**Attributes:**

- `id`: Unique identifier
- `gender`: "Male", "Female", or "Other"
- `age`: Age of the patient
- `hypertension`: 0 (No), 1 (Yes)
- `heart_disease`: 0 (No), 1 (Yes)
- `ever_married`: "No" or "Yes"
- `work_type`: "children", "Govt_jov", "Never_worked", "Private", or "Self-employed"
- `Residence_type`: "Rural" or "Urban"
- `avg_glucose_level`: Average glucose level in blood
- `bmi`: Body mass index
- `smoking_status`: "formerly smoked", "never smoked", "smokes", or "Unknown"
- **Target:** `stroke`: 1 (Stroke), 0 (No Stroke)

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Imbalanced-learn (SMOTE)
- **Models:** Logistic Regression, Random Forest, XGBoost (Example)

## 📊 Key Analysis & Steps

1.  **Data Cleaning:** Handling missing values in `bmi` and treating "Unknown" values in `smoking_status`.
2.  **EDA (Exploratory Data Analysis):** Analyzing the distribution of features and their relationship with the target variable.
3.  **Data Preprocessing:**
    - One-Hot Encoding for categorical variables.
    - Feature Scaling for numerical variables (`avg_glucose_level`, `bmi`, `age`).
    - **Handling Imbalance:** The dataset is highly imbalanced. Techniques like SMOTE (Synthetic Minority Over-sampling Technique) are applied.
4.  **Model Training & Evaluation:** Comparing models based on Recall, F1-Score, and AUC-ROC (Accuracy is not a good metric here due to class imbalance).

## 🚀 How to Run

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/stroke-prediction-project.git](https://github.com/YOUR_USERNAME/stroke-prediction-project.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the notebook in `notebooks/` directory.

## 📈 Results

_(Replace this with your actual results)_

- **Best Model:** Random Forest Classifier
- **Accuracy:** 95%
- **Recall (Stroke class):** 80%
- **AUC-ROC:** 0.92

## 📜 License

This project is for educational purposes.
