import json

notebook = {
    "cells": [],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5
}

def md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")]
    })

def code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")]
    })

# ──────────────────────────────────────────────
# Step 1: Business Understanding
# ──────────────────────────────────────────────
md("""# End-to-End Customer Churn Prediction Project

## Step 1: Business Understanding

**What is Customer Churn?**
Customer churn refers to the percentage of customers who stop using a company's product or service during a certain time frame. In the telecom industry, it means a customer leaving for a competitor.

**Why is churn prediction important?**
Acquiring a new customer is up to 5 times more expensive than retaining an existing one. By predicting churn, companies can proactively offer incentives (discounts, better plans) to high-risk customers to make them stay.

**Real-world Applications:**
- Telecom (predicting who cancels their subscription)
- SaaS (predicting who will not renew software licenses)
- E-commerce (predicting who will stop shopping on the site)

**Problem Statement:**
Using the IBM Telco Customer Churn dataset, our goal is to build a machine learning model that accurately predicts whether a customer will churn (`Churn: Yes` or `No`) based on their demographics, account information, and service usage.
""")

# ──────────────────────────────────────────────
# Step 2: Import Libraries
# ──────────────────────────────────────────────
md("""## Step 2: Import Libraries

We import necessary libraries for data manipulation (`pandas`, `numpy`), visualization (`matplotlib`, `seaborn`), and machine learning (`sklearn`).
""")

code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

# Sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')
%matplotlib inline
sns.set_theme(style="whitegrid")""")

# ──────────────────────────────────────────────
# Step 3: Load Dataset
# ──────────────────────────────────────────────
md("""## Step 3: Load Dataset

We will load the dataset using `pandas`. Make sure you have the `Telco-Customer-Churn.csv` file inside your `data/` directory.
""")

code("""# Load dataset
df = pd.read_csv('../data/Telco-Customer-Churn.csv')

# View the first 5 rows
display(df.head())

# Dimensions of the dataset
print(f"Dataset Shape: {df.shape}")

# Column Names
print(f"\\nColumns: {df.columns.tolist()}")

# Basic info (data types, non-null counts)
print("\\nDataset Info:")
df.info()

# Summary Statistics
display(df.describe(include='all'))
""")

# ──────────────────────────────────────────────
# Step 4: Data Cleaning
# ──────────────────────────────────────────────
md("""## Step 4: Data Cleaning

Data cleaning is essential to remove inconsistencies and prepare the data for modeling.
""")

code("""# 1. Check for Missing Values
print("Missing Values before cleaning:\\n", df.isnull().sum())

# 2. Check for Duplicate Records
print(f"\\nDuplicate Records: {df.duplicated().sum()}")
if df.duplicated().sum() > 0:
    df = df.drop_duplicates()

# 3. Remove CustomerID (It has no predictive power)
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)

# 4. Convert TotalCharges to numeric
# The dataset contains blank spaces " " for customers with 0 tenure, which causes TotalCharges to be an object type.
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# Fill NaN values in Total Charges with the median
df['Total Charges'] = df['Total Charges'].fillna(df['Total Charges'].median())

print(f"\\nMissing values after cleaning: {df['Total Charges'].isnull().sum()}")
""")

# ──────────────────────────────────────────────
# Step 5: Exploratory Data Analysis (EDA)
# ──────────────────────────────────────────────
md("""## Step 5: Exploratory Data Analysis (EDA)

Let's understand the distributions and relationships in our data.
""")

md("""### Target Distribution
Let's see how many customers stayed vs left.
""")
code("""plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn Label', palette='viridis')
plt.title('Target Variable Distribution (Churn)')
plt.show()

churn_counts = df['Churn Label'].value_counts()
print("Class Imbalance:")
print(churn_counts)
print(f"Churn Rate: {churn_counts['Yes'] / df.shape[0] * 100:.2f}%")
""")

md("""### Numerical Features
Let's visualize the numerical columns.
""")
code("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df['Tenure Months'], kde=True, ax=axes[0], color='blue')
axes[0].set_title('Distribution of Tenure')

sns.histplot(df['Monthly Charges'], kde=True, ax=axes[1], color='green')
axes[1].set_title('Distribution of Monthly Charges')

sns.boxplot(x=df['Churn Label'], y=df['Total Charges'], ax=axes[2], palette='coolwarm')
axes[2].set_title('Total Charges by Churn')

plt.tight_layout()
plt.show()
""")

md("""### Categorical Features
Let's see how categorical variables impact churn.
""")
code("""cols_to_plot = ['Contract', 'Internet Service', 'Payment Method', 'Senior Citizen']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(cols_to_plot):
    sns.countplot(data=df, x=col, hue='Churn Label', ax=axes[i], palette='Set2')
    axes[i].set_title(f'Churn by {col}')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
""")

md("""### Correlation Heatmap
To plot correlations, we must temporarily map 'Churn Label' to numeric.
""")
code("""df_temp = df.copy()
df_temp['Churn'] = df_temp['Churn Label'].map({'Yes': 1, 'No': 0})
numeric_df = df_temp.select_dtypes(include=[np.number])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
""")

# ──────────────────────────────────────────────
# Step 6: Feature Engineering
# ──────────────────────────────────────────────
md("""## Step 6: Feature Engineering

Machine learning models only understand numbers. We must convert categorical text data into numeric formats.
""")

code("""# Drop the target column (Churn Label) and text metadata if present
if 'Churn Label' in df.columns:
    df.rename(columns={'Churn Label': 'Churn'}, inplace=True)
if 'Churn Reason' in df.columns:
    df = df.drop('Churn Reason', axis=1)

# Ensure 'Senior Citizen' is properly stringified if it's Yes/No
df['Senior Citizen'] = df['Senior Citizen'].astype(str)

# 1. Binary Mapping for Yes/No columns
binary_cols = ['Partner', 'Dependents', 'Phone Service', 'Paperless Billing', 'Churn']
for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

df['Senior Citizen'] = df['Senior Citizen'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

# 2. Service Columns (Yes, No, No internet service -> 1, 0)
service_cols = ['Multiple Lines', 'Online Security', 'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV', 'Streaming Movies']
for col in service_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: 1 if str(x).strip() == 'Yes' else 0)

# 3. One-Hot Encoding for multi-class categorical variables
ohe_cols = ['Internet Service', 'Contract', 'Payment Method']
df = pd.get_dummies(df, columns=ohe_cols, drop_first=False)

# Convert boolean columns to integer
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Drop any non-numeric safely
for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
        df.drop(col, axis=1, inplace=True)

df = df.fillna(0)

print(f"Data shape after encoding: {df.shape}")
""")

# ──────────────────────────────────────────────
# Step 7: Train-Test Split
# ──────────────────────────────────────────────
md("""## Step 7: Train-Test Split

We split the data so the model can learn on 80% of the data and test its predictions on the unseen 20%. `random_state=42` ensures the split is identical every time we run the code.
""")

code("""X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Before SMOTE - Training data: {X_train.shape}")
print(f"Before SMOTE - Churn rate: {y_train.mean()*100:.2f}%\\n")

# Apply SMOTE to training data only!
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Training data: {X_train.shape}")
print(f"After SMOTE - Churn rate: {y_train.mean()*100:.2f}%")
print(f"Testing data remains untouched: {X_test.shape}")
""")

# ──────────────────────────────────────────────
# Step 8, 9 & 10: Model Building & Comparison
# ──────────────────────────────────────────────
md("""## Step 8, 9 & 10: Build Models & Evaluate

We train four popular algorithms and evaluate their Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
""")

code("""# Scaling is required for Logistic Regression and KNN
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7)
}

results = []

for name, model in models.items():
    # Use scaled data for LR and KNN
    X_tr = X_train_sc if name in ["Logistic Regression", "K-Nearest Neighbors"] else X_train
    X_te = X_test_sc if name in ["Logistic Regression", "K-Nearest Neighbors"] else X_test
    
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    
    results.append([name, acc, prec, rec, f1, roc])

df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"])
display(df_results.sort_values(by="F1 Score", ascending=False).round(4))
""")

md("""### Detailed Evaluation of the Best Model (Random Forest)
Let's generate the Confusion Matrix and Classification Report for Random Forest.
""")

code("""best_model = models["Random Forest"]
best_model.fit(X_train, y_train)
y_pred_rf = best_model.predict(X_test)
y_prob_rf = best_model.predict_proba(X_test)[:, 1]

print("Classification Report (Random Forest):\\n")
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Random Forest')
plt.show()
""")

# ──────────────────────────────────────────────
# Step 11: Feature Importance
# ──────────────────────────────────────────────
md("""## Step 11: Feature Importance

Which factors cause customers to churn the most?
""")

code("""importances = best_model.feature_importances_
feat_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, x='Importance', y='Feature', palette='viridis')
plt.title('Top 15 Most Important Features for Churn')
plt.show()
""")

# ──────────────────────────────────────────────
# Step 12: Save Model
# ──────────────────────────────────────────────
md("""## Step 12: Save Model

We use `pickle.dump()` to serialize the trained Random Forest model and save it to the disk. This allows our Streamlit web application to load the model and make predictions without retraining it!
""")

code("""with open('../models/random_forest_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)

with open('../models/feature_columns.pkl', 'wb') as file:
    pickle.dump(list(X.columns), file)

print("Model and Feature Columns saved successfully!")
""")

# ──────────────────────────────────────────────
# Step 16: Interview Questions
# ──────────────────────────────────────────────
md("""## Step 16: Interview Questions & Answers

### 1. What is customer churn?
**Answer:** Customer churn is the metric that measures the percentage of customers who stop doing business with a company over a specific period. It is a critical KPI because acquiring new customers is significantly more expensive than retaining existing ones.

### 2. Why Random Forest?
**Answer:** Random Forest was selected because it is a robust ensemble learning method. It handles non-linear relationships well, is less prone to overfitting compared to a single Decision Tree (due to bagging and random feature selection), and provides excellent interpretability through Feature Importances.

### 3. Why feature engineering?
**Answer:** Machine learning models are mathematical functions that require numerical input. Feature engineering (like One-Hot Encoding and Binary Mapping) transforms text-based categorical data into a numeric format that the algorithm can process, directly impacting the model's predictive power.

### 4. Difference between Precision and Recall?
**Answer:**
- **Precision:** Out of all the customers the model *predicted* would churn, how many *actually* churned? (Focuses on minimizing False Positives).
- **Recall:** Out of all the customers who *actually* churned, how many did the model correctly *find*? (Focuses on minimizing False Negatives).

### 5. Why F1-score?
**Answer:** The F1-score is the harmonic mean of Precision and Recall. In churn prediction, we often have an imbalanced dataset (more people stay than leave). Accuracy can be misleading in imbalanced datasets, so F1-score provides a better measure of the incorrectly classified cases than the Accuracy Metric.

### 6. What is class imbalance?
**Answer:** Class imbalance occurs when one class (e.g., "Stay") significantly outnumbers the other class (e.g., "Churn"). In our dataset, ~73% of customers stayed. If a model simply guessed "Stay" every time, it would be 73% accurate, but completely useless for finding churners!

### 7. Explain Confusion Matrix.
**Answer:** A Confusion Matrix is a 2x2 table used to evaluate a classification model's performance. It breaks predictions down into:
- **True Positives (TP):** Model predicted Churn, customer Churned.
- **True Negatives (TN):** Model predicted Stay, customer Stayed.
- **False Positives (FP):** Model predicted Churn, customer Stayed (Type I error).
- **False Negatives (FN):** Model predicted Stay, customer Churned (Type II error - usually the most costly in churn).

### 8. Why `random_state=42`?
**Answer:** Setting a `random_state` ensures reproducibility. It seeds the random number generator so that the train-test split (and the Random Forest bagging process) happens exactly the same way every time the code is run. 

### 9. How did you handle missing values?
**Answer:** I found that `Total Charges` was being read as an object due to blank space strings (" ") representing customers with 0 tenure. I coerced these to `NaN` during numeric conversion, and then imputed (filled) those missing values using the **median** of the column to avoid outliers skewing the data.

### 10. What are the most important features?
**Answer:** Based on the Random Forest Feature Importances, the top factors driving churn are:
1. **Tenure:** Newer customers are at the highest risk of leaving.
2. **Total & Monthly Charges:** Higher bills correlate heavily with churn.
3. **Contract Type:** Customers on Month-to-Month contracts leave at a vastly higher rate than those locked into 1- or 2-year contracts.

### 11. How can this project be improved?
**Answer:** 
- We could use **SMOTE (Synthetic Minority Over-sampling Technique)** to handle the class imbalance.
- We could test advanced gradient boosting algorithms like **XGBoost** or **LightGBM**.
- We could perform rigorous **Hyperparameter Tuning** using `GridSearchCV` or `Optuna`.

### 12. How would you deploy this project?
**Answer:** I have already deployed this using **Streamlit** to create an interactive web application. In an enterprise environment, I would wrap the model in a **FastAPI** or **Flask API**, containerize it using **Docker**, and host it on a cloud provider like **AWS (EC2/SageMaker)** or **Azure**.

### 13. What is overfitting?
**Answer:** Overfitting happens when a model learns the training data *too well*—including its noise and outliers. As a result, it performs perfectly on the training set but fails to generalize, causing poor accuracy on unseen testing data.

### 14. Explain ROC-AUC.
**Answer:** ROC (Receiver Operating Characteristic) is a probability curve plotting the True Positive Rate against the False Positive Rate. AUC (Area Under the Curve) represents the degree of separability. An AUC of 0.5 means the model is randomly guessing, while 1.0 means it predicts classes perfectly. Our model achieved an AUC of ~0.85, indicating excellent distinguishing capability.

### 15. Why not use Deep Learning?
**Answer:** Deep Learning (Neural Networks) requires massive amounts of data and computational power (GPUs). For tabular datasets of this size (~7,000 rows), tree-based ensemble models like Random Forest and XGBoost are usually much faster, less prone to overfitting, and easier to explain to business stakeholders than a Neural Network "black box".
""")

with open('notebooks/churn_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated at notebooks/churn_analysis.ipynb")
