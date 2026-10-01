# Oasis Infobyte Data Science Internship (OIBSIP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-darkblue?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

This repository contains my completed machine learning and data science projects for the **Oasis Infobyte Data Science Internship**. 

All projects follow end-to-end industry best practices: thorough Exploratory Data Analysis (EDA), domain feature engineering, data leakage prevention using Scikit-Learn `Pipeline`, rigorous cross-evaluation across multiple models, and production model serialization using `joblib`.

---

## 📌 Projects Overview

| Task # | Project Name | Domain / Type | Key Algorithms | Best Model Score |
| :---: | :--- | :--- | :--- | :---: |
| **Task 1** | **[Iris Flower Classification](./Project%201%20-%20Iris%20Flower%20Classification/)** | Multiclass Classification | Logistic Regression, Random Forest | **93.33% Accuracy** |
| **Task 3** | **[Car Price Prediction](./Project%202%20-%20Car%20Price%20Prediction/)** | Supervised Regression | Linear Regression, Random Forest Regressor | **R²: 0.6082, MAE: ₹1.32L** |
| **Task 4** | **[Email/SMS Spam Detection](./Project%203%20-%20Spam%20Detection/)** | NLP & Binary Classification | Multinomial Naive Bayes, Logistic Regression | **97.87% Acc, 100% Prec** |

---

## 🌸 Task 1: Iris Flower Classification

### Objective
Classify Iris flowers into one of three species (*Iris setosa*, *Iris versicolor*, or *Iris virginica*) based on sepal and petal morphology.

### Workflow & Highlights
- **Complete Dataset**: Analyzed all 150 observations across all 4 features (sepal length, sepal width, petal length, petal width).
- **EDA & Visualisations**: Pairplot and species-grouped boxplots revealed that **Petal Length** and **Petal Width** provide near-linear separability for *Setosa* and clean, non-overlapping Interquartile Ranges (IQR) between *Versicolor* and *Virginica*.
- **Data Leakage Prevention**: Encapsulated `StandardScaler` and estimators inside Scikit-Learn `Pipeline`.
- **Validation**: Stratified 80/20 train-test split (`random_state=42`) ensuring exactly 10 samples per class in testing.

### Performance Summary
| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Best)** | **93.33%** | **0.9333** | **0.9333** | **0.9333** |
| Random Forest Classifier | 90.00% | 0.9024 | 0.9000 | 0.8997 |

*Winning Pipeline saved as: `iris_logistic_regression_pipeline.joblib`*

---

## 🚗 Task 3: Car Price Prediction

### Objective
Predict the market resale value of used cars (in Lakhs INR) based on vehicle age, showroom price, mileage, fuel type, transmission, seller type, and brand.

### Workflow & Highlights
- **Data Cleaning & Normalization**: Addressed null values, removed duplicate records, and normalized categorical strings (`.str.strip().str.lower()`).
- **Feature Engineering**:
  - Extracted vehicle `Brand` make from raw `Car_Name` prior to dimensionality reduction.
  - Calculated mechanical depreciation via `Car_Age = 2024 - Year` and dropped collinear `Year`.
- **EDA Insights**: Verified high positive correlation between `Present_Price` and `Selling_Price` ($r = 0.88$) and visualized vehicular depreciation across age.
- **Categorical Encoding**: Handled multi-level categories using One-Hot Encoding (`drop_first=True`) to avoid the Dummy Variable Trap.
- **Feature Importance**: Random Forest confirmed that **Present Price (89.2%)** and **Car Age (6.6%)** dominate used car valuations.

### Performance Summary
| Model | MAE (Lakhs INR) | RMSE (Lakhs INR) | R² Score |
| :--- | :---: | :---: | :---: |
| **Random Forest Regressor (Best)** | **₹1.3207 L** | **₹3.1777 L** | **0.6082** |
| Linear Regression | ₹1.8088 L | ₹3.3993 L | 0.5517 |

*Winning Model saved as: `car_price_rf_model.joblib`*

---

## ✉️ Task 4: Email/SMS Spam Detection

### Objective
Develop an automated NLP pipeline that classifies incoming SMS/email messages as **Ham** (legitimate) or **Spam** (unsolicited / phishing).

### Workflow & Highlights
- **Text Preprocessing**: Lowercase normalization, punctuation stripping via regex, and English stopword elimination with NLTK.
- **WordCloud Visualisations**: Uncovered key spam keywords (`"free"`, `"call"`, `"txt"`, `"claim"`, `"prize"`, `"win"`, `"guaranteed"`) vs. casual ham conversation.
- **Vectorization**: Extracted high-signal n-gram vectors using `TfidfVectorizer(max_features=3000)`.
- **Stratified Validation**: Handled severe $87\% / 13\%$ class imbalance using stratified train-test splitting.
- **Analytical Deep-Dive on Recall**: Documented the trade-offs of False Positives (legitimate emails blocked) vs. False Negatives (spam entering inbox).

### Performance Summary
| Model | Accuracy | Precision (Spam) | Recall (Spam) | F1 Score (Spam) |
| :--- | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes (Best)** | **97.87%** | **100.00%** | **83.21%** | **0.9083** |
| Logistic Regression | 95.16% | 95.51% | 64.89% | 0.7727 |

*Winning Pipeline saved as: `spam_detector_pipeline.joblib`*

---

## 📂 Repository Structure

```text
Oasis-Infobyte/
│
├── Project 1 - Iris Flower Classification/
│   ├── iris_flower_classification_oasis.ipynb     # Complete Task 1 notebook
│   └── iris_logistic_regression_pipeline.joblib   # Trained pipeline artifact
│
├── Project 2 - Car Price Prediction/
│   ├── car.csv                                    # Used car sales dataset
│   ├── car_price_prediction_oasis.ipynb           # Complete Task 3 notebook
│   └── car_price_rf_model.joblib                  # Trained Random Forest regressor
│
├── Project 3 - Spam Detection/
│   ├── spam.csv                                   # SMS spam collection dataset
│   ├── spam_detection_nlp_oasis.ipynb             # Complete Task 4 notebook
│   └── spam_detector_pipeline.joblib              # Trained NLP pipeline artifact
│
├── .gitignore                                     # Clean environment exclusions
├── requirements.txt                               # Project dependencies
└── README.md                                      # Documentation & project summaries
```

---

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Linux/macOS
   .\venv\Scripts\activate        # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

---

## 👤 Author
- **Swayam**
- *Oasis Infobyte Data Science Intern*
- Email: swayamkerekar@gmail.com
