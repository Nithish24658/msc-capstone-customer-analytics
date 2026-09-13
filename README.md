# 👥 Customer Segmentation & Churn Analytics

> End-to-end M.Sc. Data Science project combining customer segmentation, churn prediction, and survival analysis.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)](https://xgboost.readthedocs.io/)

## 🎯 Project overview

This project demonstrates a practical customer analytics workflow using a **55,000-record synthetic CRM and clickstream dataset**. The pipeline answers three business questions:

1. **Who are the different customer segments?**
2. **Which customers are most likely to churn?**
3. **How does retention change over the customer lifecycle?**

The project is designed as a portfolio demonstration of data preparation, unsupervised learning, supervised machine learning, and survival analysis.

## 🔎 Key techniques

| Area | Approach |
|---|---|
| Customer segmentation | K-Means clustering with feature scaling |
| Churn prediction | XGBoost classification |
| Retention analysis | Kaplan-Meier survival curves |
| Feature engineering | RFM + clickstream engagement features |
| Visualization | Matplotlib and Seaborn |

## 📊 Dataset

The included data-generation pipeline creates **55,000 synthetic customer records** with:

- Recency, frequency, and monetary value
- Customer tenure
- Web sessions and page views
- Cart additions
- Click-through rate
- Binary churn target

Because the dataset is synthetic, model metrics should be interpreted as **demonstration results rather than production business outcomes**.

## 🧠 Analysis pipeline

```text
Synthetic customer data
        │
        ▼
Feature generation & validation
        │
        ├───────────────┐
        ▼               ▼
K-Means segmentation   XGBoost churn model
        │               │
        └───────┬───────┘
                ▼
       Customer risk insights
                │
                ▼
       Kaplan-Meier analysis
                │
                ▼
       Retention visualization
```

### 1. Customer segmentation

K-Means clustering is applied to standardized behavioral and value-related features to identify distinct customer groups. The resulting segments can be used to support differentiated engagement strategies.

### 2. Churn prediction

An XGBoost classifier estimates churn risk from customer behavior. The workflow accounts for class imbalance and evaluates predictive performance using classification metrics.

### 3. Survival analysis

Kaplan-Meier estimation is used to study customer retention over time and visualize the probability of remaining active across the observed lifecycle.

## 📁 Repository structure

```text
├── data/                         # Generated datasets (local)
├── models/                       # Generated model artifacts (local)
├── notebooks/plots/              # Generated visualizations
│   └── survival_curves.png
├── generate_data.py              # Generate the synthetic dataset
├── customer_segmentation.py      # K-Means segmentation
├── churn_model.py                # XGBoost churn prediction
├── survival_curves.py            # Kaplan-Meier analysis
├── requirements.txt
└── README.md
```

## 🚀 Quick start

```bash
git clone https://github.com/Nithish24658/msc-capstone-customer-analytics.git
cd msc-capstone-customer-analytics
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python generate_data.py
python customer_segmentation.py
python churn_model.py
python survival_curves.py
```

## 📈 Output

The pipeline produces segmented customer data, churn-risk outputs, and a Kaplan-Meier survival visualization. Generated data and model artifacts are intentionally kept out of version control when configured through `.gitignore`.

## 💼 Portfolio value

This project demonstrates skills relevant to **Data Analyst, Data Scientist, and Machine Learning** roles:

- Python and Pandas data processing
- Exploratory customer analytics
- Feature engineering
- Unsupervised and supervised machine learning
- Model evaluation
- Survival analysis
- Business-oriented interpretation of analytical results

## 👤 Author

**Nithishsaran KM**  
M.Sc. Data Science | Data Analytics | Machine Learning

GitHub: [@Nithish24658](https://github.com/Nithish24658)
