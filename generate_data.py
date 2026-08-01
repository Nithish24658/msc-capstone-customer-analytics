import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("Creating balanced dataset...")
np.random.seed(42)
n_samples = 55000

# Base IDs and Dates
customer_ids = [f"CUST_{i:05d}" for i in range(n_samples)]
tenure_days = np.random.randint(15, 730, size=n_samples)
signup_dates = [datetime(2026, 8, 1) - timedelta(days=int(t)) for t in tenure_days]

# Core metrics
recency = np.random.exponential(scale=45, size=n_samples).astype(int)
recency = np.clip(recency, 1, 365)

frequency = np.random.poisson(lam=6, size=n_samples) + 1
monetary_value = frequency * np.random.normal(loc=42, scale=12, size=n_samples)
monetary_value = np.where(monetary_value < 5, 5, monetary_value)

# Clickstream features 
web_sessions = frequency * np.random.randint(2, 8, size=n_samples)
page_views = web_sessions * np.random.randint(3, 12, size=n_samples)
cart_additions = (frequency * 0.6 + np.random.normal(2, 1, n_samples)).astype(int)
cart_additions = np.clip(cart_additions, 0, None)
ctr = np.random.beta(a=2, b=6, size=n_samples)

# Enhanced mathematical predictive signal for target churn modeling
# Stronger negative weights on frequency/tenure and heavy positive weight on high recency
churn_score = (recency * 0.04) - (frequency * 0.35) - (tenure_days * 0.003) + (1.5 * (1 - ctr))
churn_prob = 1 / (1 + np.exp(-churn_score))

# Set a threshold that yields a ~20% churn rate for robust ML optimization
churn = np.where(churn_prob > 0.55, 1, 0)

df = pd.DataFrame({
    'customer_id': customer_ids,
    'signup_date': signup_dates,
    'tenure_days': tenure_days,
    'recency': recency,
    'frequency': frequency,
    'monetary_value': np.round(monetary_value, 2),
    'web_sessions': web_sessions,
    'page_views': page_views,
    'cart_additions': cart_additions,
    'click_through_rate': np.round(ctr, 4),
    'churn': churn
})

df.to_csv('data/customer_raw_data.csv', index=False)
print(f"Dataset compiled! Saved 55,000 records to 'data/customer_raw_data.csv'. Churn rate: {df['churn'].mean():.2%}")
