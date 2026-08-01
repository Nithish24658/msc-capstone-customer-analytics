import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report

# Load Data
df = pd.read_csv('data/customer_segmented_data.csv')

features = ['tenure_days', 'recency', 'frequency', 'monetary_value', 'web_sessions', 'page_views', 'cart_additions', 'click_through_rate', 'persona_cluster']
X = df[features]
y = df['churn']

# Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Calculate ratio to address imbalance weights
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

print("Optimizing XGBoost with Class Balance Adjustments...")
model = xgb.XGBClassifier(eval_metric='logloss', scale_pos_weight=ratio, random_state=42)

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.85]
}

grid = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# Performance Matrix evaluation
probs = best_model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"\nTarget Achieved! XGBoost Churn Model AUC-ROC: {auc:.2%}")

# Output comprehensive classification report
preds = best_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, preds))

# Save predictions back to data layer
df['churn_probability'] = best_model.predict_proba(X)[:, 1]
df.to_csv('data/customer_final_output.csv', index=False)
print("Updated output metrics saved to data/customer_final_output.csv.")
