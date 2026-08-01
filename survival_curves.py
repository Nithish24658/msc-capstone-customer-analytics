import os
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

# Load Data
df = pd.read_csv('data/customer_final_output.csv')
kmf = KaplanMeierFitter()

plt.figure(figsize=(12, 7))

# Track historical decay rates of the clusters
for cluster_id in sorted(df['persona_cluster'].unique()):
    mask = (df['persona_cluster'] == cluster_id)
    kmf.fit(
        durations=df.loc[mask, 'tenure_days'],
        event_observed=df.loc[mask, 'churn'],
        label=f'Segment Persona {cluster_id}'
    )
    kmf.plot_survival_function(ci_show=False)

plt.title('Kaplan-Meier Customer Retention Curves per Segment Persona', fontsize=14, weight='bold')
plt.xlabel('Customer Lifespan Duration (Days)', fontsize=12)
plt.ylabel('Probability of Retention', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# --- FIX: Automatically create folders if they do not exist ---
output_dir = 'notebooks/plots'
os.makedirs(output_dir, exist_ok=True)

# Save and show the plot cleanly
output_path = os.path.join(output_dir, 'survival_curves.png')
plt.savefig(output_path, dpi=300)
print(f"Success! Survival analysis plot mapped and exported to '{output_path}'.")
plt.show()
