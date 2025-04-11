# survival_analysis/km_validation.py

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

def plot_km_curve(df, duration_col='time', event_col='event', group_col=None):
    kmf = KaplanMeierFitter()

    if group_col:
        for name, grouped_df in df.groupby(group_col):
            kmf.fit(grouped_df[duration_col], event_observed=grouped_df[event_col], label=str(name))
            kmf.plot_survival_function()
    else:
        kmf.fit(df[duration_col], event_observed=df[event_col])
        kmf.plot_survival_function()

    plt.title("Kaplan-Meier Survival Curve")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.show()

# Example usage:
# df = pd.read_csv("clinical_data.csv")
# plot_km_curve(df, duration_col="follow_up_days", event_col="death_event", group_col="treatment_group")
