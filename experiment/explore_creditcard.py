"""Explore the OpenML credit-card dataset (ULB) for the auditable experiment.

We use it only for its real transaction-amount distribution, not the fraud labels.
Prints a summary, shows a sample, and saves a 10k amounts sample for reuse so the
experiment does not re-download.
"""
import os

import pandas as pd
from sklearn.datasets import fetch_openml

print("Fetching OpenML credit-card dataset (name='creditcard', version=1)...")
bunch = fetch_openml(name="creditcard", version=1, as_frame=True)
df = bunch.frame
print("shape:", df.shape)
print("first cols:", list(df.columns)[:5], "| last cols:", list(df.columns)[-3:])

amt = pd.to_numeric(df["Amount"], errors="coerce").dropna()
print()
print("Amount (real transaction amounts):")
print(amt.describe(percentiles=[0.5, 0.9, 0.99]).to_string())
print("nonzero:", int((amt > 0).sum()), "of", len(amt))
print()
print("12 sample amounts:", amt.sample(n=12, random_state=0).round(2).tolist())

out_dir = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "creditcard_amounts_10k.csv")
amt.sample(n=10000, random_state=0).to_csv(out, index=False, header=["Amount"])
print("saved 10k amount sample to", out)
