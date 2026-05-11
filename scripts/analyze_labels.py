#!/usr/bin/env python3
"""Analyze label distributions for consolidation planning."""
import pandas as pd
from collections import Counter
from pathlib import Path

READY = Path(r"c:\Users\hp\Desktop\AI assistant\data\ready")

# ── Triage ──
print("=== TRIAGE ===")
df = pd.read_csv(READY / "triage" / "train.csv")
labels = df["condition"].values
c = Counter(labels)
print(f"Total rows: {len(df)}")
print(f"Unique labels: {len(c)}")
print(f"Labels with <30 samples: {sum(1 for v in c.values() if v < 30)}")
print(f"Labels with <10 samples: {sum(1 for v in c.values() if v < 10)}")
print(f"Labels with >=100 samples: {sum(1 for v in c.values() if v >= 100)}")
print()
print("Top 30:")
for label, count in c.most_common(30):
    print(f"  {label}: {count}")
print()
print("Case variant pairs:")
seen = {}
for label in c:
    key = label.lower().strip()
    if key in seen:
        print(f'  "{seen[key]}" vs "{label}"')
    else:
        seen[key] = label

# ── Dialogue ──
print("\n=== DIALOGUE ===")
df2 = pd.read_csv(READY / "dialogue" / "train.csv")
labels2 = df2["intent"].values
c2 = Counter(labels2)
print(f"Total rows: {len(df2)}")
print(f"Unique intents: {len(c2)}")
print(f"Intents with <30 samples: {sum(1 for v in c2.values() if v < 30)}")
print(f"Intents with <10 samples: {sum(1 for v in c2.values() if v < 10)}")
print(f"Intents with >=100 samples: {sum(1 for v in c2.values() if v >= 100)}")
print()
print("Top 30:")
for label, count in c2.most_common(30):
    print(f"  {label}: {count}")
print()
print("Case variant pairs:")
seen2 = {}
for label in c2:
    key = label.lower().strip()
    if key in seen2:
        print(f'  "{seen2[key]}" vs "{label}"')
    else:
        seen2[key] = label
