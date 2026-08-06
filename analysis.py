import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "player_data.csv"))

print(df.head())
print(df.describe())

print(df.head(10))
print(df.dtypes)
print(df.isnull().sum())

df['Price'] = df['Price'].str.replace('$', '', regex=False).astype(float)

print(df.dtypes)
print(df.describe())

weights = {
    'Pace':       0.15,
    'Shooting':   0.20,
    'Passing':    0.15,
    'Dribbling':  0.20,
    'Defense':    0.15,
    'Physical':   0.15
}

stat_cols = list(weights.keys())
for col in stat_cols:
    df[col + '_z'] = (df[col] - df[col].mean()) / df[col].std()

df['Composite_Score'] = sum(df[col + '_z'] * w for col, w in weights.items())

print(df[['Name', 'Overall', 'Composite_Score', 'Price']].sort_values('Composite_Score', ascending=False).head(10))

df['Value_Score'] = df['Composite_Score'] / df['Price']

undervalued = df[df['Composite_Score'] > 0][['Name', 'Overall', 'Composite_Score', 'Price', 'Value_Score']]
undervalued = undervalued.sort_values('Value_Score', ascending=False).head(15)

print(undervalued.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(df['Price'], df['Composite_Score'], alpha=0.6, color='steelblue')
for _, row in undervalued.head(5).iterrows():
    ax1.annotate(row['Name'], (row['Price'], row['Composite_Score']),
                 fontsize=7, ha='right')
ax1.set_xlabel('Price (millions)')
ax1.set_ylabel('Composite Score')
ax1.set_title('Composite Score vs Price')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

top10 = undervalued.head(10)
ax2.barh(top10['Name'], top10['Value_Score'], color='steelblue')
ax2.set_xlabel('Value Score (composite per million)')
ax2.set_title('Top 10 Undervalued Players')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('fc26_analysis.png', dpi=150)
plt.show()