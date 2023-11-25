import os
import pandas as pd
from pathlib import Path
  
# Ścieżka źródłowa zbioru .csv
# Paths
SRC_DIR = Path('./data/')
RAW_DIR = SRC_DIR / 'raw datasets'
DATA_DIR = SRC_DIR / 'datasets'
STATS_DIR = SRC_DIR / 'statistics'
SRC_FILE = RAW_DIR / 'raw_dataset_spamassassin.csv'

print(SRC_FILE)

df = pd.read_csv(SRC_FILE)
df = df.drop('email', axis=1)
df['label'] = df['label'].astype(int)
mapping = {1: "spam", 0: "ham"}
df['class'] = df['label'].map(mapping)
for index, row in df.iterrows():
    i = index + 1
    df.at[index, 'id'] = f"{i:04d}"

df = df.rename(columns={'Subject': 'subject', 'content': 'body'})
df = df[['id', 'class', 'label', 'subject', 'body']]

# Zapis do .csv
csv_filename = DATA_DIR / "dataset_spamassassin.csv"
df.to_csv(csv_filename, index=False)
