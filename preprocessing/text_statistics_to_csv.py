import sys
print("Python version: ", sys.version)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import requests
import tensorflow_text as tf_text
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import string
# import chardet
from collections import Counter
from nltk import FreqDist
# from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('stopwords')

def dataset_to_df(num):
    # Odczyt pliku .csv

    dataset_df = pd.read_csv(os.path.join(DATA_DIR, f"dataset_enron{num}.csv"))
    return dataset_df   

def statistics(dataset_df):
    # Statystyka słów

    body_column = dataset_df["body"]
    class_column = dataset_df["class"]
    spam_tokens = []
    ham_tokens = []
    spam_data = []
    ham_data = []
    df = pd.DataFrame(columns=['word', 'freq'])
    tokenizer = tf_text.WhitespaceTokenizer()

    for body_text, class_text in tqdm(zip(body_column, class_column), total=len(dataset_df)):
        new_body = str(body_text).translate(str.maketrans("", "", string.punctuation))
        tokens = tokenizer.tokenize([new_body])

        for token in tokens:
            token_list = token.numpy().tolist()
            if class_text == "spam":
                spam_tokens.extend(token_list)
            elif class_text == "ham":
                ham_tokens.extend(token_list)

    spam_counts = Counter(spam_tokens)
    ham_counts = Counter(ham_tokens)

    for word, freq in spam_counts.most_common():
        spam_data.append({'word': word.decode('utf-8'), 'freq': freq})

    for word, freq in ham_counts.most_common():
        ham_data.append({'word': word.decode('utf-8'), 'freq': freq})

    spam_df = pd.DataFrame(spam_data)
    ham_df = pd.DataFrame(ham_data)

    return spam_df, ham_df

def statistics_to_csv(spam_df, ham_df, num):
    # Zapis df do .csv

    csv_spam_filename = STATS_DIR / f"stats_enron{num}_spam.csv"
    csv_ham_filename = STATS_DIR / f"stats_enron{num}_ham.csv"
    spam_df.to_csv(csv_spam_filename, index=False)
    ham_df.to_csv(csv_ham_filename, index=False)


# Paths
SRC_DIR = Path('./data/')
DATA_DIR = SRC_DIR / 'datasets'
STATS_DIR = SRC_DIR / 'statistics'

for num in range(1, 7):
    df = dataset_to_df(num)
    spam_df, ham_df = statistics(df)
    statistics_to_csv(spam_df, ham_df, num)
