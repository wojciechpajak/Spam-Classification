import os
import pandas as pd
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from unidecode import unidecode
import string
from tqdm import tqdm

def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

stopword = stopwords.words('english')
email_stopword = stopword.copy()
email_stopword.remove("re")
wnl = WordNetLemmatizer()

text = 'She has 42 apples..'
tokens = word_tokenize(text)
new_tokens = []
num_tag = 'NUMBER'

for token in tokens:
    if token.isdigit():
        new_tokens.append('NUMBER')
    elif token.lower() not in email_stopword and token is not num_tag:
        token = token.lower()
        new_tokens.append(token)

# Lemmatization
pos_tags = nltk.pos_tag(new_tokens)
print('TT: ', pos_tags)
lemma_tokens = []
for token, pos_tag in pos_tags:
    print(pos_tag)
    pos_tag = get_pos(pos_tag)
    print(pos_tag)
    lemma_token = wnl.lemmatize(token, pos=pos_tag)
    lemma_tokens.append(lemma_token)

print(lemma_tokens)
text = ' '.join(lemma_tokens)
print(text)
