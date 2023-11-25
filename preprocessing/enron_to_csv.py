import os
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
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

def preprocess_text(text):
    # Usuwanie znaków specjalnych - uznano za ważne
    # text = text.translate(str.maketrans("", "", string.punctuation))

    # Usuwanie znaków akcentowanych języka - zaimplementowano w tfidf
    # text = unidecode(text)

    # Zmiana tekstu na tokeny
    tokens = word_tokenize(text)
    new_tokens = []
    num_tag = 'NUMBER'

    # Zmiana numerów na NUMBER, zmiana tekstu na małe litery, usunięcie stopwords (bez 're')
    for token in tokens:
        if token.isdigit():
            new_tokens.append('NUMBER')
        elif token.lower() not in email_stopword and token is not num_tag:
            token = token.lower()
            new_tokens.append(token)

    # Lemmatization
    pos_tags = nltk.pos_tag(new_tokens)
    lemma_tokens = []
    for token, pos_tag in pos_tags:
        pos_tag = get_pos(pos_tag)
        lemma_token = wnl.lemmatize(token, pos=pos_tag)
        lemma_tokens.append(lemma_token)

    text = ' '.join(lemma_tokens)

    '''
    Detokenizacja
    lemma_tokens = ['this', 'is', 'an', 'example', 'sentence', '.']
    Detokenizer: 'This is an example sentence.'
    Fn. join: 'this is an example sentence .
    '''
    # Detokenizacja detokenizerem
    # detokenizer = TreebankWordDetokenizer()
    # return detokenizer.detokenize(lemma)
    
    # Detokenizacja fn. join
    text = ' '.join(lemma_tokens)
    return text

  
for num in range(1, 7):

    #Ścieżka źródłowa zbioru
    SRC_DIR_PATH = Path(f'../Data/enron/enron{num}/')
    print(SRC_DIR_PATH)

    counter = 1
    data = []
    df = pd.DataFrame(columns=['id', 'class', 'label', 'subject', 'body'])

    for directory in os.listdir(SRC_DIR_PATH):
        # Iteracja po folderze spam/ham
        DIR_PATH = SRC_DIR_PATH / directory
        if os.path.isdir(DIR_PATH):
            file_class = directory # Przypisanie nazwy foldery spam/ham do zmiennej

            for filename in tqdm(os.listdir(DIR_PATH)):
                # Iteracja po plikach w folderze
                # print("file ", counter, ": ", filename)
                FILE_PATH = DIR_PATH / filename

                if os.path.isfile(FILE_PATH):
                    filename_id = filename.split('.')[0]
                    with open(FILE_PATH, 'r', encoding='ANSI') as file:
                        file.seek(9)
                        subject_text = file.readline()
                        body_text = file.read()

                        subject_text = preprocess_text(subject_text)
                        body_text = preprocess_text(body_text)
                        # unicodedata.normalize('NFKD', subject_text)
                    file.close()

                    # datasets_df = []
                    # mapping = {"spam": 1, "ham": 0}

                    # for num in range(1, 7):
                    #     datasets_df.append(pd.read_csv(os.path.join(DATA_DIR, f"dataset_enron{num}.csv")))
                    #     datasets_df[num-1]['class'] = datasets_df[num-1]['type'].map(mapping)

                    # Tekst do kolumn
                    data.append({'id': filename_id, 'class': file_class,
                                 'subject': subject_text, 'body': body_text})
                    counter += 1

    # Tworzenie dataframe, mapowanie, organizacja kolumn
    df = pd.DataFrame(data)
    mapping = {"spam": 1, "ham": 0}
    df['label'] = df['class'].map(mapping)
    df = df[['id', 'class', 'label', 'subject', 'body']]

    SRC_DIR = Path('./data/')
    RAW_DIR = SRC_DIR / 'raw datasets'
    DATA_DIR = SRC_DIR / 'datasets'

    # Zapis do .csv
    csv_filename = DATA_DIR / f"_dataset_enron{num}.csv"
    df.to_csv(csv_filename, index=False, escapechar='\\')
