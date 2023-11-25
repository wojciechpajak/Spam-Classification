import sys
import os
from pathlib import Path
import shap
from joblib import dump, load
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

class ModelExplainer:
    def __init__(self, input, model_name):
        # Pre-processing
        self.stopword = stopwords.words('english')
        self.email_stopword = self.load_email_stopword()
        self.wnl = WordNetLemmatizer()

        # XAI
        shap.initjs()
        self.input = input
        self.X_batch = self.preprocess(input)
        self.SRC_DIR = Path('../data/')
        self.MODELS_DIR = self.SRC_DIR / "models"
        self.vectorizer = self.load_vectorizer()
        self.model = self.load_model(model_name)
        self.masker = shap.maskers.Text(tokenizer=r"\W+")
        self.explainer = shap.Explainer(self.f, self.masker)
        self.shap_values = self.explain_model()


    def load_vectorizer(self):
        vectorizer = load(f'{self.MODELS_DIR}/tfidf_vectorizer_maxfeatures1000.joblib')
        print(f'Załadowano wektoryzer')
        return vectorizer


    def load_model(self, filename):
        model = load(f'{self.MODELS_DIR}/{filename}.joblib')
        print(f'Załadowano: {filename}.joblib')
        return model

    def load_email_stopword(self):
        email_stopword = self.stopword.copy()
        email_stopword.remove("re")
        return email_stopword

    def preprocess(self, input):
        # Pre-processing tekstu tak samo jak pre-processing zbioru danych treningowych modelu - jednolitość ocenianych danych
        # Zmiana tekstu na tokeny, rozdzielanie liczb od znaków
        X_batch = []
        text = input[0]
        text = re.sub(r'(\d+)', r' \1 ', text)
        tokens = word_tokenize(text)
        new_tokens = []
        num_tag = 'NUMBER'

        # Zmiana numerów na NUMBER, zmiana tekstu na małe litery, usunięcie stopwords (bez 're')
        for token in tokens:
            if token.isdigit():
                new_tokens.append('NUMBER')
            elif token not in self.email_stopword and token is not num_tag:
                token.lower()
                new_tokens.append(token)

        # Lemantyzacja
        lemma_tokens = [self.wnl.lemmatize(token) for token in new_tokens]
        
        # Detokenizacja fn. join
        text = ' '.join(lemma_tokens)
        X_batch.append(text)
        print(text)
        return X_batch

    def f(self, inputs):
        X = self.vectorizer.transform(inputs).toarray()
        preds = self.model.predict(X)
        return preds


    def print_tokenized(self, X_batch):
        print("Samples : ")
        for tokens in X_batch:
            print(re.split(r"\W+", tokens))
            print()


    def explain_model(self):
        shap_values = self.explainer(self.X_batch)
        return shap_values
        
        
    def text_plot(self):
        shap.plots.text(self.shap_values)

    
    def bar_chart(self):
        shap.plots.bar(self.shap_values[0], max_display=15)

    
    def waterfall_plot(self):
        shap.plots.waterfall(self.shap_values[0], max_display=15)

    
    def get_model(self):
        return self.model


