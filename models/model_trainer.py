import sys
print("Python version: ", sys.version)
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from joblib import dump, load
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


class ModelTrainer:
    def __init__(self):
        self.SRC_DIR = Path('./data/')
        self.DATA_DIR = self.SRC_DIR / 'datasets'
        self.MODELS_DIR = self.SRC_DIR / 'models'
        self.PLOTS_DIR = self.SRC_DIR / 'plots'
        self.datasets_df_list = self.load_df()

        self.corpus, self.labels = self.load_full_corpus()
        self.test_corpus, self.test_labels = self.load_25_corpus()
        self.corpus_500, self.labels_500 = self.load_500_corpus()
        self.corpus_sa, self.labels_sa = self.load_sa_corpus()

        self.vectorizer = None
        self.X_test, self.X_train, self.y_test, self.y_train = None, None, None, None
        self.model = None
        self.model_name = None



    def load_df(self):
        datasets_df_list = []
        for num in range(1, 7):
            datasets_df_list.append(pd.read_csv(os.path.join(self.DATA_DIR, f"dataset_enron{num}.csv")))
        return datasets_df_list


    def load_full_corpus(self):
        corpus = []
        labels = []

        for i in range(0, 6):
            print(f'Load full corpus enron{i+1}:')
            body_column = self.datasets_df_list[i]["body"]
            label_column = self.datasets_df_list[i]["label"]

            for body, label in tqdm(zip(body_column, label_column), total=len(self.datasets_df_list[i])):
                corpus.append(str(body))
                labels.append(int(label))

        return corpus, labels


    def load_25_corpus(self):
        test_corpus = []
        test_labels = []

        for i in 2, 5:
            print(f'Load corpus 3 and 6 enron{i+1}:')
            body_column = self.datasets_df_list[i]["body"]
            label_column = self.datasets_df_list[i]["label"]

            for body, label in tqdm(zip(body_column, label_column), total=len(self.datasets_df_list[i])):
                test_corpus.append(str(body))
                test_labels.append(int(label))
        
        return test_corpus, test_labels


    def load_500_corpus(self):
        corpus_500 = []
        labels_500 = []

        for i in range(1):
            body_column = self.datasets_df_list[i]["body"]
            label_column = self.datasets_df_list[i]["label"]

            # Pętla iterująca po zakresie od 0 do 499 i od 3674 do 4173
            for j in range(500):
                # Dodaj wiadomość i etykietę do corpus i labels
                corpus_500.append(str(body_column[j]))
                labels_500.append(int(label_column[j]))

            for j in range(3674, 3674+500):
                # Dodaj wiadomość i etykietę do corpus i labels
                corpus_500.append(str(body_column[j]))
                labels_500.append(int(label_column[j]))

        return corpus_500, labels_500


    def load_sa_corpus(self):
        dataset_sa_df = pd.read_csv(os.path.join(self.DATA_DIR, f"dataset_spamassassin.csv"))
        corpus_sa = []
        labels_sa = []

        print(f'Load corpus sa:')
        body_column = dataset_sa_df["body"]
        label_column = dataset_sa_df["label"]
        for body, label in tqdm(zip(body_column, label_column), total=len(dataset_sa_df)):
            corpus_sa.append(str(body))
            labels_sa.append(int(label))
        
        return corpus_sa, labels_sa


    def feature_extraction_split(self):
        X = self.tfidf_vectorization()
        y = self.labels

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)


    def feature_extraction(self):
        self.X_train = self.tfidf_vectorization()
        self.y_train = self.labels


    def tfidf_vectorization(self):
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(strip_accents='ascii', max_features=1000)
            X = self.vectorizer.fit_transform(self.corpus)
            self.tfidf_save_vectorizer()
        else:
            X = self.vectorizer.transform(self.corpus)

        return X


    def tfidf_load_vectorizer(self, filename):
        self.vectorizer = load(f'{self.MODELS_DIR}/{filename}.joblib')
        print(f'Załadowano wektoryzer {self.MODELS_DIR}\{filename}.joblib')
    

    def tfidf_save_vectorizer(self, filename):
        dump(self.vectorizer, f'{self.MODELS_DIR}/{filename}.joblib')
        print(f'Zapisano wektoryzer {self.MODELS_DIR}\{filename}.joblib')


    def train_multinomial_nb_model(self):
        self.model = MultinomialNB(force_alpha=True).fit(self.X_train, self.y_train)

        
    def train_svc_model(self):
        self.model = SVC(kernel='linear', C=1, verbose=True, probability=True, random_state=42).fit(self.X_train, self.y_train)
        

    def train_mlp_model(self):
        self.model = MLPClassifier(verbose=1, random_state=42).fit(self.X_train, self.y_train)


    def tts_test_model(self):
        train_preds = self.model.predict(self.X_train)
        test_preds = self.model.predict(self.X_test)

        print(f"Train Accuracy : {accuracy_score(self.y_train, train_preds):.4f}")
        print(f"Test  Accuracy : {accuracy_score(self.y_test, test_preds):.4f}")
        print("\nClassification Report : ")
        print(classification_report(self.y_test, test_preds))


    def cv_test_model(self):
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        cv_results = cross_validate(self.model, self.X_train, self.y_train, cv=10, scoring=scoring)

        print(f"Test Accuracy  : {cv_results['test_accuracy'].mean():.4f}")
        print(f"Test Precision : {cv_results['test_precision'].mean():.4f}")
        print(f"Test Recall    : {cv_results['test_recall'].mean():.4f}")
        print(f"Test f1        : {cv_results['test_f1'].mean():.4f}")
        print(f"Fit time       : {cv_results['fit_time'].mean():.4f}")
        print(f"Std deviation  : {cv_results['test_accuracy'].std():.2f}")


    def save_model(self, filename):
        dump(self.model, f'{self.MODELS_DIR}/{filename}.joblib')
        print(f'Zapisano: {filename}.joblib')


    def load_model(self, filename):
        self.model = load(f'{self.MODELS_DIR}/{filename}.joblib')
        self.model_name = filename
        print(f'Załadowano: {filename}.joblib')

    
    def get_model(self):
        return self.model


    def treshold_test(self, corpus_name):
        
        acc_list = [] # Lista dokładności całego zbioru
        ham_acc_list = []  # Lista dokładności dla "ham"
        spam_acc_list = []  # Lista dokładności dla "spam"
        threshold_values = []  # Lista wartości progu

        if corpus_name == 'Enron36':    # enron3 i enron6
            corpus = self.test_corpus
            labels = self.test_labels
        elif corpus_name == 'SpamAssassin':
            corpus = self.corpus_sa
            labels = self.labels_sa
      
        X = self.vectorizer.transform(corpus)

        '''### Bez thresholda
        predicted_labels = model.predict(vec_text)

        diff = 0
        for label, pred in tqdm(zip(labels, predicted_labels), total=len(labels)):
            if label != pred:
                diff += 1
        length = len(labels)
        acc = (1 - diff / length) * 100

        print(f"\nEtykiety faktyczne: {' '.join(map(str, labels[:50]))}")
        print(f"Etykiety predykcja: {' '.join(map(str, predicted_labels[:50]))}")
        print(f"\nEtykiety faktyczne: {' '.join(map(str, labels[2661:2711]))}")
        print(f"Etykiety predykcja: {' '.join(map(str, predicted_labels[2661:2711]))}")
        print(f"Accuracy: {acc:.2f}%")'''

        #Z thresholdem, przeszukiwanie po różnych wartościach progu
        for i in tqdm(range(50, 100, 1), total=(100-50/1)):
            value = i / 100.0
            threshold = value
            predicted_probabilities = self.model.predict_proba(X)
            predicted_labels = (predicted_probabilities[:, 1] > threshold).astype(int)

            acc_diff = 0
            ham_diff = 0
            spam_diff = 0
            acc_total = 0
            ham_total = 0
            spam_total = 0

            for label, pred in zip(labels, predicted_labels):
                acc_total += 1
                if label != pred:
                    acc_diff += 1
                if label == 0:  # "ham"
                    ham_total += 1
                    if label != pred:
                        ham_diff += 1
                elif label == 1:  # "spam"
                    spam_total += 1
                    if label != pred:
                        spam_diff += 1
            length = len(labels)

            acc = (1 - acc_diff / acc_total) * 100
            ham_acc = (1 - ham_diff / ham_total) * 100 if ham_total > 0 else 100
            spam_acc = (1 - spam_diff / spam_total) * 100 if spam_total > 0 else 100

            ham_acc_list.append(round(ham_acc, 2))
            spam_acc_list.append(round(spam_acc, 2))
            acc_list.append(round(acc, 2))
            threshold_values.append(value)

            # print(f"\nEtykiety faktyczne: {' '.join(map(str, labels[:50]))}")
            # print(f"Etykiety predykcja: {' '.join(map(str, predicted_labels[:50]))}")
            # print(f"\nEtykiety faktyczne: {' '.join(map(str, labels[2661:2711]))}")
            # print(f"Etykiety predykcja: {' '.join(map(str, predicted_labels[2661:2711]))}")
            # print(f"Accuracy: {acc:.2f}%")
            # print(f"Accuracy (ham): {ham_acc:.2f}%")
            # print(f"Accuracy (spam): {spam_acc:.2f}%")

        # print(acc_list)

        # Utworzenie wykresu
        filename = self.PLOTS_DIR / f'plot_{corpus_name}_{self.model_name}'
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_values, ham_acc_list, label='Accuracy (ham)', marker='.')
        plt.plot(threshold_values, spam_acc_list, label='Accuracy (spam)', marker='.')
        plt.plot(threshold_values, acc_list, label='Accuracy', marker='.')

        plt.xlabel('Próg klasyfikacji (threshold)')
        plt.ylabel('Dokładność (accuracy)')
        plt.title(f'Zależność dokładności od progu klasyfikacji — Model {self.model_name}')
        plt.grid(True)
        plt.legend()
        plt.savefig(filename)
        plt.show()
