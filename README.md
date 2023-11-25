# Spam-Classification

Projekt implementacji podsystemu [aplikacji webowej](https://github.com/wojciechpajak/Spam-Detection-Web-App) zintegrowanej z modelami uczenia maszynowego do klasyfikacji spamu. System dedykowany jest do budowania trzech modeli klasyfikacyjnych: MNB (Multinomial Naive Bayes), SVC (Support Vector Classifier) oraz MLP (Multi-layer Perceptron).

System obejmuje proces budowania modeli uczenia maszynowego, przygotowanie oraz przetwarzanie danych treningowych, opracowanie potoku wstępnego przetwarzania danych, a także implementację techniki XAI.

## Uruchomienie

W projekcie znajdują się pliki:

- W folderze `preprocessing`: `enron_to_csv.py` i `spamassassin_to_csv.py` przetwarzają surowe dane w potoku przetwarzania danych do plików danych treningowych. `text_statistics_to_csv.py` przeprowadza statystykę słów w podzbiorach danych.
- W folderze `models`: `model_trainer.py` oraz `model_explainer.py` zawierają klasy odpowiadające budowaniu modeli uczenia maszynowego, oraz wyjaśnianiu decyzji.
- W folderze `visualization` oraz `models`: `data_overview.ipynb` oraz `model_xai.ipynb` służą do wizualizacji odpowiednio pre-processingu, oraz XAI (Explainable AI).


Aby przeprowadzić proces budowania modeli, należy użyć pliku `main.py`. Plik ten zawiera wywołania odpowiednich metod, które można wybrać odkomentowując odpowiednie linie kodu.

## Wymagania

W celu korzystania z funkcjonalności systemu należy zainstalować wymagane bibioteki za pomocą poniższej komendy:

```bash
pip install -r requirements.txt
```

Aby korzystać z pełnych funkcjonalności systemu, w tym z tworzenia zbiorów danych, należy w folderze projektu umieścić niniejsze repozytorium oraz repozytorium [Spam-Detection-Data](https://github.com/wojciechpajak/Spam-Classification-Data) (pod nazwą "Data").

*Wojciech Pająk 2023 Politechnika Wrocławska*