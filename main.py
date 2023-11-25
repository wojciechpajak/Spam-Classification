from models.model_trainer import ModelTrainer

if __name__ == "__main__":
    trainer = ModelTrainer()
    
    # Ładowanie zbioru danych i wektoryzatora
    trainer.tfidf_load_vectorizer(filename='TFIDF_Vectorizer_MaxFeatures1000')
    # trainer.feature_extraction_split()  # Ekstrakcja cech oraz podział niestandardowy - trening na 80% danych, test na 20% danych
    # trainer.feature_extraction()        # Ekstrakcja cech oraz podział standartowy CV - trening na wszystkich danych zbioru enron 1-6
    # trainer.tfidf_save_vectorizer(filename='TFIDF_Vectorizer_MaxFeatures1000')


    # Ładowanie modelu
    trainer.load_model(filename='MNB_EnronAll_MaxFeatures1000')


    # Trening modelu
    # trainer.train_multinomial_nb_model()
    # trainer.train_svc_model()
    # trainer.train_mlp_model()


    # Wybór metody testowania
    # trainer.tts_test_model()
    # trainer.cv_test_model()


    # Zapis modelu
    # trainer.save_model(filename='SVC_Enron#num_MaxFeatures1000')


    # Test accuracy od progu na zbiorze SpamAssassin lub Enron 3 i 6
    # trainer.treshold_test('SpamAssassin')  # Enron36/SpamAssassin

