# multilingual-sentiment-analyzer
# Wielojęzyczny Analizator Sentymentu – System Ciągłej Oceny Satysfakcji

## Opis projektu

Projekt stanowi zaawansowany system analityczny NLP, który wykorzystuje nowoczesne metody uczenia głębokiego oraz modele regresyjne do precyzyjnego badania emocjonalnego nacechowania wypowiedzi[cite: 6]. W przeciwieństwie do klasycznego, uproszczonego podejścia zero-jedynkowego, system dokonuje płynnej predykcji stopnia satysfakcji użytkownika w ciągłej skali od 0.0 do 10.0[cite: 6].

Głównym celem systemu jest wsparcie procesów analitycznych w obszarze opinii konsumenckich oraz automatyzacja masowego przetwarzania tekstów[cite: 6]. Dzięki zastosowaniu zaawansowanych transformerów (Cross-Lingual Transfer), model doskonale interpretuje teksty w języku polskim, korzystając z wiedzy zdobytej na potężnych bazach anglojęzycznych[cite: 6].

## Źródło danych

Modele analityczne w tym projekcie zostały wytrenowane i zweryfikowane w oparciu o uznany, globalny zbiór recenzji filmowych **IMDB Dataset** pobrany z platformy Kaggle, zawierający pierwotnie 50 000 rekordów[cite: 6]. Do ostatecznej ekstrakcji cech semantycznych wykorzystano jego oczyszczoną i zbalansowaną strukturę[cite: 6].

## Główne funkcjonalności

*   **Płynna predykcja sentymentu:** Dokładna ocena poziomu emocjonalnego wypowiedzi w biznesowej skali 0-10[cite: 6].
*   **Analiza wielojęzyczna (Multilingual):** Natywne wsparcie dla języka polskiego i ponad 100 innych języków dzięki zastosowaniu reprezentacji wektorowych modelu BGE-M3[cite: 6].
*   **Masowe przetwarzanie (Batch Processing):** Możliwość wgrywania całych plików tekstowych i stabilnego procesowania ich w zoptymalizowanych paczkach obliczeniowych[cite: 6].
*   **Automatyczny moduł translacji:** Opcjonalne tłumaczenie analizowanych opinii na język angielski w czasie rzeczywistym przy użyciu API Google Translate[cite: 2, 6].
*   **Generowanie raportów biznesowych:** Eksport gotowych wyników analizy statystycznej do formatów **Excel (.xlsx)**, **HTML**, **TXT** oraz **CSV**[cite: 2, 6].

## Interfejs użytkownika

W ramach projektu zaprojektowano nowoczesny, responsywny i w pełni interaktywny interfejs webowy (oparty o Streamlit), który podzielono na dwa główne ekrany robocze: szybką ocenę pojedynczej opinii wpisanej z klawiatury oraz panel do masowej analityki plików .txt wraz z wizualizacją rozkładu statystycznego nastrojów[cite: 2, 6].

## Zastosowane technologie

*   **Język programowania:** Python[cite: 6]
*   **Biblioteki danych:** Pandas, NumPy[cite: 6]
*   **Uczenie głębokie (NLP):** PyTorch, Sentence-Transformers (model `BAAI/bge-m3`)[cite: 6]
*   **Uczenie maszynowe:** Scikit-learn (Regresor Ridge)[cite: 4, 6]
*   **Interfejs webowy:** Streamlit[cite: 6]
*   **Zasoby dodatkowe:** Deep-translator, Matplotlib, Joblib, Openpyxl[cite: 6]

## Struktura plików w repozytorium

*   `app.py` – Główny skrypt aplikacji webowej odpowiedzialny za interfejs użytkownika, logikę prezentacji danych oraz generowanie wykresów i raportów[cite: 2, 6].
*   `model.py` – Klasa produkcyjna (`BGERidgePredictor`) stanowiąca rdzeń obliczeniowy AI; zarządza pamięcią enkodera oraz wykonuje predykcję seryjną (batching)[cite: 2, 3, 6].
*   `regresor_ridge_bge.joblib` – Zamrożony, zserializowany plik binarny wytrenowanego modelu regresji Ridge (Scikit-Learn) przechowujący wyznaczone wagi matematyczne[cite: 4, 6].
*   `cleaned_imdb_data.csv` – Oczyszczona i przygotowana baza danych wykorzystywana w procesie eksperymentalnym i ewaluacji algorytmów.
*   `requirements.txt` – Wykaz wszystkich bibliotek i zależności projektowych niezbędnych do poprawnego uruchomienia środowiska[cite: 5].
*   `.gitignore` – Plik konfiguracyjny definiujący zasoby wykluczone z systemu kontroli wersji Git (np. foldery pamięci podręcznej)[cite: 2].
*   `Dokumentacja_Techniczna_Projektu_Sentymentu (1).docx` – Pełna, szczegółowa dokumentacja inżynierska obejmująca architekturę systemu, wymagania oraz scenariusze użycia[cite: 6].
