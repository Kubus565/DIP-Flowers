# Flower Species Classifier - Color Feature Extraction & k-NN

Aplikacja w Pythonie, która klasyfikuje gatunki kwiatów na podstawie analizy rozkładu barw (przestrzeń kolorów HSV oraz YCbCr) przy użyciu algorytmu k-Najbliższych Sąsiadów (k-NN).
<img width="400" alt="image" src="https://github.com/user-attachments/assets/37600cc4-3c71-493b-b965-280806578d80" />

## 🧠 Metodologia i Algorytm
Projekt nie opiera się na prostym porównywaniu pikseli, lecz na **analizie histogramów barwy (Hue)**:
1. **Pre-processing:** Przetwarzanie obrazu wejściowego i ekstrakcja cech kolorystycznych.
2. **Feature Vector:** Tworzenie wektorów cech zapisanych w formacie `.csv`.
3. **Klasyfikacja:** Porównanie wektora badanego zdjęcia z bazą treningową przy użyciu k-NN w celu znalezienia najbardziej zbliżonego gatunku.

## 🚀 Instrukcja obsługi
1. Uruchom główny interfejs użytkownika: `python GUI.py`.
2. Wskaż ścieżkę do badanego zdjęcia (Przeglądaj).
3. Wybierz algorytm klasyfikacji (np. model Jakub).
4. Wskaż plik z bazą cech: `histogramyhue.csv` (wygenerowany po procesie treningu).
5. Otrzymaj wynik dopasowania do predefiniowanych gatunków.

## 🛠️ Stack Techniczny
- **Język:** Python
- **Biblioteki:** - `NumPy` & `Pandas` (operacje na macierzach i plikach CSV)
  - `Tkinter` (interfejs graficzny GUI)
  - `Scikit-learn` (implementacja k-NN)
