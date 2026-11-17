# Projekt Zaliczeniowy: Podstawy Sieci Neuronowych

**Autor:** Bartosz Włodarczyk  
**Cel:** Realizacja kompletnego projektu zaliczeniowego na ocenę bardzo dobrą (`5.0`) poprzez wykonanie dwóch niezależnych zadań praktycznych (klasyfikacja danych tabelarycznych oraz klasyfikacja obrazów) wraz z analizą wyników i wnioskami.

---

## 1. Podjęte decyzje technologiczne i organizacyjne

* **Biblioteka do sieci neuronowych:** **Keras** (TensorFlow) – wybrany ze względu na czytelną, wysokopoziomową składnię oraz spójność kodu pomiędzy problemem tabelarycznym a wizyjnym.
* **Narzędzia pomocnicze:** `pandas`, `numpy` (przetwarzanie danych), `matplotlib`, `seaborn` (wizualizacje i statystyka), `scikit-learn` (przygotowanie danych, metryki, modele klasyczne).
* **Typ problemu (Zadanie na 4):** **Klasyfikacja** – pozwala na bezpośrednie i miarodajne porównanie sieci neuronowej z drzewem decyzyjnym oraz naiwnym klasyfikatorem Bayesa.
* **Kontrola wersji:** Lokalne repozytorium synchronizowane z platformą **GitHub** (duże zbiory zdjęć wykluczone za pomocą `.gitignore`).
* **Środowisko sprzętowe (praca hybrydowa):**
  * *Dell OptiPlex 7010 (i5-3470, 8 GB RAM):* pisanie kodu, analiza statystyczna (EDA) oraz realizacja Zadania na 4.
  * *Asus Zenbook 14 (Intel Core Ultra 5 225H):* praca mobilna, testy kodu, opracowanie dokumentacji.
  * *MSI Bravo 15 (NVIDIA RTX 4060 8 GB VRAM):* trenowanie głębokiej sieci konwolucyjnej (CNN) w Zadaniu na 5 z wykorzystaniem akceleracji GPU.

### Struktura repozytorium
```text
sieci-neuronowe-projekt/
├── data/
│   ├── tabular/
│   │   └── healthcare-dataset-stroke-data.csv
│   └── images/           # Katalog ze zdjęciami do Zadania na 5 (dodany do .gitignore)
├── notebooks/
│   ├── 01_stroke_prediction_mlp.ipynb
│   └── 02_image_classification_cnn.ipynb
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 2. Zadanie na 4: Klasyfikacja danych tabelarycznych

Pierwsza część projektu polega na rozwiązaniu problemu klasyfikacji na danych strukturalnych (tabelarycznych) z wykorzystaniem płytkiej sieci neuronowej oraz porównaniu jej skuteczności z klasycznymi algorytmami uczenia maszynowego.

### Oryginalne wymagania i ich praktyczna interpretacja
* **„ciekawy problem rozwiązany przy pomocy płytkich sieci neuronowych”** – należy zbudować klasyczny wielowarstwowy perceptron (MLP) składający się z warstwy wejściowej, 1–2 warstw ukrytych oraz warstwy wyjściowej. Problem powinien być niebanalny (unikamy oklepanych zbiorów dydaktycznych, takich jak Titanic czy Iris).
* **„poprzedzony statystyczną analizą danych”** – przed rozpoczęciem trenowania sieci konieczne jest wykonanie eksploracyjnej analizy danych (EDA): obliczenie podstawowych miar statystycznych (średnia, mediana, odchylenie standardowe, kwartyle), wykrycie braków danych i wartości odstających, sprawdzenie zbalansowania klas oraz narysowanie macierzy korelacji i histogramów.
* **„problem odpowiedni dla sieci płytkich - taki tabelkowy”** – dane wejściowe muszą mieć postać klasycznej tabeli (plik CSV/Excel), gdzie wiersze oznaczają pojedyncze obserwacje (np. klientów, pacjentów), a kolumny opisują ich konkretne cechy numeryczne lub kategoryczne.
* **„jeśli się da - to porównanie z rozwiązaniem drzewami decyzyjnymi i/lub naiwnym Bayesem”** – ponieważ wybraliśmy problem klasyfikacji, na dokładnie tym samym podziale danych (trening/test) należy wytrenować drzewo decyzyjne (`DecisionTreeClassifier`) oraz naiwny klasyfikator Bayesa (`GaussianNB`), a następnie zestawić ich wyniki z siecią neuronową w jednej tabeli porównawczej.
* **„sprawozdanie z przeprowadzonego uczenia / prezentacja + wnioski + kod”** – oprócz działającego notatnika `.ipynb` wymagana jest dokumentacja prezentująca wykresy uczenia, tabele metryk oraz autorskie wnioski (np. ocena, czy prosta sieć neuronowa faktycznie radzi sobie lepiej od drzewa decyzyjnego na danych tabelarycznych).
* **„w grupach góra-cy / ale przesyła jedna osoba (wszystkie nazwiska w sprawozdaniu)”** – projekt można realizować maksymalnie w 2 osoby (lub samodzielnie); w przypadku pracy w parze komplet plików wysyła na platformę tylko jedna osoba, ale dokumentacja musi zawierać dane wszystkich autorów.

### Specyfikacja zbioru danych (Zadanie na 4)
* **Wybrany zbiór danych:** Stroke Prediction Dataset (`data/tabular/healthcare-dataset-stroke-data.csv`)
* **Źródło danych:** Kaggle Datasets (5110 rekordów, 12 kolumn)
* **Zmienna docelowa (target):** `stroke` – zmienna binarna określająca, czy pacjent przeszedł udar mózgu (`1` – tak, `0` – nie).
* **Opis cech wejściowych:**
  * *Cechy numeryczne:* `age` (wiek), `avg_glucose_level` (średni poziom glukozy we krwi), `bmi` (wskaźnik masy ciała).
  * *Cechy binarne:* `hypertension` (nadciśnienie tętnicze: 0/1), `heart_disease` (choroby serca: 0/1).
  * *Cechy kategoryczne (tekstowe):* `gender` (płeć), `ever_married` (stan cywilny), `work_type` (rodzaj zatrudnienia), `Residence_type` (miejsce zamieszkania: miasto/wieś), `smoking_status` (status palenia tytoniu).
  * *Kolumny techniczne:* `id` (unikalny identyfikator pacjenta – do usunięcia przed trenowaniem modeli).
* **Zidentyfikowane wyzwania w danych (do uwzględnienia w analizie i sprawozdaniu):**
  * Braki danych (`N/A`) w kolumnie `bmi` – wymagają uzupełnienia (np. medianą) lub odpowiedniej imputacji.
  * Kategoria `Unknown` w kolumnie `smoking_status`.
  * Silne niezbalansowanie klas w zmiennej `stroke` (ok. 5% przypadków pozytywnych vs 95% negatywnych) – wymaga zastosowania podziału stratyfikowanego (`stratify`), wag klas (`class_weight='balanced'`) oraz oceny modeli za pomocą metryk *Precision*, *Recall* i *F1-score*, a nie samego *Accuracy*.

### Checklista zadań (Zadanie na 4)
- [ ] **Statystyczna analiza danych (EDA):**
  - [ ] Sprawdzenie wymiarów zbioru (liczba wierszy i kolumn), typów zmiennych oraz duplikatów.
  - [ ] Identyfikacja brakujących wartości (`N/A` w `bmi`) oraz wartości odstających (*outliers*).
  - [ ] Wyznaczenie podstawowych statystyk opisowych (średnia, mediana, odchylenie standardowe, kwartyle).
  - [ ] Wizualizacja rozkładu klas (sprawdzenie zbalansowania zbioru), histogramy cech oraz macierz korelacji (*heatmap*).
- [ ] **Przygotowanie danych (Preprocessing):**
  - [ ] Usunięcie kolumny `id` oraz uzupełnienie braków w kolumnie `bmi` medianą.
  - [ ] Kodowanie zmiennych kategorycznych (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`) na postać numeryczną (*One-Hot Encoding*).
  - [ ] Podział zbioru na część treningową (80%) i testową (20%) z zachowaniem proporcji klas (`stratify`).
  - [ ] Standaryzacja cech numerycznych (`StandardScaler`) dopasowana wyłącznie do zbioru treningowego.
- [ ] **Budowa i trening płytkiej sieci neuronowej (MLP):**
  - [ ] Zaprojektowanie architektury w Keras (warstwa wejściowa, 1–2 warstwy ukryte z funkcją aktywacji `ReLU`, warstwa wyjściowa z funkcją `Sigmoid`).
  - [ ] Trening modelu (optymalizator `Adam`, uwzględnienie niezbalansowania klas) z wydzieleniem zbioru walidacyjnego oraz zapisaniem historii uczenia (wykresy *Loss* i *Accuracy*).
- [ ] **Modele porównawcze i ewaluacja:**
  - [ ] Wytrenowanie na tych samych danych **drzewa decyzyjnego** (`DecisionTreeClassifier`) oraz **naiwnego klasyfikatora Bayesa** (`GaussianNB`).
  - [ ] Zestawienie wszystkich trzech modeli według metryk: *Accuracy*, *Precision*, *Recall* oraz *F1-score*.
  - [ ] Sformułowanie wniosków dotyczących skuteczności płytkiej sieci względem klasycznych algorytmów na danych tabelarycznych.

---

## 3. Zadanie na 5: Klasyfikacja obrazów za pomocą sieci CNN

Druga część projektu rozszerza zakres prac o przetwarzanie obrazów z wykorzystaniem niezbyt głębokiej konwolucyjnej sieci neuronowej (CNN).

### Oryginalne wymagania i ich praktyczna interpretacja
* **„to, co na 4 plus...”** – warunkiem koniecznym uzyskania oceny 5.0 jest pełne i poprawne zrealizowanie wszystkich punktów z Zadania na 4 (dane tabelaryczne + EDA + sieć płytka + porównanie z drzewem/Bayesem).
* **„...ciekawy problem rozwiązany przy pomocy sieci głębokich”** – należy zastosować konwolucyjną sieć neuronową (CNN), zaprojektowaną specjalnie do automatycznego wykrywania wzorców przestrzennych (krawędzi, tekstur, kształtów) na danych obrazowych.
* **„ale nie za głębokich”** – nie należy używać ogromnych, gotowych architektur typu ResNet-152 czy VGG-19. Wystarczy własna, lekka i przejrzysta sieć zbudowana z 3–4 bloków konwolucyjnych (`Conv2D` + `MaxPooling2D`), którą można wytrenować w rozsądnym czasie i łatwo przeanalizować.
* **„coś, czego nie było na zajęciach”** – zbiór obrazów musi dotyczyć innego zagadnienia niż przykłady przerabiane podczas laboratoriów (wykluczamy m.in. rozpoznawanie odręcznych cyfr MNIST, ubrań Fashion-MNIST oraz klasyczny zbiór psów i kotów).
* **„coś podobnego do piesków-kotków”** – zadanie ma mieć formę klasyfikacji obrazów na 2 lub kilka rozłącznych kategorii na podstawie folderów ze zdjęciami (np. rozpoznawanie gatunków roślin, typów odpadów, warunków pogodowych lub zmian na zdjęciach medycznych).
* **„sprawozdanie z przeprowadzonego uczenia / prezentacja + wnioski + kod”** – konieczne jest przedstawienie przebiegu uczenia na wykresach (*Loss* i *Accuracy* dla zbioru treningowego i walidacyjnego), pokazanie macierzy pomyłek oraz wyciągnięcie wniosków dotyczących zjawiska przeuczenia (*overfitting*).
* **„w grupach góra-cy / ale przesyła jedna osoba (wszystkie nazwiska w sprawozdaniu)”** – obowiązują te same zasady organizacyjne i formalne przy oddawaniu pracy co w Zadaniu na 4.

### Specyfikacja zbioru danych (Zadanie na 5)
* **Wybrany zbiór danych:** `[DO UZUPEŁNIENIA – np. klasyfikacja odpadów / zdjęcia medyczne / pogoda]`
* **Źródło danych:** `[DO UZUPEŁNIENIA – link do zbioru]`
* **Liczba i nazwy klas:** `[DO UZUPEŁNIENIA]`

### Checklista zadań (Zadanie na 5)
- [ ] **Przygotowanie i eksploracja zbioru obrazów:**
  - [ ] Weryfikacja liczebności zdjęć w poszczególnych folderach (klasach) i prezentacja przykładowych obrazów.
  - [ ] Ujednolicenie rozdzielczości obrazów (np. skalowanie do `128x128` pikseli) oraz normalizacja wartości pikseli do przedziału `[0, 1]`.
  - [ ] Podział danych na zbiór treningowy, walidacyjny oraz testowy.
- [ ] **Budowa i trening niezbyt głębokiej sieci CNN:**
  - [ ] Zaprojektowanie własnej architektury w Keras składającej się z 3–4 bloków konwolucyjnych (`Conv2D` + `ReLU` + `MaxPooling2D`), warstwy spłaszczającej (`Flatten`), warstwy gęstej (`Dense`) oraz warstwy wyjściowej.
  - [ ] Uruchomienie procesu uczenia na karcie graficznej NVIDIA RTX 4060.
- [ ] **Analiza wyników i diagnostyka modelu:**
  - [ ] Wygenerowanie krzywych uczenia (*Loss* oraz *Accuracy* dla zbioru treningowego i walidacyjnego) w celu oceny zjawiska przeuczenia (*overfitting*).
  - [ ] Wyznaczenie macierzy pomyłek (*Confusion Matrix*) oraz raportu klasyfikacji na zbiorze testowym.
  - [ ] Sformułowanie wniosków na temat działania ekstrakcji cech w warstwach konwolucyjnych oraz skuteczności przyjętej architektury.

---

## 4. Harmonogram i status prac

| Etap | Opis zadania | Status |
| :---: | :--- | :--- |
| **1** | Wybór narzędzi, środowiska sprzętowego i założenie repozytorium GitHub | ✅ Zakończony |
| **2** | Wybór i pobranie zbioru tabelarycznego do Zadania na 4 (`healthcare-dataset-stroke-data.csv`) | ✅ Zakończony |
| **3** | Analiza statystyczna (EDA), preprocessing i trening modeli (Zadanie na 4) | 🔄 W trakcie |
| **4** | Wybór zbioru obrazów i trening sieci CNN na GPU (Zadanie na 5) | ⏳ Do zrobienia |
| **5** | Opracowanie końcowego sprawozdania/prezentacji z wnioskami i kodem | ⏳ Do zrobienia |

---

## 5. Wymagania formalne przed oddaniem projektu

- [ ] Przygotowanie kompletnego sprawozdania lub prezentacji zawierającej opis obu zadań, wykresy, tabele porównawcze oraz wnioski.
- [ ] Dołączenie uporządkowanego i skomentowanego kodu źródłowego (notatniki `.ipynb`).
- [ ] Weryfikacja danych autora (Bartosz Włodarczyk) w sprawozdaniu oraz przesłanie kompletnej paczki projektowej na platformę uczelnianą.