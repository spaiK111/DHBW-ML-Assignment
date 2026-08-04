# Fetal Health Classification: Systematischer Modellvergleich

Assignment „Systematischer Modellvergleich mit klassischen Modellen und neuronalen Netzen,
Mehrklassen-Klassifikation“
**Gruppe C (Gruppe 3)** · Kurs *Maschinelles Lernen mit Python, scikit-learn und Keras* ·
DHBW Stuttgart · SS 2026

---

## 1. Worum geht es?

Aus Kardiotokogramm-Daten (CTG) soll der Zustand eines Fötus in drei Klassen eingestuft werden.
Ein CTG zeichnet die fetale Herzfrequenz und die Wehentätigkeit auf; daraus werden 21 numerische
Kennwerte abgeleitet, nämlich Baseline-Herzfrequenz, Akzelerationen, Dezelerationen,
Variabilitätsmaße und Histogramm-Statistiken.

| | |
|---|---|
| Aufgabentyp | Mehrklassen-Klassifikation (3 Klassen) |
| Zielvariable | `fetal_health`: 1 = Normal, 2 = Suspect, 3 = Pathological |
| Umfang | 2.126 Samples, 21 Features (alle numerisch) |
| Klassenverteilung | 1.655 (77,85 %) / 295 (13,88 %) / 176 (8,28 %), stark imbalanciert |
| Quelle | Kaggle: [andrewmvd/fetal-health-classification](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) |

Verglichen werden fünf Modelle: Logistic Regression (multinomial), ein Baumverfahren (Decision
Tree, Random Forest oder XGBoost), k-NN oder SVM sowie zwei MLP-Varianten in Keras.

Fachlich entscheidend ist dabei: Weil ein übersehener pathologischer Fall die teuerste
Fehlklassifikation ist, bewerten wir nicht nur die Accuracy, sondern vor allem den F1-Score
(weighted) und den Recall der Klasse Pathological. Eine naive Vorhersage „immer Normal“ erreicht
bereits 77,9 % Accuracy, daran muss sich jedes Modell messen lassen.

---

## 2. Projektstruktur

```
DHBW-ML-Assignment/
├── README.md                               ← diese Datei
├── Management_Summary.md                   ← Abgabe-Deliverable: Zusammenfassung der Ergebnisse
├── ADR.md                                  ← alle Projektentscheidungen mit Begründung
├── Todo.txt                                ← interne Restpunkte der Gruppe
├── Context/                                ← Aufgabenstellung und Kursmaterial (kein Code)
│   ├── Assignment.pdf
│   ├── Anweisungen.md
│   ├── Script.pdf
│   └── Notebook-LM-*/                      ← Übungsnotebooks aus der Vorlesung
└── src/                                    ← gesamte Implementierung
    ├── requirements.txt                    ← Bibliotheksversionen
    ├── fetal_health.csv                    ← Datensatz (liegt im Repo)
    └── fetal_health_modellvergleich.ipynb  ← Haupt-Notebook (alle 7 Schritte)
```

Der Ordner `Context/` dient ausschließlich als Kontext und enthält keinen Projektcode.

---

## 3. Einrichtung und Ausführung

### Variante A: Google Colab (empfohlen für die Abgabe)

1. `fetal_health_modellvergleich.ipynb` in Colab öffnen (*Datei → Notebook hochladen*).
2. Ab Schritt 4 (Keras): *Laufzeit → Laufzeittyp ändern → T4 GPU*.
3. *Laufzeit → Alle ausführen*.

**Ein manueller Upload des Datensatzes ist nicht nötig.** Das Notebook lädt die Daten
zweistufig (→ ADR-001):

1. Liegt `fetal_health.csv` neben dem Notebook, wird diese lokale Kopie verwendet.
2. Andernfalls wird der Datensatz über `kagglehub` direkt von Kaggle geladen
   (`andrewmvd/fetal-health-classification`, die im Assignment genannte Quelle).

Beide Wege wurden geprüft und liefern identische Daten (2.126 × 22). Alle im Notebook
verwendeten Bibliotheken inklusive `kagglehub` sind in Colab vorinstalliert, eine
Nachinstallation ist nicht nötig. Wer lieber mit der lokalen Kopie arbeitet, lädt
`fetal_health.csv` über die linke Seitenleiste (Ordnersymbol, *Hochladen*) hoch – sie hat
dann Vorrang.

### Variante B: Lokal

```bash
# im Ordner src/
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook fetal_health_modellvergleich.ipynb
```

Getestet mit Python 3.12. Das Notebook sollte aus dem Ordner `src/` heraus gestartet werden,
damit der relative Pfad zu `fetal_health.csv` aufgeht. Wird es aus einem anderen Verzeichnis
gestartet, greift automatisch der Kaggle-Fallback (`kagglehub` erforderlich).

---

## 4. Code-Konventionen

Damit der Code über alle sieben Schritte hinweg gleich aussieht, gilt im Notebook durchgehend:

| Regel | Beispiel |
|---|---|
| Bezeichner auf Englisch, wie in sklearn, Keras und den Kursnotebooks | `models`, `history_a`, `final_model` |
| Kommentare, Docstrings und Markdown auf Deutsch | `# Kontrolle: Größen und Klassenanteile je Teilmenge` |
| `snake_case` für Variablen und Funktionen | `plot_learning_curves`, `evaluate_predictions` |
| `UPPER_SNAKE_CASE` für Konstanten, die einmal gesetzt werden | `RANDOM_STATE`, `TARGET_NAMES`, `TOLERANCE` |
| Großes `X` für Feature-Matrizen, kleines `y` für Zielvektoren (sklearn-Konvention) | `X_train`, `y_val` |
| Suffixe beschreiben die Teilmenge bzw. die Aufbereitung | `_train`, `_val`, `_test`, `_scaled` |
| Modelle, Trainingsverläufe und Vorhersagen tragen sprechende Präfixe | `model_b_opt`, `history_b_opt`, `y_pred_b_opt` |
| Keine Umschreibungen von Umlauten in Bezeichnern oder Kommentaren | `chosen` statt `gewaehlt`, „für“ statt „fuer“ |

Der Grund für die Zweisprachigkeit: Bezeichner stehen direkt neben sklearn- und Keras-Aufrufen und
bleiben so einheitlich lesbar, während die Erklärungen in der Sprache der Abgabe bleiben.

---

## 5. Aufbau des Notebooks

Das Notebook folgt exakt den sieben Schritten des Assignments. Jede Codezelle ist von einer
Markdown-Zelle mit Begründung oder Interpretation begleitet.

| Schritt | Inhalt | Status |
|---|---|---|
| **1** | Datenanalyse: Struktur, Datenqualität, Klassenverteilung, 4 Visualisierungen, Feature-Relevanz | ✅ fertig |
| **2** | Preprocessing: Duplikate, Zielkodierung, stratifizierter 70/15/15-Split, Skalierung | ✅ fertig |
| **3** | Drei klassische Modelle mit Default-Parametern, 5-fache stratifizierte Kreuzvalidierung | ✅ fertig |
| **4** | MLP Variante A: `Dense(64, relu) → Dense(3, softmax)`, Adam, Early Stopping | ✅ fertig |
| **5** | MLP Variante B: zusätzlich `Dropout(0.3)` und `Dense(32, relu)`, SGD | ✅ fertig |
| **6** | Vergleichstabelle, Hyperparameter-Optimierung, finale Evaluation auf dem Testset, Konfusionsmatrix | ✅ fertig |
| **7** | Reflexion, Empfehlung für den Produktiveinsatz, Grenzen | ✅ fertig |

### Ergebnisse aus Schritt 1 (Kurzfassung)

* **Datenqualität:** keine fehlenden Werte; alle 22 Spalten numerisch (`float64`); 13 exakte
  Duplikat-Zeilen, die vor dem Split entfernt werden (ADR-004).
* **Klassenverteilung:** 77,85 / 13,88 / 8,28 %, also stark imbalanciert mit rund 9,4 zu 1.
* **Stärkste Zusammenhänge zur Zielvariable:** `prolongued_decelerations` (r = +0,49),
  `abnormal_short_term_variability` (+0,47),
  `percentage_of_time_with_abnormal_long_term_variability` (+0,43), `accelerations` (−0,36).
* **Multikollinearität:** `histogram_mode`, `histogram_mean` und `histogram_median` korrelieren
  untereinander mit r zwischen 0,89 und 0,95; `histogram_width` und `histogram_min` mit r um
  −0,90 (ADR-005).
* **Nicht-monotone Muster:** `histogram_mean` und `histogram_variance` verhalten sich bei *Suspect*
  gegenläufig zu *Pathological*. Das spricht dafür, dass nichtlineare Modelle hier Vorteile
  gegenüber der Logistischen Regression haben könnten.

### Ergebnisse aus Schritt 2 (Kurzfassung)

* **Fehlende Werte:** keine vorhanden, also keine Imputation (ADR-003).
* **Duplikate:** 13 entfernt, und zwar vor dem Split, es bleiben 2.113 Zeilen; die
  Klassenverteilung bleibt praktisch unverändert bei 77,90 / 13,82 / 8,28 % (ADR-004).
* **Feature-Kodierung:** nicht nötig, es gibt keine Text-Features; `histogram_tendency` mit den
  Werten −1/0/1 ist ordinal und bleibt numerisch (ADR-006).
* **Zielvariable:** per `LabelEncoder` kodiert, Mapping Normal (1.0) auf **0**, Suspect (2.0) auf
  **1**, Pathological (3.0) auf **2**.
* **Split (stratifiziert, 70/15/15):** Train 1.479, Validation 317, Test 317; Klassenanteile in
  allen Teilmengen bei rund 77,9 / 13,8 / 8,2 bis 8,3 %. Wichtig: Validation und Test enthalten je
  nur 26 Pathological-Fälle, ein Fall entspricht also rund 3,8 Prozentpunkten Recall (ADR-009).
* **Skalierung:** `StandardScaler`, ausschließlich auf Train gefittet; skalierte Daten für LogReg,
  k-NN/SVM und MLPs, unskalierte für das Baumverfahren. Den Scaler auf dem Gesamtdatensatz zu
  fitten wäre Data Leakage, die ausführliche Begründung steht in Notebook-Abschnitt 2.7 (ADR-008).

### Ergebnisse aus Schritt 3 (Kurzfassung)

* **Modellwahl:** Random Forest statt Decision Tree oder XGBoost (ADR-011), SVM (RBF) statt k-NN
  (ADR-012). Skalierungssensitive Modelle laufen in einer `Pipeline` mit `StandardScaler`, damit
  die Skalierung je CV-Fold neu gefittet wird.
* **5-fache stratifizierte CV (Trainingsdaten):**

  | Modell | Accuracy | F1 (weighted) | Recall Pathological |
  |---|---|---|---|
  | Logistic Regression | 0,891 ± 0,017 | 0,890 ± 0,016 | 0,740 ± 0,058 |
  | Random Forest | 0,937 ± 0,014 | 0,934 ± 0,016 | 0,862 ± 0,070 |
  | SVM (RBF) | 0,900 ± 0,013 | 0,896 ± 0,014 | 0,708 ± 0,075 |

* **Validierungsset:** Der Random Forest führt in allen Metriken (Accuracy 0,946, F1w 0,945,
  Recall Pathological 0,885, also 23 von 26 erkannt) und geht als Favorit in den Vergleich mit den
  MLPs. Die SVM hat den schwächsten Suspect-Recall mit 0,568; `class_weight` ist als Tuning-Hebel
  für Schritt 6 vorgemerkt (ADR-013).
* Alle Modelle liegen deutlich über der 77,9-%-Baseline. *Suspect* ist durchgängig die
  schwierigste Klasse, mit Recall zwischen 0,57 und 0,77.

### Ergebnisse aus Schritt 4 und 5 (Kurzfassung)

* **Variante A (Adam, Startkonfiguration):** Early Stopping nach 63 Epochen, beste Epoche 58 mit
  `val_loss` 0,231; mildes Overfitting ohne ansteigenden `val_loss`, die Architektur ist also
  geeignet und wir greifen nicht ein. Validierung: Accuracy 0,890, F1w 0,886, Recall Pathological
  0,654, damit auf dem Niveau der Logistischen Regression.
* **Variante B (SGD, Startkonfiguration):** konvergiert sichtbar langsamer und schlechter
  (`val_loss` 0,273, F1w 0,877, Suspect-Recall 0,50) und ist trotz größerer Architektur schlechter
  als A. Dropout eliminiert den Abstand zwischen Trainings- und Validierungskurve; reines SGD mit
  konstanter Lernrate und ohne Momentum ist der Engpass.
* **Dokumentierter Eingriff (ADR-014):** `SGD(momentum=0.9)` und `patience=10`. Die Gegenprobe
  zeigt, dass Patience allein fast nichts bringt. Ergebnis: `val_loss` 0,204, Accuracy 0,912,
  F1w 0,909, Suspect-Recall 0,68. Damit besser als A, aber weiterhin hinter dem Random Forest
  (0,946 / 0,945), vor allem beim Pathological-Recall (0,654 gegenüber 0,885).
* **Antwort auf die zentrale Frage** („Warum ist B nicht automatisch besser als A?“): Kapazität
  nützt nur, wenn die Optimierung sie erschließt; Dropout bekämpft ein Problem, das A kaum hatte;
  die Datenbasis ist klein; und der Optimizer ist Teil des Modells. Ausführlich in
  Notebook-Abschnitt 5.2.

### Ergebnisse aus Schritt 6 (Kurzfassung)

Vergleichstabelle auf dem Validierungsset mit 317 Samples:

| Modell | Typ | Accuracy | F1 (weighted) | Recall Pathological |
|---|---|---|---|---|
| Logistic Regression | Klassisch | 0,8927 | 0,8895 | 0,6538 |
| **Random Forest** | **Klassisch** | **0,9464** | **0,9451** | **0,8846** |
| SVM (RBF) | Klassisch | 0,9022 | 0,8963 | 0,6923 |
| MLP Variante A | Neural Net | 0,8896 | 0,8864 | 0,6538 |
| MLP Variante B (optimiert) | Neural Net | 0,9117 | 0,9085 | 0,6538 |

* **Bestes Modell: Random Forest**, er führt in allen drei Metriken gleichzeitig. Da es ein
  klassisches Modell ist, erfolgt die Optimierung per `GridSearchCV`, KerasTuner entfällt.
* **GridSearchCV:** 4 Parameter (`n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`),
  also 54 Kombinationen mit je 5 Folds.
* **Gewählte Konfiguration, bewusst nicht der Grid-Sieger (ADR-015):** Die Top-5 liegen im F1w
  zwischen 0,9346 und 0,9356, eine Spanne von 0,001 und damit weit innerhalb der CV-Streuung von
  ± 0,016, im Pathological-Recall aber zwischen 0,838 und 0,919. Nach der Priorität aus ADR-002
  fällt die Wahl auf `class_weight="balanced"` mit einem CV-Recall von 0,919. Damit ist auch
  ADR-013 entschieden.
* **Finale Test-Evaluation** (einmaliger Zugriff, Modell auf Train plus Validation trainiert):
  Accuracy 0,927, F1w 0,928.

  | | Precision | Recall | F1 | Support |
  |---|---|---|---|---|
  | Normal | 0,963 | 0,951 | 0,957 | 247 |
  | Suspect | 0,733 | 0,750 | 0,742 | 44 |
  | Pathological | 0,929 | **1,000** | 0,963 | 26 |

* **Konfusionsmatrix:** Häufigste Verwechslung ist Normal gegen Suspect (12 plus 9 = 21 der 23
  Fehler). Schlechtester Recall: Suspect mit 0,750. Alle 26 pathologischen Fälle wurden erkannt,
  kein Fall wurde als Normal durchgewinkt. Einschränkung: Bei nur 26 Fällen reicht das
  95-%-Konfidenzintervall bis etwa 0,87 hinunter.
* **Schwellwert-Analyse (Abschnitt 6.3):** ROC-AUC 0,988 (ovr, weighted) bestätigt die Modellwahl
  schwellenunabhängig. Eine Absenkung der Alarmschwelle für Pathological von 0,50 auf 0,25 würde
  den Recall von 0,885 auf 0,962 heben, das sind aber nur 2 Fälle von 26 und damit Rauschen. Wir
  haben uns bewusst gegen die Verschiebung entschieden (ADR-016).
* **Feature Importances** bestätigen die Analyse aus Schritt 1: `abnormal_short_term_variability`
  (0,141) und `percentage_of_time_with_abnormal_long_term_variability` (0,123) führen. Eine
  Korrektur: `prolongued_decelerations` hatte die höchste Korrelation (r = 0,49), landet als
  Importance aber nur im Mittelfeld, denn es ist in 92 % der Zeilen 0 und trägt zu wenige Splits.

### Ergebnisse aus Schritt 7 (Kurzfassung)

* **Statistische Einordnung (Wilson-95-%-KI):** Recall Pathological 1,000 mit [0,871; 1,000],
  Suspect 0,750 mit [0,606; 0,854], Accuracy 0,927 mit [0,893; 0,951]. Der perfekte Recall ist
  die am wenigsten belastbare Zahl des Projekts, denn „26 von 26“ ist mit einem wahren Recall von
  90 % gut vereinbar.
* **Empfehlung:** Random Forest mit `class_weight="balanced"`, ausdrücklich als Assistenzsystem
  mit ärztlicher Letztentscheidung. Gründe: kein pathologischer Fall als Normal durchgewinkt, in
  allen Metriken vorn, vorsichtiges Fehlerprofil, kein GPU-Bedarf, über Feature Importances
  teilweise erklärbar.
* **Grenzen:** Suspect ist die eigentliche Schwachstelle (Recall 0,750, Precision 0,733); die
  Datenbasis ist zu klein für ein Qualitätsversprechen; die Übertragbarkeit auf andere Kliniken
  ist ungeprüft; und das Modell lernt Befundungsverhalten statt des fetalen Zustands, denn die
  Labels sind ärztliche CTG-Einschätzungen und keine Geburtsausgänge.
* **Mit mehr Daten:** Die `learning_curve` zeigt, dass die letzten 30 % der Trainingsdaten nur noch
  0,009 F1w bringen, bei ± 0,008 Streuung. Mehr Daten verbessern also vor allem die
  Messgenauigkeit und kaum noch das Modell. Für ± 5 Prozentpunkte Präzision beim Recall bräuchte
  es rund 73 Pathological-Fälle im Testset, also etwa 5.900 Samples gesamt statt heute 26
  beziehungsweise 2.113. Rechenleistung war nicht der Engpass, der gesamte GridSearch lief in
  unter 10 Sekunden.
* **Bonusfrage (CNN oder Transfer Learning):** Auf diesen Daten nein, denn die 21 Spalten sind
  eine ungeordnete Menge aggregierter Kennzahlen ohne räumliche oder zeitliche Nachbarschaft, die
  ein Faltungskern ausnutzen könnte. Auf dem rohen CTG-Zeitsignal wäre ein 1D-CNN dagegen
  naheliegend, etwa für die Kopplung zwischen Wehe und Dezeleration, die in den Aggregaten
  verloren ist.

---

## 6. Methodische Grundregeln des Projekts

Diese Regeln gelten über alle Schritte und sind in der `ADR.md` ausführlich begründet:

1. **Kein Data Leakage.** Der `StandardScaler` wird ausschließlich auf den Trainingsdaten
   gefittet; innerhalb der Kreuzvalidierung geschieht die Skalierung in einer `Pipeline`, damit
   sie je Fold neu gefittet wird (ADR-008).
2. **Das Testset wird genau einmal angefasst**, erst in Schritt 6 nach allen Entscheidungen. Alle
   Modellvergleiche laufen über das Validierungsset (ADR-009).
3. **Immer stratifiziert splitten** über `stratify=y` und `StratifiedKFold`, damit die kleinen
   Klassen überall im gleichen Verhältnis vertreten sind (ADR-009).
4. **Fester Seed** `RANDOM_STATE = 42` für alle Zufallsoperationen (ADR-001).
5. **Metrik vor Accuracy:** F1 (weighted) und der Recall der Klasse Pathological entscheiden,
   welches Modell das beste ist (ADR-002).

---

## 7. Hinweis zur KI-Nutzung

KI-Werkzeuge sind laut Assignment ausdrücklich erlaubt. Entsprechend sind alle KI-generierten
Codeblöcke im Notebook mit `# Quelle: Claude` gekennzeichnet. Interpretationen,
Entscheidungsbegründungen und die finale Modellempfehlung verantwortet die Gruppe. Jedes
Gruppenmitglied muss jeden Code-Abschnitt erklären können.

---

## 8. Abgabe-Deliverables (bis 10. August 2026)

1. ✅ Vollständig ausgeführtes Notebook (`.ipynb`) → [src/fetal_health_modellvergleich.ipynb](src/fetal_health_modellvergleich.ipynb)
2. ⬜ PDF-Export des Notebooks (*Datei → Drucken → Als PDF speichern*), aus Colab zu erzeugen
3. ✅ Management Summary → [Management_Summary.md](Management_Summary.md)
