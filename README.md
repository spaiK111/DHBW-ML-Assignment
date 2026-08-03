# Fetal Health Classification – Systematischer Modellvergleich

Assignment „Systematischer Modellvergleich mit klassischen Modellen und neuronalen Netzen –
Mehrklassen-Klassifikation“
**Gruppe C (Gruppe 3)** · Kurs *Maschinelles Lernen mit Python, scikit-learn und Keras* ·
DHBW Stuttgart · SS 2026

---

## 1. Worum geht es?

Aus **Kardiotokogramm-Daten (CTG)** soll der Zustand eines Fötus in drei Klassen eingestuft
werden. Ein CTG zeichnet die fetale Herzfrequenz und die Wehentätigkeit auf; daraus werden
21 numerische Kennwerte abgeleitet (Baseline-Herzfrequenz, Akzelerationen, Dezelerationen,
Variabilitätsmaße und Histogramm-Statistiken).

| | |
|---|---|
| **Aufgabentyp** | Mehrklassen-Klassifikation (3 Klassen) |
| **Zielvariable** | `fetal_health`: 1 = Normal, 2 = Suspect, 3 = Pathological |
| **Umfang** | 2.126 Samples · 21 Features (alle numerisch) |
| **Klassenverteilung** | 1.655 (77,85 %) / 295 (13,88 %) / 176 (8,28 %) – stark imbalanciert |
| **Quelle** | Kaggle: [andrewmvd/fetal-health-classification](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) |

Verglichen werden fünf Modelle: Logistic Regression (multinomial), ein Baumverfahren
(Decision Tree / Random Forest / XGBoost), k-NN oder SVM sowie zwei MLP-Varianten in Keras.

**Fachlich entscheidend:** Weil ein übersehener pathologischer Fall die teuerste
Fehlklassifikation ist, wird nicht nur die Accuracy bewertet, sondern vor allem der
**F1-Score (weighted)** und der **Recall der Klasse Pathological**. Eine naive Vorhersage
„immer Normal“ erreicht bereits 77,9 % Accuracy – daran muss sich jedes Modell messen lassen.

---

## 2. Projektstruktur

```
ML-Projekt/
├── README.md                               ← diese Datei
├── ADR.md                                  ← alle Projektentscheidungen mit Begründung
├── Context/                                ← Aufgabenstellung & Kursmaterial (kein Code)
│   ├── Assignment.pdf
│   ├── Anweisungen.md
│   ├── Script.pdf
│   └── Notebook-LM-*/                      ← Übungsnotebooks aus der Vorlesung
└── src/                                    ← gesamte Implementierung
    ├── requirements.txt                    ← Bibliotheksversionen
    ├── fetal_health.csv                    ← Datensatz (lokal; nicht im Repo, siehe .gitignore)
    └── fetal_health_modellvergleich.ipynb  ← Haupt-Notebook (alle 7 Schritte)
```

Der Ordner `Context/` dient ausschließlich als Kontext und enthält keinen Projektcode.

---

## 3. Einrichtung und Ausführung

### Variante A: Google Colab (empfohlen für die Abgabe)

1. `fetal_health_modellvergleich.ipynb` in Colab öffnen
   (*Datei → Notebook hochladen*).
2. **`fetal_health.csv` in die Colab-Sitzung hochladen** (linke Seitenleiste → Ordnersymbol →
   *Hochladen*). Das Notebook lädt die Datei über einen **relativen Pfad**, sie muss also neben
   dem Notebook liegen.
3. Ab Schritt 4 (Keras): *Laufzeit → Laufzeittyp ändern → T4 GPU*.
4. *Laufzeit → Alle ausführen*.

Alle im Notebook verwendeten Bibliotheken sind in Colab vorinstalliert. Falls `keras-tuner`
für Schritt 6 fehlt:

```python
!pip install keras-tuner
```

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

Getestet mit **Python 3.12**. Das Notebook muss aus dem Ordner `src/` heraus gestartet werden,
damit der relative Pfad zu `fetal_health.csv` aufgeht.

---

## 4. Aufbau des Notebooks

Das Notebook folgt exakt den sieben Schritten des Assignments. Jede Codezelle ist von einer
Markdown-Zelle mit Begründung bzw. Interpretation begleitet.

| Schritt | Inhalt | Status |
|---|---|---|
| **1** | Datenanalyse: Struktur, Datenqualität, Klassenverteilung, 4 Visualisierungen, Feature-Relevanz | ✅ fertig |
| **2** | Preprocessing: Duplikate, Zielkodierung, stratifizierter 70/15/15-Split, Skalierung | ✅ fertig |
| **3** | Drei klassische Modelle mit Default-Parametern + 5-fache stratifizierte Kreuzvalidierung | ✅ fertig |
| **4** | MLP Variante A: `Dense(64, relu) → Dense(3, softmax)`, Adam, Early Stopping | ⬜ offen |
| **5** | MLP Variante B: zusätzlich `Dropout(0.3)` und `Dense(32, relu)`, SGD | ⬜ offen |
| **6** | Vergleichstabelle, Hyperparameter-Optimierung, finale Evaluation auf dem Testset, Konfusionsmatrix | ⬜ offen |
| **7** | Reflexion, Empfehlung für den Produktiveinsatz, Grenzen | ⬜ offen |

### Ergebnisse aus Schritt 1 (Kurzfassung)

- **Datenqualität:** keine fehlenden Werte; alle 22 Spalten numerisch (`float64`); **13 exakte
  Duplikat-Zeilen**, die vor dem Split entfernt werden (→ ADR-004).
- **Klassenverteilung:** 77,85 % / 13,88 % / 8,28 % – stark imbalanciert (≈ 9,4 : 1).
- **Stärkste Zusammenhänge zur Zielvariable:** `prolongued_decelerations` (r = +0,49),
  `abnormal_short_term_variability` (+0,47),
  `percentage_of_time_with_abnormal_long_term_variability` (+0,43), `accelerations` (−0,36).
- **Multikollinearität:** `histogram_mode`/`mean`/`median` korrelieren untereinander mit
  r ≈ 0,89–0,95; `histogram_width` ↔ `histogram_min` mit r ≈ −0,90 (→ ADR-005).
- **Nicht-monotone Muster:** `histogram_mean` und `histogram_variance` verhalten sich bei
  *Suspect* gegenläufig zu *Pathological*. Das spricht dafür, dass nichtlineare Modelle hier
  Vorteile gegenüber der Logistischen Regression haben könnten.

### Ergebnisse aus Schritt 2 (Kurzfassung)

- **Fehlende Werte:** keine vorhanden → keine Imputation (→ ADR-003).
- **Duplikate:** 13 entfernt, **vor** dem Split → 2.113 Zeilen; Klassenverteilung praktisch
  unverändert (77,90 / 13,82 / 8,28 %) (→ ADR-004).
- **Feature-Kodierung:** nicht nötig – keine Text-Features; `histogram_tendency` (−1/0/1) ist
  ordinal und bleibt numerisch (→ ADR-006).
- **Zielvariable:** per `LabelEncoder` kodiert, Mapping: Normal (1.0) → **0**,
  Suspect (2.0) → **1**, Pathological (3.0) → **2**.
- **Split (stratifiziert, 70/15/15):** Train 1.479 / Validation 317 / Test 317; Klassenanteile
  in allen Teilmengen ≈ 77,9 / 13,8 / 8,2–8,3 %. Achtung: Validation und Test enthalten je nur
  **26 Pathological-Fälle** → 1 Fall entspricht ≈ 3,8 Prozentpunkten Recall (→ ADR-009).
- **Skalierung:** `StandardScaler`, **nur auf Train gefittet**; skalierte Daten für LogReg,
  k-NN/SVM und MLPs, unskalierte für das Baumverfahren. Den Scaler auf dem Gesamtdatensatz zu
  fitten wäre Data Leakage – ausführliche Begründung in Notebook-Abschnitt 2.7 (→ ADR-008).

### Ergebnisse aus Schritt 3 (Kurzfassung)

- **Modellwahl:** Random Forest statt Decision Tree/XGBoost (→ ADR-011); SVM (RBF) statt
  k-NN (→ ADR-012). Skalierungssensitive Modelle laufen in einer `Pipeline` mit
  `StandardScaler`, damit die Skalierung je CV-Fold neu gefittet wird.
- **5-fache stratifizierte CV (Trainingsdaten):**

  | Modell | Accuracy | F1 (weighted) | Recall Pathological |
  |---|---|---|---|
  | Logistic Regression | 0,891 ± 0,017 | 0,890 ± 0,016 | 0,740 ± 0,058 |
  | Random Forest | 0,937 ± 0,014 | 0,934 ± 0,016 | 0,862 ± 0,070 |
  | SVM (RBF) | 0,900 ± 0,013 | 0,896 ± 0,014 | 0,708 ± 0,075 |

- **Validierungsset:** Random Forest führt in allen Metriken (Accuracy 0,946 / F1w 0,945 /
  Recall Pathological 0,885 = 23 von 26 erkannt) und geht als Favorit in den Vergleich mit
  den MLPs. Die SVM hat den schwächsten Suspect-Recall (0,568) – `class_weight` ist als
  Tuning-Hebel für Schritt 6 vorgemerkt (→ ADR-013).
- **Alle Modelle** liegen deutlich über der 77,9-%-Baseline; *Suspect* ist durchgängig die
  schwierigste Klasse (Recall 0,57–0,77).

---

## 5. Methodische Grundregeln des Projekts

Diese Regeln gelten über alle Schritte und sind in der `ADR.md` ausführlich begründet:

1. **Kein Data Leakage.** Der `StandardScaler` wird ausschließlich auf den Trainingsdaten
   gefittet; innerhalb der Kreuzvalidierung geschieht die Skalierung in einer `Pipeline`,
   damit sie je Fold neu gefittet wird (→ ADR-008).
2. **Das Testset wird genau einmal angefasst** – erst in Schritt 6 nach allen Entscheidungen.
   Alle Modellvergleiche laufen über das Validierungsset (→ ADR-009).
3. **Immer stratifiziert splitten** (`stratify=y`, `StratifiedKFold`), damit die kleinen Klassen
   überall im gleichen Verhältnis vertreten sind (→ ADR-009).
4. **Fester Seed** `RANDOM_STATE = 42` für alle Zufallsoperationen (→ ADR-001).
5. **Metrik vor Accuracy:** F1 (weighted) und der Recall der Klasse Pathological entscheiden,
   welches Modell „das beste“ ist (→ ADR-002).

---

## 6. Hinweis zur KI-Nutzung

KI-Werkzeuge sind laut Assignment ausdrücklich erlaubt. Entsprechend sind alle
KI-generierten Codeblöcke im Notebook mit `# Quelle: Claude` gekennzeichnet.
Interpretationen, Entscheidungsbegründungen und die finale Modellempfehlung werden von der
Gruppe verantwortet. Jedes Gruppenmitglied muss jeden Code-Abschnitt erklären können.

---

## 7. Abgabe-Deliverables (bis 10. August 2026)

1. Vollständig ausgeführtes Notebook (`.ipynb`)
2. PDF-Export des Notebooks (*Datei → Drucken → Als PDF speichern*)
3. Management Summary
