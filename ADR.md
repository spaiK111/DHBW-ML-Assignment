# Architecture Decision Records (ADR)

**Projekt:** Assignment „Systematischer Modellvergleich – Mehrklassen-Klassifikation“
**Gruppe:** C (Gruppe 3) · DHBW Stuttgart · SS 2026
**Datensatz:** Fetal Health Classification (`src/fetal_health.csv`)

Dieses Dokument hält **alle Projektentscheidungen** fest. Jeder Eintrag beschreibt Kontext,
Entscheidung, Begründung und Konsequenzen. Die Nummern werden im Notebook
(`src/fetal_health_modellvergleich.ipynb`) referenziert.

**Status-Legende:** `Angenommen` = umgesetzt · `Beschlossen` = entschieden, Umsetzung in einem
späteren Schritt · `Offen` = noch zu entscheiden.

---

## ADR-001 – Projektstruktur, Werkzeuge und Reproduzierbarkeit

**Status:** Angenommen (Schritt 1)

**Kontext:**
Das Assignment verlangt ein vollständig ausgeführtes Colab-Notebook als Abgabe. Die
gruppeninternen Anweisungen verlangen zusätzlich, den gesamten Code im Ordner `src/`
abzulegen und Entscheidungen zentral zu dokumentieren.

**Entscheidung:**
- Die gesamte Bearbeitung erfolgt in **einem** Jupyter-/Colab-Notebook:
  `src/fetal_health_modellvergleich.ipynb`, gegliedert nach den sieben Schritten des Assignments.
- Der Datensatz liegt als lokale Kopie unter `src/fetal_health.csv` und wird mit einem
  **relativen Pfad** geladen (`pd.read_csv("fetal_health.csv")`).
- Verwendet wird ausschließlich der aus dem Unterricht bekannte Stack:
  `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, ab Schritt 4 `TensorFlow/Keras`.
- Ein globales `RANDOM_STATE = 42` wird an alle Funktionen mit Zufallsanteil übergeben
  (Splits, Kreuzvalidierung, Modelle).
- Der Ordner `Context/` enthält ausschließlich Aufgabenstellung und Kursmaterial und wird
  nicht Teil der Implementierung.

**Begründung:**
- Ein einzelnes Notebook entspricht dem geforderten Abgabeformat und hält Code, Ergebnis und
  Begründung (Markdown-Zellen) an derselben Stelle – wie im Kurs praktiziert.
- Relativer Pfad statt `kagglehub`-Download: Der Datensatz ist mit der lokalen CSV eindeutig
  fixiert; ein späterer Upstream-Wechsel auf Kaggle kann die Ergebnisse nicht verändern.
- Ein fester Seed macht alle berichteten Zahlen reproduzierbar – Voraussetzung dafür, dass der
  Modellvergleich in Schritt 6 überhaupt aussagekräftig ist.

**Konsequenzen:**
Zum Ausführen in Colab muss `fetal_health.csv` neben das Notebook hochgeladen werden
(siehe `src/Readme.md`). Ergebnisse sind bei gleicher Bibliotheksversion exakt reproduzierbar;
bei abweichenden Versionen können minimale numerische Unterschiede auftreten.

---

## ADR-002 – Bewertungsmetriken und Erfolgsmaßstab

**Status:** Angenommen (Schritt 1, gilt für alle folgenden Schritte)

**Kontext:**
Die Zielvariable ist stark imbalanciert: 77,85 % Normal, 13,88 % Suspect, 8,28 % Pathological
(Schritt 1.4). Ein Modell, das immer „Normal“ vorhersagt, erreicht damit bereits ≈ 77,9 %
Accuracy, ohne einen einzigen kritischen Fall zu erkennen.

**Entscheidung:**
Wir bewerten Modelle anhand von drei Größen in dieser Priorität:
1. **F1-Score (weighted)** – vom Assignment gefordert, berücksichtigt alle Klassen.
2. **Recall der Klasse Pathological** – fachliches Hauptkriterium.
3. **Accuracy** – nur als Kontextgröße, immer gegen die naive Baseline von **77,9 %** gelesen.

Ergänzend wird für das finale Modell der `classification_report` (Precision/Recall/F1 je Klasse)
sowie die 3×3-Konfusionsmatrix ausgegeben.

**Begründung:**
Im Praxiskontext eines CTG-Screenings ist ein **falsch-negativer pathologischer Fall**
(Pathological wird als Normal eingestuft) ungleich teurer als ein falsch-positiver Alarm:
Ersteres kann eine unentdeckte fetale Notlage bedeuten, Letzteres führt „nur“ zu einer
zusätzlichen ärztlichen Kontrolle. Accuracy allein bildet dieses Kostenverhältnis nicht ab –
genau darauf weist auch der didaktische Hinweis im Assignment hin.

**Konsequenzen:**
Ein Modell mit höherer Accuracy, aber schlechterem Recall auf Pathological, wird **nicht**
als „bestes Modell“ ausgewählt. Die Entscheidung in Schritt 6 wird explizit anhand dieser
Rangfolge begründet.

---

## ADR-003 – Umgang mit fehlenden Werten

**Status:** Angenommen (umgesetzt in Schritt 2.1)

**Kontext:**
Schritt 2 des Assignments verlangt, fehlende Werte zu identifizieren, zu behandeln und die
Entscheidung zu begründen.

**Entscheidung:**
**Keine Imputation.** Die Prüfung mit `df.isna().sum()` ergibt **0 fehlende Werte** in allen
2.126 × 22 Zellen.

**Begründung:**
Es gibt nichts zu imputieren. Jede Imputationsstrategie (Mittelwert, Median, KNN) würde hier
nur zusätzliche Komplexität und eine potenzielle Fehlerquelle einführen, ohne einen einzigen
Wert zu verändern.

**Konsequenzen:**
Die Prüfung bleibt trotzdem als Codezelle im Notebook, um den geforderten Nachweis zu erbringen.
Falls der Datensatz später ausgetauscht wird, muss diese Entscheidung neu getroffen werden.

---

## ADR-004 – Behandlung der 13 Duplikat-Zeilen

**Status:** Angenommen (erkannt in Schritt 1.2, umgesetzt in Schritt 2.2)

**Kontext:**
`df.duplicated().sum()` liefert **13 exakte Duplikat-Zeilen** (≈ 0,6 % der Daten).

**Entscheidung:**
Die Duplikate werden in **Schritt 2 vor dem Train/Validation/Test-Split** entfernt
(`df.drop_duplicates()`). In Schritt 1 bleiben sie für die deskriptive Analyse enthalten.

**Begründung:**
- Bleiben identische Zeilen erhalten, können sie durch den zufälligen Split gleichzeitig im
  Trainings- **und** im Validierungs-/Testset landen. Das Modell würde auf Validierungs-/Testdaten
  bewertet, die es exakt so schon im Training gesehen hat – die Generalisierungsleistung würde
  **überschätzt** (eine Form von Data Leakage).
- Das Entfernen muss **vor** dem Split geschehen, sonst wirkt es nicht.
- In Schritt 1 (rein deskriptiv, kein Modell) verändern 13 von 2.126 Zeilen die Statistiken
  praktisch nicht; die Klassenverteilung wird deshalb auf dem vollständigen Originaldatensatz
  berichtet, wie im Assignment beschrieben.

**Alternative (verworfen):** Duplikate behalten – mit dem Argument, dass zwei CTGs zufällig
identische abgeleitete Kennwerte haben könnten. Verworfen, weil das Leakage-Risiko schwerer
wiegt als der Verlust von 0,6 % der Daten.

**Konsequenzen:**
Ab Schritt 2 wird mit 2.113 Zeilen gearbeitet (bestätigt in Schritt 2.2; die
Klassenverteilung ändert sich dadurch nur minimal: 77,90 / 13,82 / 8,28 %). Die Zeilenzahl
wird im Notebook vor und nach dem Entfernen ausgegeben, damit der Unterschied zu Schritt 1
nachvollziehbar bleibt.

---

## ADR-005 – Umgang mit Multikollinearität (keine Feature-Selektion vorab)

**Status:** Angenommen (Schritt 1.6)

**Kontext:**
Die Korrelationsmatrix zeigt mehrere sehr stark korrelierte Feature-Paare:

| Feature-Paar | r |
|---|---|
| `histogram_mean` ↔ `histogram_median` | 0,948 |
| `histogram_mode` ↔ `histogram_median` | 0,933 |
| `histogram_mode` ↔ `histogram_mean` | 0,893 |
| `histogram_width` ↔ `histogram_min` | −0,899 |

**Entscheidung:**
Es werden **keine Features vorab entfernt**. Alle 21 Features gehen in den Modellvergleich ein.

**Begründung:**
- Multikollinearität beeinträchtigt vor allem die **Interpretierbarkeit der Koeffizienten** der
  Logistischen Regression, kaum aber deren Vorhersagegüte. Baumverfahren und neuronale Netze
  sind davon weitgehend unberührt.
- Ein Vergleich von fünf Modellen ist nur dann fair, wenn alle **dieselbe Feature-Menge**
  sehen. Eine manuelle Vorauswahl würde die Vergleichbarkeit beeinträchtigen.
- Eine datengetriebene Selektion anhand der *gesamten* Daten wäre zudem eine Form von
  Leakage, da die Auswahl Information aus dem Testset nutzen würde.

**Konsequenzen:**
Koeffizienten der Logistischen Regression werden **nicht** kausal interpretiert. Die Redundanz
ist als möglicher Vereinfachungshebel für Schritt 7 („Was würden wir mit mehr Zeit anders
machen?“) vorgemerkt.

---

## ADR-006 – Ordinal wirkende Features bleiben numerisch (keine Kodierung)

**Status:** Angenommen (umgesetzt in Schritt 2.3)

**Kontext:**
Schritt 2 verlangt, kategorische Features zu kodieren (One-Hot oder Label Encoding – mit
Begründung). Alle 22 Spalten liegen als `float64` vor. Zwei Features nehmen jedoch nur wenige
diskrete Werte an: `histogram_tendency` ∈ {−1, 0, 1} und `histogram_number_of_zeroes`
(kleine ganze Zahlen).

**Entscheidung:**
**Keine Kodierung.** Alle Features werden als numerisch behandelt.

**Begründung:**
- Es gibt **keine Text-Features** im Datensatz – der klassische Anwendungsfall für One-Hot- oder
  Label-Encoding entfällt vollständig.
- `histogram_tendency` ist **ordinal mit sinnvoller Ordnung** (linksschief < symmetrisch <
  rechtsschief) und bereits passend als −1/0/1 kodiert. Ein One-Hot-Encoding würde die
  Ordnungsinformation zerstören und ohne Not drei Spalten erzeugen.
- `histogram_number_of_zeroes` ist eine echte Zählgröße; der numerische Abstand ist inhaltlich
  sinnvoll.

**Konsequenzen:**
Schritt 2 dokumentiert die Prüfung explizit („keine kategorischen Features vorhanden“), damit
nachvollziehbar ist, dass der Punkt bearbeitet und nicht übersehen wurde.

---

## ADR-007 – Keine Entfernung von Ausreißern

**Status:** Angenommen (Schritt 1.3 / 1.7)

**Kontext:**
Die Boxplots zeigen bei mehreren Features zahlreiche Punkte außerhalb der Whisker, z. B. bei
`histogram_variance`, `accelerations` und `mean_value_of_long_term_variability`.

**Entscheidung:**
Es werden **keine Ausreißer entfernt**.

**Begründung:**
- Alle Werte liegen in **physiologisch plausiblen Bereichen** (z. B. FHR-Baseline 106–160 bpm);
  es gibt keine Hinweise auf Messfehler oder unmögliche Werte.
- Bei einem medizinischen Screening sind **gerade die Extremwerte die diagnostisch relevanten
  Fälle**. Ein „Ausreißer“ in `prolongued_decelerations` ist mit hoher Wahrscheinlichkeit ein
  pathologischer Fall – ihn zu entfernen würde die ohnehin kleinste Klasse weiter ausdünnen und
  dem Ziel des Modells direkt zuwiderlaufen.

**Konsequenzen:**
Skalensensitive Modelle müssen mit den Extremwerten umgehen. Wir verwenden dennoch den
`StandardScaler` (siehe ADR-008) und behalten `RobustScaler` als Alternative in Reserve, falls
sich in Schritt 3 Auffälligkeiten zeigen.

---

## ADR-008 – Skalierung: StandardScaler, nur auf den Trainingsdaten gefittet

**Status:** Angenommen (umgesetzt in Schritt 2.6)

**Kontext:**
Die Features liegen auf sehr unterschiedlichen Skalen: `accelerations` ≈ 0–0,02,
`baseline value` ≈ 106–160, `histogram_variance` ≈ 0–269. Schritt 2 stellt außerdem die Frage,
ob der Scaler auf dem gesamten Datensatz gefittet werden darf.

**Entscheidung:**
- **StandardScaler** (Standardisierung auf Mittelwert 0, Standardabweichung 1) auf **allen 21
  Features**.
- Der Scaler wird **ausschließlich auf den Trainingsdaten gefittet** (`fit_transform` auf Train,
  nur `transform` auf Validation und Test).
- Für Decision Tree / Random Forest / XGBoost werden zusätzlich die **unskalierten** Daten
  verwendet, da Baumverfahren skaleninvariant sind.

**Begründung:**
- **Warum skalieren:** Distanzbasierte Verfahren (k-NN, SVM) und gradientenbasierte Verfahren
  (Logistische Regression, neuronale Netze) werden sonst von den betragsmäßig großen Features
  dominiert; die kleinen, aber hochinformativen Features wie `accelerations` gingen praktisch unter.
- **Warum nur auf Train fitten:** Nein, der Scaler darf **nicht** auf dem gesamten Datensatz
  gefittet werden. Mittelwert und Standardabweichung sind gelernte Parameter. Werden sie aus
  allen Daten berechnet, fließt Information aus Validierungs- und Testset in die
  Trainingsvorverarbeitung ein (**Data Leakage**) – die berichtete Leistung wäre optimistisch
  verzerrt und nicht mehr repräsentativ für neue, ungesehene Daten.
- **Warum StandardScaler statt MinMaxScaler:** MinMax skaliert auf den Wertebereich [0, 1] und
  reagiert damit sehr empfindlich auf die in ADR-007 bewusst behaltenen Extremwerte – ein
  einzelner Extremwert staucht alle übrigen Werte zusammen. Der StandardScaler ist hier robuster.

**Konsequenzen:**
Innerhalb der Kreuzvalidierung muss die Skalierung in einer **`Pipeline`** stattfinden, damit
sie je Fold neu gefittet wird und kein Leakage über die Folds entsteht.

---

## ADR-009 – Stratifizierter 70/15/15-Split

**Status:** Angenommen (umgesetzt in Schritt 2.5)

**Kontext:**
Das Assignment verlangt einen stratifizierten Train/Validation/Test-Split im Verhältnis
70/15/15. Die kleinste Klasse (Pathological) hat nur 176 Fälle.

**Entscheidung:**
Zweistufiger Split mit `train_test_split(..., stratify=y, random_state=42)`:
1. 70 % Train ↔ 30 % Rest
2. Rest hälftig in 15 % Validation ↔ 15 % Test

**Begründung:**
- **Stratifizierung** hält die Klassenanteile (78/14/8 %) in allen drei Teilmengen konstant.
  Ohne sie könnte das Testset zufällig deutlich zu wenige Pathological-Fälle enthalten
  (bei 15 % von 176 sind es nur ca. 26) – die Metriken der kritischen Klasse wären dann kaum
  belastbar.
- Ein **separates Validierungsset** ist nötig, weil alle Modellvergleiche und die
  Hyperparameter-Optimierung darauf stattfinden. Das Testset bleibt bis zur finalen Evaluation
  in Schritt 6 unangetastet („exakt einmal anfassen“).

**Konsequenzen:**
Bei 2.113 Zeilen nach Duplikat-Entfernung ergeben sich **1.479 / 317 / 317** Samples mit
jeweils exakt **26 Pathological-Fällen** in Validation und Test (bestätigt in Schritt 2.5).
Die kleine Absolutzahl bedeutet, dass der Recall dieser Klasse eine **hohe Varianz** hat –
ein einziger Fall entspricht ≈ 3,8 Prozentpunkten Recall. Das ist bei der Interpretation in
Schritt 6 und 7 zu berücksichtigen.

---

## ADR-010 – Visualisierung: feste Klassenfarben in Ampel-Logik

**Status:** Angenommen (Schritt 1.1)

**Kontext:**
Die drei Zielklassen tauchen in mehreren Diagrammen (Balkendiagramm, Boxplots, später
Konfusionsmatrix) auf.

**Entscheidung:**
Feste Farbzuordnung über das gesamte Notebook: **Normal = grün**, **Suspect = gelb**,
**Pathological = rot**. Die Klassennamen stehen zusätzlich immer als Text an den Achsen.

**Begründung:**
Eine über alle Diagramme konstante Farbzuordnung verhindert Fehlinterpretationen beim
Diagrammwechsel. Die Ampel-Logik entspricht der medizinischen Konvention (grün = unauffällig,
rot = kritisch) und macht den Schweregrad ohne Legende lesbar. Da die Klassennamen immer
zusätzlich beschriftet sind, ist die Information nicht allein farbcodiert – das Diagramm bleibt
auch bei Farbfehlsichtigkeit und im Schwarzweißdruck lesbar.

**Konsequenzen:**
Die Konstanten `CLASS_NAMES`, `CLASS_ORDER` und `CLASS_COLORS` werden einmal zentral im Setup
definiert und in allen weiteren Schritten wiederverwendet.

---

## Offene Entscheidungen (kommende Schritte)

| Nr. | Thema | Schritt |
|---|---|---|
| ADR-011 | Modellwahl Kapitel 5: Decision Tree, Random Forest oder XGBClassifier | Schritt 3 |
| ADR-012 | Modellwahl Kapitel 7/8: k-NN oder SVM | Schritt 3 |
| ADR-013 | Umgang mit der Klassen-Imbalance im Modell (`class_weight` / Klassengewichte) | Schritt 3–5 |
| ADR-014 | Eventuelle Anpassungen der MLP-Startarchitekturen A und B nach Analyse der Lernkurven | Schritt 4–5 |
| ADR-015 | Auswahl des „besten“ Modells und Parameterraum der Hyperparameter-Optimierung | Schritt 6 |
