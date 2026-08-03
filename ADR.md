# Architecture Decision Records (ADR)

**Projekt:** Assignment „Systematischer Modellvergleich – Mehrklassen-Klassifikation“
**Gruppe:** C (Gruppe 3) · DHBW Stuttgart · SS 2026
**Datensatz:** Fetal Health Classification (`src/fetal_health.csv`)

Dieses Dokument hält **alle Projektentscheidungen** fest. Jeder Eintrag beschreibt Kontext,
Entscheidung, Begründung und Konsequenzen. Die Nummern werden im Notebook
(`src/fetal_health_modellvergleich.ipynb`) referenziert.

**Status-Legende:** `Angenommen` = umgesetzt · `Beschlossen` = entschieden, Umsetzung in einem
späteren Schritt · `Offen` = noch zu entscheiden.

**Stand nach Abschluss aller sieben Schritte: alle 16 Entscheidungen sind `Angenommen` –
es ist keine Entscheidung mehr offen.**

| Nr. | Entscheidung | Schritt |
|---|---|---|
| ADR-001 | Projektstruktur, Werkzeuge, `RANDOM_STATE = 42` | 1 |
| ADR-002 | Metrikstrategie: F1w → Recall Pathological → Accuracy | 1 |
| ADR-003 | Keine Imputation (keine fehlenden Werte) | 2.1 |
| ADR-004 | 13 Duplikate vor dem Split entfernen | 2.2 |
| ADR-005 | Keine Feature-Selektion trotz Multikollinearität | 1.6 |
| ADR-006 | Keine Feature-Kodierung (alles numerisch/ordinal) | 2.3 |
| ADR-007 | Keine Ausreißer entfernen | 1.3 / 1.7 |
| ADR-008 | StandardScaler, nur auf Train gefittet | 2.6 |
| ADR-009 | Stratifizierter 70/15/15-Split | 2.5 |
| ADR-010 | Feste Klassenfarben in Ampel-Logik | 1.1 |
| ADR-011 | Baumverfahren: Random Forest | 3.1 |
| ADR-012 | Distanzverfahren: SVM (RBF) statt k-NN | 3.1 |
| ADR-013 | Imbalance über `class_weight="balanced"` | 6.2 |
| ADR-014 | MLP: A unverändert, B mit SGD-Momentum + patience 10 | 4 / 5 |
| ADR-015 | Finales Modell: Random Forest, `class_weight="balanced"` | 6.2–6.5 |
| ADR-016 | Entscheidungsregel bleibt `argmax` (keine Schwellwert-Verschiebung) | 6.3 |

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

## ADR-011 – Modellwahl Baumverfahren: Random Forest

**Status:** Angenommen (umgesetzt in Schritt 3.1)

**Kontext:**
Das Assignment lässt für Modell 2 die Wahl zwischen Decision Tree, Random Forest und
XGBClassifier („wähle ein geeignetes Modell“).

**Entscheidung:**
**Random Forest** (`RandomForestClassifier` mit Default-Parametern, `random_state=42`),
trainiert auf den **unskalierten** Daten.

**Begründung:**
- Ein einzelner Entscheidungsbaum hat hohe Varianz und neigt ohne Beschneidung stark zum
  Overfitting (Kurs, Kapitel 6). Da Schritt 3 ausdrücklich Default-Parameter verlangt, wäre
  ein unregularisierter Einzelbaum die instabilste Wahl. Der Random Forest mittelt viele
  dekorrelierte Bäume (Bagging + zufällige Feature-Auswahl) und generalisiert dadurch auch
  ohne Tuning stabil.
- Robust gegenüber der Multikollinearität aus Schritt 1.6 (→ ADR-005) und skaleninvariant.
- Die nicht-monotonen Feature-Klassen-Zusammenhänge aus Schritt 1.7 kann ein Baumensemble
  direkt abbilden.
- Feature Importances liefern später einen Interpretationszugang.
- XGBoost wäre ähnlich leistungsfähig, war im Kurs aber nur Praxisausblick und bringt eine
  externe Abhängigkeit mit; der Random Forest ist Kernbestandteil von Kapitel 6.

**Konsequenzen:**
Der Random Forest erwies sich in Schritt 3 als bestes klassisches Modell (Validierung:
Accuracy 0,946, F1w 0,945, Recall Pathological 0,885) und ist Favorit für die
Hyperparameter-Optimierung in Schritt 6.

---

## ADR-012 – Modellwahl Distanz-/Kernelverfahren: SVM mit RBF-Kernel

**Status:** Angenommen (umgesetzt in Schritt 3.1)

**Kontext:**
Das Assignment lässt für Modell 3 die Wahl zwischen k-Nächste-Nachbarn und Support Vector
Machine.

**Entscheidung:**
**SVM mit RBF-Kernel** (`SVC()` mit Default-Parametern: C=1, `gamma="scale"`), in einer
Pipeline mit `StandardScaler`.

**Begründung:**
- Bei 21 Dimensionen leidet k-NN unter dem Fluch der Dimensionalität – Abstände verlieren an
  Aussagekraft (Kurs, Kapitel 7).
- Bei stark imbalancierten Klassen benachteiligt das Mehrheitsvoting von k-NN die kleinen
  Klassen systematisch: Die Nachbarschaft ist fast immer von „Normal“ dominiert. Die SVM
  maximiert stattdessen den Rand zwischen den Klassen.
- Der RBF-Kernel kann die nichtlinearen Strukturen aus Schritt 1.7 abbilden.
- Laufzeit bei n = 1.479 unkritisch.

**Konsequenzen:**
Die SVM zeigt mit Defaults den schwächsten Suspect-Recall aller Modelle (0,568) –
`class_weight="balanced"` ist als Optimierungshebel für Schritt 6 vorgemerkt (→ ADR-013).

---

## ADR-013 – Klassen-Imbalance über `class_weight="balanced"` behandeln

**Status:** Angenommen (umgesetzt in Schritt 6.2)

**Kontext:**
Die Klassen sind mit 78 / 14 / 8 % stark imbalanciert (→ ADR-002). Bis Schritt 5 wurden alle
Modelle **ohne** Gegenmaßnahme trainiert, weil Schritt 3 ausdrücklich Default-Parameter
verlangt. Die Folge war durchgängig ein schwacher Recall auf den kleinen Klassen
(Pathological 0,65–0,88, Suspect 0,50–0,77).

**Entscheidung:**
Die Imbalance wird über **`class_weight="balanced"`** im finalen Random Forest behandelt –
nicht über Resampling (SMOTE, Undersampling).

**Begründung:**
- **Datengetrieben statt gesetzt:** `class_weight` wurde als Parameter in den GridSearchCV
  aufgenommen (None / balanced / balanced_subsample) und die Kreuzvalidierung hat entschieden.
  Die Gewichtung hebt den CV-Recall auf Pathological von 0,838 auf 0,919, ohne den F1w
  nennenswert zu beschädigen (0,9346 vs. 0,9356).
- **Warum kein Resampling:** SMOTE erzeugt synthetische Fälle durch Interpolation zwischen
  Nachbarn. Bei medizinischen Messdaten mit nur 175 pathologischen Fällen entstünden damit
  CTG-Profile, die so nie gemessen wurden – fachlich schwer zu rechtfertigen. Undersampling
  würde Daten der ohnehin knappen Gesamtmenge verwerfen. `class_weight` erreicht denselben
  Effekt durch Umgewichtung im Loss, ohne die Datenbasis zu verfälschen.
- Das Verfahren ist außerdem in scikit-learn eingebaut und benötigt keine externe
  Bibliothek (`imbalanced-learn`).

**Konsequenzen:**
Bewusster Kompromiss zulasten der Mehrheitsklasse: Auf dem Validierungsset steigt der Recall
für Suspect von 0,773 auf 0,864 und für Pathological von 0,885 auf 0,923, während Normal von
0,984 auf 0,951 nachgibt. Im finalen Testergebnis wurden **alle 26 pathologischen Fälle
erkannt** – erkauft mit 12 Fehlalarmen (Normal → Suspect). Im Screening-Kontext ist das der
gewollte Tausch (→ ADR-002).

---

## ADR-014 – MLP-Startkonfigurationen: Variante A unverändert, Variante B mit SGD-Momentum

**Status:** Angenommen (umgesetzt in Schritt 4/5)

**Kontext:**
Das Assignment gibt für beide MLP-Varianten Startkonfigurationen vor und erlaubt
ausdrücklich dokumentierte Eingriffe, wenn die Lernkurven Schwächen zeigen („Bewertet wird
nicht, dass die Startvariante perfekt läuft, sondern dass Sie ihre Schwächen erkennen und
begründet darauf reagieren“).

**Entscheidung:**
- **Variante A (Adam): keine Änderung.** Die Startkonfiguration ist geeignet.
- **Variante B: Architektur unverändert**, aber Optimierung angepasst:
  `SGD(momentum=0.9)` statt reinem SGD sowie Early-Stopping-`patience` 10 statt 5.
  In die Vergleichstabelle geht die optimierte Variante B ein; die Startkonfiguration
  bleibt im Notebook dokumentiert.

**Begründung:**
- Variante A zeigte saubere Lernkurven: kein vorzeitiger Abbruch (Early Stopping erst in
  Epoche 63, bestes `val_loss` 0,231 in Epoche 58), klarer Lernfortschritt, nur mildes
  Overfitting ohne ansteigenden `val_loss` – kein Handlungsbedarf.
- Variante B (Start) konvergierte mit reinem SGD (konstante Lernrate 0,01, kein Momentum)
  sichtbar langsamer und schlechter: bestes `val_loss` 0,273, F1w 0,877 – trotz größerer
  Architektur schlechter als Variante A (0,886). Der Suspect-Recall brach auf 0,50 ein.
- **Gegenprobe:** Nur `patience` auf 10 zu erhöhen (ohne Momentum) verbesserte fast nichts
  (`val_loss` 0,264 nach 76 Epochen) – der Engpass war der Optimierer, nicht das
  Abbruchkriterium.
- Momentum 0,9 ist der Standard-Zusatz für SGD (Kurs, Kapitel 9) und bleibt bei der
  Assignment-Vorgabe „SGD statt Adam“; die höhere Patience verhindert, dass die flache,
  leicht verrauschte SGD-Kurve ein zufälliges 5-Epochen-Plateau als Konvergenz
  fehlinterpretiert.

**Konsequenzen:**
Die optimierte Variante B erreicht `val_loss` 0,204 / Accuracy 0,912 / F1w 0,909 und
übertrifft Variante A – bleibt aber hinter dem Random Forest (F1w 0,945), insbesondere beim
Recall der Klasse Pathological (0,654 vs. 0,885). Der Random Forest bleibt Favorit für
Schritt 6.

---

## ADR-015 – Finales Modell: Random Forest mit `class_weight="balanced"`

**Status:** Angenommen (umgesetzt in Schritt 6.2–6.4)

**Kontext:**
Nach dem Vergleich aller fünf Modelle auf dem Validierungsset musste das beste Modell für die
Hyperparameter-Optimierung und die finale Test-Evaluation bestimmt werden.

**Entscheidung:**
1. **Bestes Modell: Random Forest.** Er führt auf dem Validierungsset in allen drei Metriken
   gleichzeitig (Accuracy 0,946 / F1w 0,945 / Recall Pathological 0,885) – deutlich vor den
   neuronalen Netzen (F1w 0,886 bzw. 0,909).
2. **Optimierung mit `GridSearchCV`** über vier Parameter (`n_estimators`, `max_depth`,
   `min_samples_leaf`, `class_weight`) = 54 Kombinationen × 5 Folds. KerasTuner entfällt, da
   das beste Modell kein neuronales Netz ist.
3. **Gewählte Konfiguration – bewusst nicht der Grid-Sieger:**
   `class_weight="balanced"`, `max_depth=None`, `min_samples_leaf=1`, `n_estimators=100`.
4. **Finales Training auf Train + Validation** (1.796 Samples), dann **genau ein** Zugriff
   auf das Testset.

**Begründung:**
- **Zur Abweichung vom Grid-Sieger:** Die Top-5-Konfigurationen liegen im F1 (weighted)
  zwischen 0,9346 und 0,9356 – eine Spanne von 0,001, die weit innerhalb der CV-Streuung
  von ± 0,016 (Schritt 3.2) liegt. Nach der Hauptmetrik sind sie statistisch nicht
  unterscheidbar. Im Recall auf Pathological unterscheiden sie sich dagegen deutlich
  (0,838–0,919). Damit greift die zweite Stufe unserer Metrikstrategie (→ ADR-002) und wählt
  die Konfiguration mit CV-Recall 0,919. Die Auswahl beruht **ausschließlich auf
  CV-Ergebnissen der Trainingsdaten** – Validierungs- und Testset waren nicht beteiligt.
- **Zum Training auf Train + Validation:** Nach Abschluss aller Entscheidungen hat das
  Validierungsset seine Aufgabe erfüllt. Bei nur 175 pathologischen Fällen insgesamt ist der
  Zugewinn real: Das Endmodell sieht 149 statt 123 davon. Das Testset bleibt unberührt und
  damit ein unverfälschter Schätzer.

**Konsequenzen:**
Finales Testergebnis: **Accuracy 0,927 / F1 (weighted) 0,928**, Recall Pathological
**1,000** (26/26), Suspect 0,750, Normal 0,951. Die Zahlen liegen erwartungsgemäß leicht
unter den Validierungswerten – ein Zeichen dafür, dass die Validierungsmetriken durch die
wiederholte Modellauswahl leicht optimistisch waren. Da das Testset nur 26 pathologische
Fälle enthält, ist der perfekte Recall mit Vorsicht zu lesen (95-%-Konfidenzintervall bis
ca. 0,87 hinunter).

---

## ADR-016 – Entscheidungsregel bleibt `argmax` (keine Schwellwert-Verschiebung)

**Status:** Angenommen (analysiert und entschieden in Schritt 6.3)

**Kontext:**
Ein Random Forest liefert über `predict_proba()` Wahrscheinlichkeiten für alle drei Klassen;
die Klassenzuweisung entsteht daraus per `argmax`. Für ein Screening-System ist das nicht
zwingend die richtige Regel: Man könnte bewusst alarmieren, sobald die Wahrscheinlichkeit für
*Pathological* eine Schwelle t überschreitet – auch wenn rechnerisch eine andere Klasse
wahrscheinlicher ist. Der Kurs führt diese Schwellwert-Anpassung für imbalancierte Daten
ausdrücklich ein (Kapitel 6, Fraud-Detection-Notebook).

**Entscheidung:**
Wir behalten die **`argmax`-Standardregel** bei. Die Schwellwert-Analyse wird durchgeführt und
dokumentiert, führt aber zu keiner Änderung am Modell.

**Begründung:**
- Die Analyse auf dem Validierungsset zeigt den erwarteten Verlauf: Eine Absenkung von
  t = 0,50 auf t = 0,25 hebt den Recall für Pathological von 0,885 auf 0,962, senkt die
  Precision aber von 0,958 auf 0,833. Unter t = 0,20 bricht die Precision ein (bis 0,581),
  ohne dass der Recall weiter steigt.
- Der scheinbare Gewinn entspricht **genau zwei Fällen** (23 → 25 von 26). Bei dieser
  Stichprobengröße ist das nicht von Rauschen zu unterscheiden – dieselbe Vorsicht, die wir
  in Schritt 7.1 auf die Testergebnisse anwenden (Konfidenzintervalle).
- Eine auf 26 Fällen kalibrierte Schwelle wäre mit hoher Wahrscheinlichkeit auf das
  Validierungsset überangepasst und würde auf neuen Daten nicht halten.
- `class_weight="balanced"` (→ ADR-013) verschiebt die Wahrscheinlichkeiten bereits zugunsten
  der kleinen Klassen; eine zusätzliche Schwellenverschiebung würde denselben Effekt ein
  zweites Mal anwenden.

**Konsequenzen:**
Die Schwelle bleibt als Stellhebel dokumentiert und ist in Schritt 7.5 als Maßnahme mit dem
besten Verhältnis von Aufwand zu klinischem Nutzen vermerkt – sobald belastbare Fehlerkosten
(„ein übersehener Fall entspricht X Fehlalarmen") und mehr Fälle vorliegen. Ergänzend wurde
die **ROC-AUC** als schwellenunabhängiges Gütemaß berechnet (0,988 ovr/weighted auf der
Validierung); sie stützt die Modellwahl unabhängig von der gewählten Entscheidungsregel.
