# Architecture Decision Records (ADR)

**Projekt:** Assignment „Systematischer Modellvergleich, Mehrklassen-Klassifikation“
**Gruppe:** C (Gruppe 3) · DHBW Stuttgart · SS 2026
**Datensatz:** Fetal Health Classification (`src/fetal_health.csv`)

Dieses Dokument hält alle Projektentscheidungen fest. Jeder Eintrag beschreibt Kontext,
Entscheidung, Begründung und Konsequenzen. Die Nummern werden im Notebook
(`src/fetal_health_modellvergleich.ipynb`) referenziert.

**Status-Legende:** `Angenommen` heißt umgesetzt, `Beschlossen` heißt entschieden mit Umsetzung
in einem späteren Schritt, `Offen` heißt noch zu entscheiden.

**Stand nach Abschluss aller sieben Schritte: alle 18 Entscheidungen sind `Angenommen`, es ist
keine Entscheidung mehr offen.**

| Nr. | Entscheidung | Schritt |
|---|---|---|
| ADR-001 | Projektstruktur, Werkzeuge, `RANDOM_STATE = 42` | 1 |
| ADR-002 | Metrikstrategie: F1w, dann Recall Pathological, dann Accuracy | 1 |
| ADR-003 | Keine Imputation (keine fehlenden Werte) | 2.1 |
| ADR-004 | 13 Duplikate vor dem Split entfernen | 2.2 |
| ADR-005 | Keine Feature-Selektion trotz Multikollinearität | 1.6 |
| ADR-006 | Keine Feature-Kodierung (alles numerisch oder ordinal) | 2.3 |
| ADR-007 | Keine Ausreißer entfernen | 1.3 / 1.7 |
| ADR-008 | StandardScaler, nur auf Train gefittet | 2.6 |
| ADR-009 | Stratifizierter 70/15/15-Split | 2.5 |
| ADR-010 | Feste Klassenfarben in Ampel-Logik | 1.1 |
| ADR-011 | Baumverfahren: Random Forest | 3.1 |
| ADR-012 | Distanzverfahren: SVM (RBF) statt k-NN | 3.1 |
| ADR-013 | Imbalance über `class_weight="balanced"` | 6.2 |
| ADR-014 | MLP: A unverändert, B mit SGD-Momentum und patience 10 | 4 / 5 |
| ADR-015 | Finales Modell: Random Forest, `class_weight="balanced"` | 6.2 bis 6.5 |
| ADR-016 | Entscheidungsregel bleibt `argmax` (keine Schwellwert-Verschiebung) | 6.3 |
| ADR-017 | Namenskonvention und Code-Stil im Notebook | alle |
| ADR-018 | Bewusste Abweichungen von den Kursnotebooks | alle |

---

## ADR-001: Projektstruktur, Werkzeuge und Reproduzierbarkeit

**Status:** Angenommen (Schritt 1)

**Kontext:**
Das Assignment verlangt ein vollständig ausgeführtes Colab-Notebook als Abgabe. Die
gruppeninternen Anweisungen verlangen zusätzlich, den gesamten Code im Ordner `src/` abzulegen
und Entscheidungen zentral zu dokumentieren.

**Entscheidung:**

* Die gesamte Bearbeitung erfolgt in *einem* Jupyter- oder Colab-Notebook,
  `src/fetal_health_modellvergleich.ipynb`, gegliedert nach den sieben Schritten des Assignments.
* Der Datensatz wird **zweistufig** geladen: Liegt `fetal_health.csv` neben dem Notebook, wird
  diese lokale Kopie verwendet; andernfalls lädt `kagglehub` den Datensatz direkt aus der im
  Assignment genannten Quelle `andrewmvd/fetal-health-classification`. Die lokale Datei hat
  dabei immer Vorrang.
* Verwendet wird ausschließlich der aus dem Unterricht bekannte Stack: `pandas`, `numpy`,
  `matplotlib`, `seaborn`, `scikit-learn` und ab Schritt 4 TensorFlow/Keras.
* Ein globales `RANDOM_STATE = 42` wird an alle Funktionen mit Zufallsanteil übergeben, also an
  Splits, Kreuzvalidierung und Modelle.
* Der Ordner `Context/` enthält ausschließlich Aufgabenstellung und Kursmaterial und wird nicht
  Teil der Implementierung.

**Begründung:**
Ein einzelnes Notebook entspricht dem geforderten Abgabeformat und hält Code, Ergebnis und
Begründung in den Markdown-Zellen an derselben Stelle, so wie im Kurs praktiziert. Ein fester
Seed macht alle berichteten Zahlen reproduzierbar, was Voraussetzung dafür ist, dass der
Modellvergleich in Schritt 6 überhaupt aussagekräftig ist.

Zum **Vorrang der lokalen Datei**: Beide Ziele lassen sich so gleichzeitig erreichen. Die
lokale CSV fixiert den Datensatz eindeutig – ein späterer Wechsel der Quelldaten auf Kaggle
kann die Ergebnisse nicht unbemerkt verändern. Gleichzeitig macht der `kagglehub`-Fallback das
Notebook in Google Colab ohne manuellen Upload lauffähig, was die Abgabe- und
Nachvollziehbarkeitsanforderung des Assignments erfüllt. Wäre die Reihenfolge umgekehrt, ginge
die Reproduzierbarkeit verloren.

**Konsequenzen:**
Das Notebook läuft in Colab ohne Vorbereitung: öffnen und ausführen. `kagglehub` ist in Colab
vorinstalliert; lokal ist es nur nötig, wenn keine CSV vorliegt (`pip install kagglehub`).
Beide Ladewege wurden geprüft und liefern **byte-identische Daten** (2.126 × 22, `df.equals()`
= True). Ergebnisse sind bei gleicher Bibliotheksversion exakt reproduzierbar; bei abweichenden
Versionen können minimale numerische Unterschiede auftreten.

---

## ADR-002: Bewertungsmetriken und Erfolgsmaßstab

**Status:** Angenommen (Schritt 1, gilt für alle folgenden Schritte)

**Kontext:**
Die Zielvariable ist stark imbalanciert, mit 77,85 % Normal, 13,88 % Suspect und 8,28 %
Pathological (Schritt 1.4). Ein Modell, das immer „Normal“ vorhersagt, erreicht damit bereits
rund 77,9 % Accuracy, ohne einen einzigen kritischen Fall zu erkennen.

**Entscheidung:**
Wir bewerten Modelle anhand von drei Größen in dieser Priorität:

1. **F1-Score (weighted)**, vom Assignment gefordert, berücksichtigt alle Klassen.
2. **Recall der Klasse Pathological**, unser fachliches Hauptkriterium.
3. **Accuracy**, nur als Kontextgröße und immer gegen die naive Baseline von 77,9 % gelesen.

Ergänzend geben wir für das finale Modell den `classification_report` mit Precision, Recall und
F1 je Klasse sowie die 3×3-Konfusionsmatrix aus.

**Begründung:**
Im Praxiskontext eines CTG-Screenings ist ein falsch-negativer pathologischer Fall, bei dem
Pathological als Normal eingestuft wird, ungleich teurer als ein falsch-positiver Alarm. Ersteres
kann eine unentdeckte fetale Notlage bedeuten, Letzteres führt nur zu einer zusätzlichen
ärztlichen Kontrolle. Accuracy allein bildet dieses Kostenverhältnis nicht ab, worauf auch der
didaktische Hinweis im Assignment zielt.

**Konsequenzen:**
Ein Modell mit höherer Accuracy, aber schlechterem Recall auf Pathological, wird nicht als bestes
Modell ausgewählt. Die Entscheidung in Schritt 6 wird explizit anhand dieser Rangfolge begründet.

---

## ADR-003: Umgang mit fehlenden Werten

**Status:** Angenommen (umgesetzt in Schritt 2.1)

**Kontext:**
Schritt 2 des Assignments verlangt, fehlende Werte zu identifizieren, zu behandeln und die
Entscheidung zu begründen.

**Entscheidung:**
Keine Imputation. Die Prüfung mit `df.isna().sum()` ergibt 0 fehlende Werte in allen 2.126 × 22
Zellen.

**Begründung:**
Es gibt nichts zu imputieren. Jede Imputationsstrategie, ob Mittelwert, Median oder k-NN, würde
hier nur zusätzliche Komplexität und eine potenzielle Fehlerquelle einführen, ohne einen einzigen
Wert zu verändern.

**Konsequenzen:**
Die Prüfung bleibt trotzdem als Codezelle im Notebook, um den geforderten Nachweis zu erbringen.
Falls der Datensatz später ausgetauscht wird, muss diese Entscheidung neu getroffen werden.

---

## ADR-004: Behandlung der 13 Duplikat-Zeilen

**Status:** Angenommen (erkannt in Schritt 1.2, umgesetzt in Schritt 2.2)

**Kontext:**
`df.duplicated().sum()` liefert 13 exakte Duplikat-Zeilen, das sind rund 0,6 % der Daten.

**Entscheidung:**
Die Duplikate werden in Schritt 2 vor dem Train/Validation/Test-Split entfernt, per
`df.drop_duplicates()`. In Schritt 1 bleiben sie für die deskriptive Analyse enthalten.

**Begründung:**
Bleiben identische Zeilen erhalten, können sie durch den zufälligen Split gleichzeitig im
Trainings- und im Validierungs- oder Testset landen. Das Modell würde dann auf Daten bewertet,
die es exakt so schon im Training gesehen hat, und die Generalisierungsleistung wäre überschätzt.
Das ist eine Form von Data Leakage. Das Entfernen muss vor dem Split geschehen, sonst wirkt es
nicht. In Schritt 1, der rein deskriptiv ist und noch kein Modell enthält, verändern 13 von 2.126
Zeilen die Statistiken praktisch nicht; die Klassenverteilung berichten wir deshalb auf dem
vollständigen Originaldatensatz, wie im Assignment beschrieben.

**Alternative (verworfen):**
Duplikate behalten, mit dem Argument, dass zwei CTGs zufällig identische abgeleitete Kennwerte
haben könnten. Verworfen, weil das Leakage-Risiko schwerer wiegt als der Verlust von 0,6 % der
Daten.

**Konsequenzen:**
Ab Schritt 2 wird mit 2.113 Zeilen gearbeitet, bestätigt in Schritt 2.2. Die Klassenverteilung
ändert sich dadurch nur minimal, auf 77,90 / 13,82 / 8,28 %. Die Zeilenzahl wird im Notebook vor
und nach dem Entfernen ausgegeben, damit der Unterschied zu Schritt 1 nachvollziehbar bleibt.

---

## ADR-005: Umgang mit Multikollinearität (keine Feature-Selektion vorab)

**Status:** Angenommen (Schritt 1.6)

**Kontext:**
Die Korrelationsmatrix zeigt mehrere sehr stark korrelierte Feature-Paare:

| Feature-Paar | r |
|---|---|
| `histogram_mean` und `histogram_median` | 0,948 |
| `histogram_mode` und `histogram_median` | 0,933 |
| `histogram_mode` und `histogram_mean` | 0,893 |
| `histogram_width` und `histogram_min` | −0,899 |

**Entscheidung:**
Es werden keine Features vorab entfernt. Alle 21 Features gehen in den Modellvergleich ein.

**Begründung:**
Multikollinearität beeinträchtigt vor allem die Interpretierbarkeit der Koeffizienten der
Logistischen Regression, kaum aber deren Vorhersagegüte; Baumverfahren und neuronale Netze sind
davon weitgehend unberührt. Außerdem ist ein Vergleich von fünf Modellen nur dann fair, wenn alle
dieselbe Feature-Menge sehen, eine manuelle Vorauswahl würde die Vergleichbarkeit beeinträchtigen.
Und eine datengetriebene Selektion anhand der *gesamten* Daten wäre selbst eine Form von Leakage,
da die Auswahl Information aus dem Testset nutzen würde.

**Konsequenzen:**
Koeffizienten der Logistischen Regression werden nicht kausal interpretiert. Die Redundanz ist als
möglicher Vereinfachungshebel für Schritt 7 vorgemerkt („Was würden wir mit mehr Zeit anders
machen?“).

---

## ADR-006: Ordinal wirkende Features bleiben numerisch (keine Kodierung)

**Status:** Angenommen (umgesetzt in Schritt 2.3)

**Kontext:**
Schritt 2 verlangt, kategorische Features zu kodieren, per One-Hot oder Label Encoding und mit
Begründung. Alle 22 Spalten liegen als `float64` vor. Zwei Features nehmen jedoch nur wenige
diskrete Werte an: `histogram_tendency` mit {−1, 0, 1} und `histogram_number_of_zeroes` mit
kleinen ganzen Zahlen.

**Entscheidung:**
Keine Kodierung. Alle Features werden als numerisch behandelt.

**Begründung:**
Es gibt keine Text-Features im Datensatz, der klassische Anwendungsfall für One-Hot- oder
Label-Encoding entfällt damit vollständig. `histogram_tendency` ist ordinal mit sinnvoller Ordnung
(linksschief kleiner symmetrisch kleiner rechtsschief) und bereits passend als −1/0/1 kodiert; ein
One-Hot-Encoding würde die Ordnungsinformation zerstören und ohne Not drei Spalten erzeugen.
`histogram_number_of_zeroes` ist eine echte Zählgröße, bei der der numerische Abstand inhaltlich
sinnvoll ist.

**Konsequenzen:**
Schritt 2 dokumentiert die Prüfung explizit („keine kategorischen Features vorhanden“), damit
nachvollziehbar ist, dass der Punkt bearbeitet und nicht übersehen wurde.

---

## ADR-007: Keine Entfernung von Ausreißern

**Status:** Angenommen (Schritt 1.3 und 1.7)

**Kontext:**
Die Boxplots zeigen bei mehreren Features zahlreiche Punkte außerhalb der Whisker, etwa bei
`histogram_variance`, `accelerations` und `mean_value_of_long_term_variability`.

**Entscheidung:**
Es werden keine Ausreißer entfernt.

**Begründung:**
Alle Werte liegen in physiologisch plausiblen Bereichen, etwa die FHR-Baseline mit 106 bis 160
bpm, und es gibt keine Hinweise auf Messfehler oder unmögliche Werte. Bei einem medizinischen
Screening sind gerade die Extremwerte die diagnostisch relevanten Fälle: Ein „Ausreißer“ in
`prolongued_decelerations` ist mit hoher Wahrscheinlichkeit ein pathologischer Fall. Ihn zu
entfernen würde die ohnehin kleinste Klasse weiter ausdünnen und dem Ziel des Modells direkt
zuwiderlaufen.

**Konsequenzen:**
Skalensensitive Modelle müssen mit den Extremwerten umgehen. Wir verwenden dennoch den
`StandardScaler` (siehe ADR-008) und behalten `RobustScaler` als Alternative in Reserve, falls
sich in Schritt 3 Auffälligkeiten zeigen.

---

## ADR-008: Skalierung mit StandardScaler, nur auf den Trainingsdaten gefittet

**Status:** Angenommen (umgesetzt in Schritt 2.6)

**Kontext:**
Die Features liegen auf sehr unterschiedlichen Skalen: `accelerations` zwischen 0 und 0,02,
`baseline value` zwischen 106 und 160, `histogram_variance` zwischen 0 und 269. Schritt 2 stellt
außerdem die Frage, ob der Scaler auf dem gesamten Datensatz gefittet werden darf.

**Entscheidung:**

* `StandardScaler`, also Standardisierung auf Mittelwert 0 und Standardabweichung 1, auf allen 21
  Features.
* Der Scaler wird ausschließlich auf den Trainingsdaten gefittet, also `fit_transform` auf Train
  und nur `transform` auf Validation und Test.
* Für Decision Tree, Random Forest oder XGBoost werden zusätzlich die unskalierten Daten
  verwendet, da Baumverfahren skaleninvariant sind.

**Begründung:**
Warum überhaupt skalieren: Distanzbasierte Verfahren wie k-NN und SVM und gradientenbasierte
Verfahren wie Logistische Regression und neuronale Netze werden sonst von den betragsmäßig großen
Features dominiert, und die kleinen, aber hochinformativen Features wie `accelerations` gingen
praktisch unter.

Warum nur auf Train fitten: Nein, der Scaler darf nicht auf dem gesamten Datensatz gefittet
werden. Mittelwert und Standardabweichung sind gelernte Parameter. Werden sie aus allen Daten
berechnet, fließt Information aus Validierungs- und Testset in die Trainingsvorverarbeitung ein,
und das ist Data Leakage. Die berichtete Leistung wäre optimistisch verzerrt und nicht mehr
repräsentativ für neue, ungesehene Daten.

Warum StandardScaler statt MinMaxScaler: MinMax skaliert auf den Wertebereich [0, 1] und reagiert
damit sehr empfindlich auf die in ADR-007 bewusst behaltenen Extremwerte, denn ein einzelner
Extremwert staucht alle übrigen Werte zusammen. Der StandardScaler ist hier robuster.

**Konsequenzen:**
Innerhalb der Kreuzvalidierung muss die Skalierung in einer `Pipeline` stattfinden, damit sie je
Fold neu gefittet wird und kein Leakage über die Folds entsteht.

---

## ADR-009: Stratifizierter 70/15/15-Split

**Status:** Angenommen (umgesetzt in Schritt 2.5)

**Kontext:**
Das Assignment verlangt einen stratifizierten Train/Validation/Test-Split im Verhältnis 70/15/15.
Die kleinste Klasse Pathological hat nur 176 Fälle.

**Entscheidung:**
Zweistufiger Split mit `train_test_split(..., stratify=y, random_state=42)`:

1. 70 % Train gegen 30 % Rest
2. Rest hälftig in 15 % Validation und 15 % Test

**Begründung:**
Die Stratifizierung hält die Klassenanteile von 78 / 14 / 8 % in allen drei Teilmengen konstant.
Ohne sie könnte das Testset zufällig deutlich zu wenige Pathological-Fälle enthalten, bei 15 % von
176 sind es nur etwa 26, und die Metriken der kritischen Klasse wären dann kaum belastbar. Ein
separates Validierungsset ist nötig, weil alle Modellvergleiche und die Hyperparameter-Optimierung
darauf stattfinden; das Testset bleibt bis zur finalen Evaluation in Schritt 6 unangetastet und
wird exakt einmal angefasst.

**Konsequenzen:**
Bei 2.113 Zeilen nach der Duplikat-Entfernung ergeben sich 1.479 / 317 / 317 Samples mit jeweils
exakt 26 Pathological-Fällen in Validation und Test, bestätigt in Schritt 2.5. Die kleine
Absolutzahl bedeutet, dass der Recall dieser Klasse eine hohe Varianz hat: Ein einziger Fall
entspricht rund 3,8 Prozentpunkten Recall. Das ist bei der Interpretation in Schritt 6 und 7 zu
berücksichtigen.

---

## ADR-010: Visualisierung mit festen Klassenfarben in Ampel-Logik

**Status:** Angenommen (Schritt 1.1)

**Kontext:**
Die drei Zielklassen tauchen in mehreren Diagrammen auf, im Balkendiagramm, in den Boxplots und
später in der Konfusionsmatrix.

**Entscheidung:**
Feste Farbzuordnung über das gesamte Notebook: Normal grün, Suspect gelb, Pathological rot. Die
Klassennamen stehen zusätzlich immer als Text an den Achsen.

**Begründung:**
Eine über alle Diagramme konstante Farbzuordnung verhindert Fehlinterpretationen beim
Diagrammwechsel. Die Ampel-Logik entspricht der medizinischen Konvention, also grün für
unauffällig und rot für kritisch, und macht den Schweregrad ohne Legende lesbar. Da die
Klassennamen immer zusätzlich beschriftet sind, ist die Information nicht allein farbcodiert; das
Diagramm bleibt auch bei Farbfehlsichtigkeit und im Schwarzweißdruck lesbar.

**Konsequenzen:**
Die Konstanten `CLASS_NAMES`, `CLASS_ORDER` und `CLASS_COLORS` werden einmal zentral im Setup
definiert und in allen weiteren Schritten wiederverwendet.

---

## ADR-011: Modellwahl beim Baumverfahren, Random Forest

**Status:** Angenommen (umgesetzt in Schritt 3.1)

**Kontext:**
Das Assignment lässt für Modell 2 die Wahl zwischen Decision Tree, Random Forest und
XGBClassifier („wähle ein geeignetes Modell“).

**Entscheidung:**
Random Forest, also `RandomForestClassifier` mit Default-Parametern und `random_state=42`,
trainiert auf den unskalierten Daten.

**Begründung:**
Ein einzelner Entscheidungsbaum hat hohe Varianz und neigt ohne Beschneidung stark zum
Overfitting (Kurs, Kapitel 6). Da Schritt 3 ausdrücklich Default-Parameter verlangt, wäre ein
unregularisierter Einzelbaum die instabilste Wahl. Der Random Forest mittelt viele dekorrelierte
Bäume über Bagging und zufällige Feature-Auswahl und generalisiert dadurch auch ohne Tuning
stabil. Er ist robust gegenüber der Multikollinearität aus Schritt 1.6 (ADR-005) und
skaleninvariant, kann die nicht-monotonen Feature-Klassen-Zusammenhänge aus Schritt 1.7 direkt
abbilden und liefert über Feature Importances später einen Interpretationszugang. XGBoost wäre
ähnlich leistungsfähig, war im Kurs aber nur Praxisausblick und bringt eine externe Abhängigkeit
mit; der Random Forest ist Kernbestandteil von Kapitel 6.

**Konsequenzen:**
Der Random Forest erwies sich in Schritt 3 als bestes klassisches Modell (Validierung: Accuracy
0,946, F1w 0,945, Recall Pathological 0,885) und ist Favorit für die Hyperparameter-Optimierung
in Schritt 6.

---

## ADR-012: Modellwahl beim Distanz- oder Kernelverfahren, SVM mit RBF-Kernel

**Status:** Angenommen (umgesetzt in Schritt 3.1)

**Kontext:**
Das Assignment lässt für Modell 3 die Wahl zwischen k-Nächste-Nachbarn und Support Vector Machine.

**Entscheidung:**
SVM mit RBF-Kernel, also `SVC()` mit den Defaults C=1 und `gamma="scale"`, in einer Pipeline mit
`StandardScaler`.

**Begründung:**
Bei 21 Dimensionen leidet k-NN unter dem Fluch der Dimensionalität, Abstände verlieren also an
Aussagekraft (Kurs, Kapitel 7). Bei stark imbalancierten Klassen benachteiligt das
Mehrheitsvoting von k-NN die kleinen Klassen zusätzlich systematisch, denn die Nachbarschaft ist
fast immer von „Normal“ dominiert. Die SVM maximiert stattdessen den Rand zwischen den Klassen,
und der RBF-Kernel kann die nichtlinearen Strukturen aus Schritt 1.7 abbilden. Die Laufzeit ist
bei n = 1.479 unkritisch.

**Konsequenzen:**
Die SVM zeigt mit Defaults den schwächsten Suspect-Recall aller Modelle mit 0,568.
`class_weight="balanced"` ist als Optimierungshebel für Schritt 6 vorgemerkt (ADR-013).

---

## ADR-013: Klassen-Imbalance über `class_weight="balanced"` behandeln

**Status:** Angenommen (umgesetzt in Schritt 6.2)

**Kontext:**
Die Klassen sind mit 78 / 14 / 8 % stark imbalanciert (ADR-002). Bis Schritt 5 wurden alle Modelle
ohne Gegenmaßnahme trainiert, weil Schritt 3 ausdrücklich Default-Parameter verlangt. Die Folge
war durchgängig ein schwacher Recall auf den kleinen Klassen, bei Pathological zwischen 0,65 und
0,88 und bei Suspect zwischen 0,50 und 0,77.

**Entscheidung:**
Die Imbalance wird über `class_weight="balanced"` im finalen Random Forest behandelt und nicht
über Resampling wie SMOTE oder Undersampling.

**Begründung:**
Datengetrieben statt gesetzt: `class_weight` wurde als Parameter in den GridSearchCV aufgenommen,
mit den Werten None, balanced und balanced_subsample, und die Kreuzvalidierung hat entschieden.
Die Gewichtung hebt den CV-Recall auf Pathological von 0,838 auf 0,919, ohne den F1w nennenswert
zu beschädigen (0,9346 gegenüber 0,9356).

Warum kein Resampling: SMOTE erzeugt synthetische Fälle durch Interpolation zwischen Nachbarn.
Bei medizinischen Messdaten mit nur 175 pathologischen Fällen entstünden damit CTG-Profile, die so
nie gemessen wurden, was fachlich schwer zu rechtfertigen ist. Undersampling würde Daten der
ohnehin knappen Gesamtmenge verwerfen. `class_weight` erreicht denselben Effekt durch Umgewichtung
im Loss, ohne die Datenbasis zu verfälschen, und ist in scikit-learn eingebaut, benötigt also
keine externe Bibliothek wie `imbalanced-learn`.

**Konsequenzen:**
Ein bewusster Kompromiss zulasten der Mehrheitsklasse: Auf dem Validierungsset steigt der Recall
für Suspect von 0,773 auf 0,864 und für Pathological von 0,885 auf 0,923, während Normal von 0,984
auf 0,951 nachgibt. Im finalen Testergebnis wurden alle 26 pathologischen Fälle erkannt, erkauft
mit 12 Fehlalarmen von Normal zu Suspect. Im Screening-Kontext ist das der gewollte Tausch
(ADR-002).

---

## ADR-014: MLP-Startkonfigurationen, Variante A unverändert, Variante B mit SGD-Momentum

**Status:** Angenommen (umgesetzt in Schritt 4 und 5)

**Kontext:**
Das Assignment gibt für beide MLP-Varianten Startkonfigurationen vor und erlaubt ausdrücklich
dokumentierte Eingriffe, wenn die Lernkurven Schwächen zeigen: „Bewertet wird nicht, dass die
Startvariante perfekt läuft, sondern dass Sie ihre Schwächen erkennen und begründet darauf
reagieren“.

**Entscheidung:**

* Variante A (Adam) bleibt unverändert, die Startkonfiguration ist geeignet.
* Variante B behält ihre Architektur, wir passen aber die Optimierung an: `SGD(momentum=0.9)`
  statt reinem SGD und Early-Stopping-`patience` 10 statt 5. In die Vergleichstabelle geht die
  optimierte Variante B ein, die Startkonfiguration bleibt im Notebook dokumentiert.

**Begründung:**
Variante A zeigte saubere Lernkurven: kein vorzeitiger Abbruch, denn Early Stopping griff erst in
Epoche 63 bei einem besten `val_loss` von 0,231 in Epoche 58, klarer Lernfortschritt und nur
mildes Overfitting ohne ansteigenden `val_loss`. Es gab also keinen Handlungsbedarf.

Variante B in der Startkonfiguration konvergierte mit reinem SGD, also konstanter Lernrate 0,01
ohne Momentum, sichtbar langsamer und schlechter: bestes `val_loss` 0,273 und F1w 0,877, und damit
trotz größerer Architektur schlechter als Variante A mit 0,886. Der Suspect-Recall brach auf 0,50
ein.

Die Gegenprobe zeigt, dass es tatsächlich am Optimierer lag: `patience` allein auf 10 zu erhöhen,
ohne Momentum, verbesserte fast nichts (`val_loss` 0,264 nach 76 Epochen).

Momentum 0,9 ist der übliche Zusatz zu SGD und bleibt bei der Assignment-Vorgabe „SGD statt
Adam“. Im Kurs selbst kam SGD nur ohne Momentum vor, als `SGD(learning_rate=0.01)` in K10_03;
genau deshalb belegen wir die Wirkung mit der Gegenprobe statt sie vorauszusetzen (siehe
ADR-018). Die höhere Patience folgt dagegen direkt der Kurslogik: In K10_03 wird die Patience für
das SGD-Modell gegenüber dem Adam-Notebook K10_02 von 3 auf 5 angehoben, „da SGD langsamer
konvergiert“. Wir tun bei einem längeren Epochenbudget dasselbe und verhindern damit, dass die
flache, leicht verrauschte SGD-Kurve ein zufälliges 5-Epochen-Plateau als Konvergenz
fehlinterpretiert.

**Konsequenzen:**
Die optimierte Variante B erreicht `val_loss` 0,204, Accuracy 0,912 und F1w 0,909 und übertrifft
damit Variante A, bleibt aber hinter dem Random Forest mit F1w 0,945, insbesondere beim Recall der
Klasse Pathological (0,654 gegenüber 0,885). Der Random Forest bleibt Favorit für Schritt 6.

---

## ADR-015: Finales Modell, Random Forest mit `class_weight="balanced"`

**Status:** Angenommen (umgesetzt in Schritt 6.2 bis 6.4)

**Kontext:**
Nach dem Vergleich aller fünf Modelle auf dem Validierungsset musste das beste Modell für die
Hyperparameter-Optimierung und die finale Test-Evaluation bestimmt werden.

**Entscheidung:**

1. Bestes Modell ist der Random Forest. Er führt auf dem Validierungsset in allen drei Metriken
   gleichzeitig (Accuracy 0,946, F1w 0,945, Recall Pathological 0,885) und liegt deutlich vor den
   neuronalen Netzen mit F1w 0,886 beziehungsweise 0,909.
2. Optimierung mit `GridSearchCV` über vier Parameter (`n_estimators`, `max_depth`,
   `min_samples_leaf`, `class_weight`), also 54 Kombinationen mit je 5 Folds. KerasTuner entfällt,
   da das beste Modell kein neuronales Netz ist.
3. Gewählte Konfiguration, bewusst nicht der Grid-Sieger: `class_weight="balanced"`,
   `max_depth=None`, `min_samples_leaf=1`, `n_estimators=100`.
4. Finales Training auf Train plus Validation mit 1.796 Samples, danach genau ein Zugriff auf das
   Testset.

**Begründung:**
Zur Abweichung vom Grid-Sieger: Die Top-5-Konfigurationen liegen im F1 (weighted) zwischen 0,9346
und 0,9356, also eine Spanne von 0,001, die weit innerhalb der CV-Streuung von ± 0,016 aus
Schritt 3.2 liegt. Nach der Hauptmetrik sind sie statistisch nicht unterscheidbar. Im Recall auf
Pathological unterscheiden sie sich dagegen deutlich, nämlich zwischen 0,838 und 0,919. Damit
greift die zweite Stufe unserer Metrikstrategie (ADR-002) und wählt die Konfiguration mit
CV-Recall 0,919. Die Auswahl beruht ausschließlich auf CV-Ergebnissen der Trainingsdaten,
Validierungs- und Testset waren nicht beteiligt. Umgesetzt ist die Regel als Code und nicht als
handgeschriebene Parameterliste, damit Begründung und tatsächlich verwendete Konfiguration nicht
auseinanderlaufen können.

Zum Training auf Train plus Validation: Nach Abschluss aller Entscheidungen hat das
Validierungsset seine Aufgabe erfüllt. Bei nur 175 pathologischen Fällen insgesamt ist der
Zugewinn real, denn das Endmodell sieht 149 statt 123 davon. Das Testset bleibt unberührt und
damit ein unverfälschter Schätzer.

**Konsequenzen:**
Finales Testergebnis: Accuracy 0,927 und F1 (weighted) 0,928, Recall Pathological 1,000 bei 26 von
26, Suspect 0,750, Normal 0,951. Die Zahlen liegen erwartungsgemäß leicht unter den
Validierungswerten, ein Zeichen dafür, dass die Validierungsmetriken durch die wiederholte
Modellauswahl leicht optimistisch waren. Da das Testset nur 26 pathologische Fälle enthält, ist
der perfekte Recall mit Vorsicht zu lesen, das 95-%-Konfidenzintervall reicht bis etwa 0,87
hinunter.

---

## ADR-016: Entscheidungsregel bleibt `argmax` (keine Schwellwert-Verschiebung)

**Status:** Angenommen (analysiert und entschieden in Schritt 6.3)

**Kontext:**
Ein Random Forest liefert über `predict_proba()` Wahrscheinlichkeiten für alle drei Klassen; die
Klassenzuweisung entsteht daraus per `argmax`. Für ein Screening-System ist das nicht zwingend die
richtige Regel: Man könnte bewusst alarmieren, sobald die Wahrscheinlichkeit für *Pathological*
eine Schwelle t überschreitet, auch wenn rechnerisch eine andere Klasse wahrscheinlicher ist. Der
Kurs führt diese Schwellwert-Anpassung für imbalancierte Daten ausdrücklich ein (Kapitel 6,
Fraud-Detection-Notebook).

**Entscheidung:**
Wir behalten die `argmax`-Standardregel bei. Die Schwellwert-Analyse wird durchgeführt und
dokumentiert, führt aber zu keiner Änderung am Modell.

**Begründung:**
Die Analyse auf dem Validierungsset zeigt den erwarteten Verlauf: Eine Absenkung von t = 0,50 auf
t = 0,25 hebt den Recall für Pathological von 0,885 auf 0,962, senkt die Precision aber von 0,958
auf 0,833. Unter t = 0,20 bricht die Precision ein, bis auf 0,581, ohne dass der Recall weiter
steigt.

Der scheinbare Gewinn entspricht genau zwei Fällen, nämlich 25 statt 23 von 26. Bei dieser
Stichprobengröße ist das nicht von Rauschen zu unterscheiden; es ist dieselbe Vorsicht, die wir in
Schritt 7.1 über Konfidenzintervalle auf die Testergebnisse anwenden. Eine auf 26 Fällen
kalibrierte Schwelle wäre außerdem mit hoher Wahrscheinlichkeit auf das Validierungsset
überangepasst und würde auf neuen Daten nicht halten. Hinzu kommt, dass
`class_weight="balanced"` (ADR-013) die Wahrscheinlichkeiten bereits zugunsten der kleinen Klassen
verschiebt; eine zusätzliche Schwellenverschiebung würde denselben Effekt ein zweites Mal
anwenden.

**Konsequenzen:**
Die Schwelle bleibt als Stellhebel dokumentiert und ist in Schritt 7.5 als Maßnahme mit dem besten
Verhältnis von Aufwand zu klinischem Nutzen vermerkt, sobald belastbare Fehlerkosten („ein
übersehener Fall entspricht X Fehlalarmen“) und mehr Fälle vorliegen. Ergänzend wurde die ROC-AUC
als schwellenunabhängiges Gütemaß berechnet, mit 0,988 (ovr, weighted) auf der Validierung; sie
stützt die Modellwahl unabhängig von der gewählten Entscheidungsregel.

---

## ADR-017: Namenskonvention und Code-Stil im Notebook

**Status:** Angenommen (gilt für das gesamte Notebook)

**Kontext:**
Das Notebook ist über sieben Schritte gewachsen. Dabei standen deutsche und englische Bezeichner
nebeneinander (`modelle` neben `X_train`, `evaluiere` neben `scaler`), und Umlaute waren in
Kommentaren teilweise als `ae`, `oe`, `ue` umschrieben. Für eine Abgabe, die von mehreren Personen
gelesen und erklärt werden muss, ist das unnötig uneinheitlich.

**Entscheidung:**
Eine feste Konvention für das gesamte Notebook:

| Regel | Beispiel |
|---|---|
| Bezeichner auf Englisch, wie in sklearn, Keras und den Kursnotebooks | `models`, `history_a`, `final_model` |
| Kommentare, Docstrings und Markdown auf Deutsch | `# Kontrolle: Größen und Klassenanteile je Teilmenge` |
| `snake_case` für Variablen und Funktionen | `plot_learning_curves`, `evaluate_predictions` |
| `UPPER_SNAKE_CASE` für Konstanten, die einmal gesetzt werden | `RANDOM_STATE`, `TARGET_NAMES`, `TOLERANCE` |
| Großes `X` für Feature-Matrizen, kleines `y` für Zielvektoren | `X_train`, `y_val` |
| Suffixe beschreiben Teilmenge und Aufbereitung | `_train`, `_val`, `_test`, `_scaled` |
| Modelle, Trainingsverläufe und Vorhersagen tragen sprechende Präfixe | `model_b_opt`, `history_b_opt`, `y_pred_b_opt` |
| Keine Umschreibung von Umlauten in Bezeichnern oder Kommentaren | `chosen` statt `gewaehlt`, „für“ statt „fuer“ |

**Begründung:**
Die Kursnotebooks benennen durchgehend englisch (`X`, `y`, `model`, `y_pred`, `cm`, `scaler`,
`history`, `param_grid`, `grid`), und die Bibliotheks-APIs selbst sind ebenfalls englisch.
Englische Bezeichner stehen damit direkt neben den Aufrufen, in die sie eingesetzt werden, und
brechen den Lesefluss nicht. Die Erklärungen bleiben dagegen deutsch, weil die Abgabe deutsch ist
und die Begründungen bewertet werden. Die Trennung ist eindeutig und damit leicht einzuhalten.

**Konsequenzen:**
Alle deutschen Bezeichner wurden ersetzt, unter anderem `modelle` zu `models`, `evaluiere` zu
`evaluate_predictions`, `finales_modell` zu `final_model`, `lernkurven_plot` zu
`plot_learning_curves`, `gewaehlt` zu `chosen` und `TOLERANZ` zu `TOLERANCE`. Da die Umbenennung
nur Bezeichner betraf und keine Zeichenketten oder Parameter, sind sämtliche Ergebnisse und
Ausgaben unverändert. Die Konvention steht zusätzlich in der `README.md`, damit sie bei künftigen
Ergänzungen greifbar ist.

---

## ADR-018: Bewusste Abweichungen von den Kursnotebooks

**Status:** Angenommen (gilt für das gesamte Notebook)

**Kontext:**
Die gruppeninternen Anweisungen verlangen, die Praktiken und Methoden aus dem Unterricht zu
verwenden. Wir haben das Notebook deshalb Baustein für Baustein gegen die Übungsnotebooks in
`Context/Notebook-LM-*` abgeglichen.

**Was direkt aus dem Kurs übernommen ist:**

| Baustein | Vorbild im Kurs |
|---|---|
| `train_test_split(..., stratify=y)` | durchgängig ab K01_02 |
| `StandardScaler` in einer Pipeline | K02_00, K03_01, K03_03, K03_05 |
| `StratifiedKFold` und `cross_validate` | K03_04, K02_00 |
| `GridSearchCV` mit `param_grid` | K03_05, K06_02 |
| `DummyClassifier` als naive Baseline | K03_03 |
| `classification_report` und `ConfusionMatrixDisplay` | K03_03, K06_05 |
| ROC-AUC, `PrecisionRecallDisplay`, Schwellwert-Analyse bei Imbalance | K06_05 (Fraud Detection) |
| `learning_curve` | K03_01 |
| `RandomForestClassifier`, `feature_importances_` | K06_03, K06_00 |
| `SVC()` mit RBF-Kernel | K08_00 bis K08_04 |
| `keras.Sequential` mit `Dense` und `Dropout`, `sparse_categorical_crossentropy` | K09_00, K10_02, K10_05 |
| `EarlyStopping` mit `restore_best_weights=True`, erzeugt über eine Factory-Funktion | K10_02 bis K10_05 (`make_early_stop()`) |
| Aufbau „Variante A gegen Variante B, Adam gegen SGD, gleiche Architektur“ | K10_03 |

Besonders eng ist die Anlehnung in Schritt 4 und 5: K10_03 vergleicht genau wie wir dieselbe
Architektur unter zwei Optimierern und hebt für SGD die Early-Stopping-Patience an, „da SGD
langsamer konvergiert“.

**Wo wir bewusst abweichen, und warum:**

1. **Seeds über `keras.utils.set_random_seed()` statt `np.random.seed()` plus
   `tf.random.set_seed()`.** Die Kursnotebooks setzen beide Seeds einmal am Anfang. Unter Keras 3
   reicht das nicht: Gewichts-Initialisierung und Dropout-Masken nutzen eigene Seed-Generatoren,
   die von den beiden Aufrufen nicht erfasst werden. Wir haben das geprüft und je nach
   Ausführungsreihenfolge abweichende Lernkurven erhalten. `set_random_seed()` deckt alle drei
   Generatoren ab und wird vor *jedem* Modellaufbau erneut aufgerufen, weil wir vier Netze
   nacheinander trainieren und diese untereinander vergleichbar sein müssen.
2. **`SGD(momentum=0.9)` in der optimierten Variante B.** Im Kurs kommt SGD nur ohne Momentum vor,
   als `SGD(learning_rate=0.01)` in K10_03. Wir setzen Momentum deshalb nicht als bekannt voraus,
   sondern belegen seine Wirkung in Abschnitt 5.3 mit einer Gegenprobe: dieselbe Architektur,
   erhöhte Patience, aber ohne Momentum. Erst dieser Vergleich rechtfertigt den Eingriff
   (siehe ADR-014).
3. **Eigener Scorer über `make_scorer(recall_score, labels=[2], average="macro")`.**
   `cross_validate` mit einem `scoring`-Dictionary steht im Kurs (K02_00), ein selbst gebauter
   Scorer nicht. Wir brauchen ihn, weil unsere fachliche Hauptmetrik der Recall *einer einzelnen*
   Klasse ist (ADR-002); sklearn bietet dafür keinen fertigen Namen. Über das Dictionary werden
   alle drei Metriken in einem Durchlauf und auf derselben Fold-Aufteilung berechnet.
4. **`Pipeline([("scaler", ...), ("model", ...)])` statt `make_pipeline(...)`.** Der Kurs nutzt
   durchgehend die Kurzform. Wir benennen die Schritte selbst, weil die Namen im Notebook mehrfach
   auftauchen und `model__C` als Parameterpfad lesbarer wäre als der automatisch erzeugte
   Kleinbuchstaben-Name der Klasse.
5. **Wilson-Konfidenzintervalle in Schritt 7.1.** Im Kurs nicht behandelt. Wir haben sie ergänzt,
   weil unsere kritische Klasse im Testset nur 26 Fälle hat und ein Punktschätzer wie „Recall
   1,000“ ohne Intervall irreführend wäre. Die Funktion ist im Notebook ausgeschrieben und nutzt
   nur `math`, es kommt also keine weitere Abhängigkeit hinzu.

**Konsequenzen:**
Die Abweichungen betreffen Handwerk, Reproduzierbarkeit und statistische Sorgfalt, nicht die
Methodik. Wer die Kursnotebooks kennt, erkennt jeden Schritt wieder. Die Begründungen stehen
zusätzlich als Kommentar an der jeweiligen Codestelle.
