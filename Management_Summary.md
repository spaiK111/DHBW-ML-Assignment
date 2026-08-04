# Management Summary

## Systematischer Modellvergleich zur CTG-basierten Einstufung des fetalen Zustands

| | |
|---|---|
| Projekt | Assignment „Systematischer Modellvergleich mit klassischen Modellen und neuronalen Netzen, Mehrklassen-Klassifikation“ |
| Gruppe | C (Gruppe 3) |
| Kurs | Maschinelles Lernen mit Python, scikit-learn und Keras · DHBW Stuttgart · SS 2026 |
| Datensatz | Fetal Health Classification (Kaggle: `andrewmvd/fetal-health-classification`) |
| Umfang | 2.126 Aufzeichnungen, 21 numerische Merkmale, 3 Zielklassen |
| Ergebnis | Random Forest mit ausgeglichenen Klassengewichten, Testset: **Accuracy 92,7 %, F1 (weighted) 92,8 %** |

---

## 1. Aufgabe und Zielsetzung

Aus Kardiotokogramm-Daten (CTG), also der Aufzeichnung fetaler Herzfrequenz und Wehentätigkeit,
soll der Zustand eines Fötus automatisch in drei Klassen eingestuft werden: Normal, Suspect
(verdächtig) und Pathological (pathologisch). Die 21 Merkmale sind aus dem CTG-Signal abgeleitete
Kennzahlen wie Baseline-Herzfrequenz, Akzelerationen, Dezelerationen, Variabilitätsmaße und
Histogramm-Statistiken; die Klassenlabels wurden von Geburtsmedizinern vergeben.

Auftragsgemäß wurden fünf Modelle systematisch verglichen, drei klassische Verfahren und zwei
Varianten eines neuronalen Netzes. Maßstab war nicht das maximale Ergebnis, sondern ein
methodisch sauberes, nachvollziehbares und kritisch reflektiertes Vorgehen.

**Der wirtschaftlich-fachliche Kern der Aufgabe:** Die Klassen sind stark unausgewogen mit
77,9 %, 13,9 % und 8,3 %. Ein Modell, das ausnahmslos „Normal“ vorhersagt, erreicht damit bereits
77,9 % Genauigkeit, ohne einen einzigen kritischen Fall zu erkennen. Die Gesamtgenauigkeit ist als
Erfolgsmaß daher irreführend. Da die teuerste Fehlklassifikation ein übersehener pathologischer
Fall ist, haben wir vorab eine Bewertungsrangfolge festgelegt: zuerst der F1-Score (weighted),
dann der Recall der Klasse Pathological, und die Accuracy nur als Kontextgröße.

---

## 2. Vorgehen

Die Bearbeitung folgte den sieben vorgegebenen Schritten. Vier methodische Grundregeln haben wir
durchgängig eingehalten, um belastbare Aussagen zu sichern.

1. **Kein Data Leakage.** Die Merkmalsskalierung wurde ausschließlich auf den Trainingsdaten
   angepasst; innerhalb der Kreuzvalidierung erfolgte sie in einer Pipeline, sodass sie je
   Datenteil neu gelernt wird. 13 exakte Duplikat-Zeilen wurden vor der Datenaufteilung entfernt,
   damit identische Fälle nicht gleichzeitig zum Trainieren und zum Bewerten dienen.
2. **Getrennte Datenrollen.** Stratifizierte Aufteilung 70 / 15 / 15 in Trainings-,
   Validierungs- und Testdaten, also 1.479 / 317 / 317 Fälle. Sämtliche Modellvergleiche und die
   Parameteroptimierung liefen über die Validierungsdaten.
3. **Das Testset wurde genau einmal verwendet**, und zwar nach allen Entscheidungen. Damit ist das
   berichtete Ergebnis eine unverfälschte Schätzung der Leistung auf neuen Daten.
4. **Vollständige Entscheidungsdokumentation.** Alle 18 Projektentscheidungen sind mit Kontext,
   Begründung, geprüften Alternativen und Konsequenzen in einem separaten Entscheidungsregister
   (`ADR.md`) festgehalten.

**Zur Datenlage:** keine fehlenden Werte, keine kategorischen Textmerkmale, der
Vorverarbeitungsaufwand war entsprechend gering. Die Merkmalsanalyse identifizierte
Variabilitätsmaße und das Fehlen von Akzelerationen als stärkste Risikoindikatoren, was klinisch
etablierten Warnzeichen entspricht.

---

## 3. Ergebnisse des Modellvergleichs

Alle Modelle wurden zunächst mit Standardparametern trainiert und über eine 5-fache
stratifizierte Kreuzvalidierung sowie auf den Validierungsdaten bewertet.

| # | Modell | Typ | Accuracy | F1 (weighted) | Recall *Pathological* |
|---|---|---|---|---|---|
| 1 | Logistic Regression (multinomial) | Klassisch | 89,3 % | 0,890 | 0,654 |
| 2 | **Random Forest** | **Klassisch** | **94,6 %** | **0,945** | **0,885** |
| 3 | Support Vector Machine (RBF) | Klassisch | 90,2 % | 0,896 | 0,692 |
| 4 | Neuronales Netz Variante A | Neuronal | 89,0 % | 0,886 | 0,654 |
| 5 | Neuronales Netz Variante B (optimiert) | Neuronal | 91,2 % | 0,909 | 0,654 |

*Werte auf den Validierungsdaten (317 Fälle). Naive Vergleichsbasis „immer Normal“: 77,9 %.*

**Zentrale Befunde:**

* **Der Random Forest ist eindeutig das beste Modell**, er führt in allen drei Metriken
  gleichzeitig. Der Vorsprung ist bei der klinisch entscheidenden Klasse am größten: Er erkennt
  23 von 26 pathologischen Fällen, alle übrigen Modelle nur 17 bis 18.
* **Die neuronalen Netze setzen sich nicht durch.** Bei 1.479 Trainingsfällen und 21
  vorverarbeiteten Merkmalen spielen sie ihre Stärke nicht aus, nämlich das Lernen von
  Repräsentationen aus großen, unstrukturierten Datenmengen, während Baumverfahren auf genau
  dieser tabellarischen Datenform besonders leistungsfähig sind.
* **Die vorgegebene, tiefere Netzvariante B war zunächst schlechter als die einfachere Variante A**
  (F1 0,877 gegenüber 0,886). Die Analyse der Lernkurven zeigte, dass nicht die Architektur die
  Ursache war, sondern das Optimierungsverfahren: Das Training blieb in einem flachen Bereich
  stehen, der vom Abbruchkriterium als Konvergenz gedeutet wurde. Eine dokumentierte Anpassung mit
  Momentum-Term und verlängerter Geduld des Abbruchkriteriums hob das Ergebnis auf F1 0,909. Eine
  Gegenprobe belegte, dass allein das Abbruchkriterium nicht ursächlich war.
* **Bei den Modellwahlmöglichkeiten** haben wir uns methodisch begründet für den Random Forest
  entschieden, also gegen Einzelbaum und XGBoost, und für die Support Vector Machine, also gegen
  k-Nächste-Nachbarn. Beides wurde vorab aus der Datenanalyse abgeleitet und nicht durch
  nachträgliche Auswahl des besten Ergebnisses.

---

## 4. Optimierung und finale Bewertung

Für das beste Modell wurde eine systematische Parametersuche über vier Parameter durchgeführt,
also 54 Kombinationen mal 5 Datenteile, bei einer Rechenzeit von unter 10 Sekunden.

**Eine bewusste Abweichung vom formal besten Suchergebnis:** Die fünf besten Konfigurationen lagen
im F1-Score nur 0,001 auseinander, ein Unterschied weit innerhalb der natürlichen Streuung von
± 0,016 und damit statistisch bedeutungslos. Im Recall der kritischen Klasse unterschieden sie
sich dagegen erheblich, nämlich zwischen 0,838 und 0,919. Gemäß unserer vorab festgelegten
Bewertungsrangfolge haben wir daher die Konfiguration mit ausgeglichenen Klassengewichten gewählt:
Wir tauschen einen methodisch irrelevanten Genauigkeitsunterschied gegen acht Prozentpunkte
Erkennungsrate bei der Klasse, deren Übersehen den größten Schaden verursacht.

### Ergebnis auf den Testdaten (einmaliger, abschließender Zugriff)

**Accuracy 92,7 %, F1 (weighted) 92,8 %**

| Klasse | Precision | Recall | F1 | Fälle | 95-%-Konfidenzintervall (Recall) |
|---|---|---|---|---|---|
| Normal | 0,963 | 0,951 | 0,957 | 247 | [0,917; 0,972] |
| **Suspect** | 0,733 | **0,750** | **0,742** | 44 | [0,606; 0,854] |
| Pathological | 0,929 | **1,000** | 0,963 | 26 | **[0,871; 1,000]** |

**Fehleranalyse über die Konfusionsmatrix:** Von 317 Testfällen wurden 23 falsch eingestuft.

* **Kein pathologischer Fall wurde als „Normal“ eingestuft**, die schwerwiegendste Fehlerart trat
  also nicht auf. Alle 26 pathologischen Fälle wurden erkannt.
* **Häufigste Verwechslung ist Normal gegen Suspect** mit 12 plus 9, also 21 der 23 Fehler. Das
  entspricht dem auch klinisch bestehenden Graubereich zwischen „noch normal“ und „schon
  verdächtig“.
* **Schwächste Klasse ist Suspect** mit einer Erkennungsrate von 75,0 %.

**Wichtige Einordnung:** Wir haben die statistische Unsicherheit bewusst quantifiziert. Der Wert
„alle 26 pathologischen Fälle erkannt“ ist bei dieser Stichprobengröße mit einer tatsächlichen
Erkennungsrate von 90 % gut vereinbar. Von einer perfekten Erkennung darf also nicht gesprochen
werden; es ist das am wenigsten belastbare Einzelergebnis des Projekts.

---

## 5. Empfehlung

**Für den Produktiveinsatz empfehlen wir den Random Forest mit ausgeglichenen Klassengewichten,
und zwar ausschließlich als Assistenzsystem mit ärztlicher Letztentscheidung, nicht als autonomes
Diagnosewerkzeug.**

Die tragenden Gründe:

1. **Stärke bei der entscheidenden Klasse.** Kein pathologischer Fall wurde als unauffällig
   durchgewinkt, und das ist der einzige Fehlertyp mit potenziell gravierenden Folgen.
2. **Durchgängige Überlegenheit.** Das Modell lag in jeder Auswertung vorn, also in
   Kreuzvalidierung, Validierung und Test. Die Empfehlung stützt sich auf Konsistenz und nicht auf
   einen einzelnen Zahlenvorsprung.
3. **Günstiges Fehlerprofil.** Die verbleibenden Fehler sind überwiegend *vorsichtige* Fehler, sie
   erzeugen zusätzlichen Prüfaufwand und keinen medizinischen Schaden.
4. **Betriebliche Vorteile.** Training in unter einer Sekunde, keine spezielle Hardware, kein
   Skalierungsschritt und damit eine Fehlerquelle weniger im Betrieb. Über Merkmalswichtigkeiten
   ist das Modell teilweise erklärbar, was in der Medizin ein reales Akzeptanzargument ist, denn
   es muss nachvollziehbar sein, worauf ein Alarm beruht.

**Bewusst nicht empfohlen:** Die neuronalen Netze bringen bei dieser Datenform keinen
Leistungsvorteil, erfordern aber deutlich mehr Trainingsaufwand, sind schlechter erklärbar und
ziehen eine schwergewichtige technische Abhängigkeit in den Betrieb. Aus Wirtschaftlichkeitssicht
ist das ein schlechtes Tauschgeschäft.

---

## 6. Grenzen und Risiken

Diese Einschränkungen sind für eine Einsatzentscheidung wesentlich:

* **Die Klasse Suspect ist unzuverlässig** mit einer Erkennung von 75,0 % und einer
  Treffergenauigkeit von 73,3 %, und ausgerechnet sie ist am ehesten handlungsleitend, denn es
  geht um die Frage „beobachten oder nicht?“. Neun von 44 verdächtigen Fällen würden aus der
  Nachbeobachtung fallen. Dies ist die wichtigste offene Baustelle.
* **Das Modell lernt Befundungsverhalten, nicht den fetalen Zustand.** Die Zielwerte sind
  ärztliche Einschätzungen des CTG und nicht der tatsächliche Geburtsausgang. Das Modell
  reproduziert damit bestenfalls die Urteilsqualität der Befundenden einschließlich deren
  systematischer Fehler. Für eine Aussage über den klinischen Nutzen wären Ergebnisdaten
  erforderlich, etwa Apgar-Score oder Nabelschnur-pH, die nicht vorliegen.
* **Die Datenbasis ist für ein Qualitätsversprechen zu klein.** 26 pathologische Testfälle
  erlauben keine belastbare Leistungsgarantie.
* **Keine geprüfte Übertragbarkeit.** Alle Daten stammen aus einer Quelle mit einheitlicher
  Befundungspraxis. Ob das Modell auf anderen Geräten, in anderen Kliniken oder bei anderen
  Patientinnenpopulationen vergleichbar funktioniert, ist ungeprüft, und das ist erfahrungsgemäß
  die häufigste Bruchstelle medizinischer KI-Systeme.
* **Regulatorik und Anwendungsrisiko.** Ein solches System wäre ein Medizinprodukt mit
  entsprechenden Zulassungspflichten. Zudem besteht die Gefahr, dass ein unauffälliges
  Modellergebnis die eigene kritische Prüfung ersetzt. Beides bestärkt die Positionierung als
  reine Zweitmeinung.

---

## 7. Empfohlene nächste Schritte

**Der Engpass ist die Datenlage und die fachliche Validierung, nicht die Technik.** Der gesamte
Optimierungsdurchlauf benötigte weniger als zehn Sekunden Rechenzeit, zusätzliche Hardware würde
das Ergebnis nicht verbessern. Konkret:

| Priorität | Maßnahme | Erwarteter Nutzen |
|---|---|---|
| **1** | **Datenbasis ausweiten**, insbesondere pathologische Fälle | Für eine Aussagegenauigkeit von ± 5 Prozentpunkten wären ca. 73 pathologische Testfälle nötig, das entspricht rund 5.900 Aufzeichnungen gesamt (heute: 26 bzw. 2.113) |
| **2** | **Daten aus mehreren Kliniken** einbeziehen | Erst damit ist Übertragbarkeit überhaupt prüfbar |
| **3** | **Validierung an echten Geburtsausgängen** | Verlagert den Maßstab von „stimmt mit dem Befund überein“ auf tatsächlichen klinischen Nutzen |
| **4** | **Gezielte Arbeit an der Klasse Suspect**, z. B. zweistufige Klassifikation | Behebt die wesentliche Modellschwäche |
| **5** | **Entscheidungsschwellen nach Fehlerkosten** festlegen statt nur Klassengewichtung | Erlaubt fachlich begründete Steuerung des Kompromisses zwischen Fehlalarmen und übersehenen Fällen |
| **6** | **Wiederholte bzw. geschachtelte Kreuzvalidierung**, Modellkalibrierung, XGBoost als Alternative | Methodische Absicherung, moderater Zusatznutzen |

**Zur häufig gestellten Frage nach Deep Learning:** Auf den vorliegenden Daten würden ein CNN oder
Transfer Learning nicht helfen, denn die 21 Merkmale sind eine ungeordnete Menge aggregierter
Kennzahlen ohne die räumliche oder zeitliche Nachbarschaftsstruktur, die solche Verfahren
ausnutzen. Auf dem rohen CTG-Zeitsignal wäre ein 1D-CNN dagegen aussichtsreich: Es könnte Muster
erfassen, die in keiner der Kennzahlen enthalten sind, etwa die zeitliche Kopplung zwischen einer
Wehe und der darauf folgenden Herzfrequenzabsenkung, klinisch ein zentrales
Unterscheidungsmerkmal. Diese Information ist im aggregierten Datensatz unwiederbringlich
verloren.

---

## 8. Fazit

Fünf Modelle wurden nach einem einheitlichen, leakage-freien Protokoll verglichen. Ein Random
Forest mit ausgeglichenen Klassengewichten erreicht auf ungesehenen Testdaten 92,7 % Genauigkeit
bei einem F1-Score (weighted) von 92,8 % und erkennt alle pathologischen Fälle der Stichprobe. Er
übertrifft beide neuronalen Netze deutlich.

Der wesentliche Erkenntnisgewinn des Projekts lag jedoch nicht in der besten Kennzahl, sondern in
der genauen Fehleranalyse. Die Konfusionsmatrix offenbarte die Schwäche bei der Klasse *Suspect*,
die die Gesamtgenauigkeit verdeckte. Die Lernkurven zeigten, dass die schwächere Netzvariante am
Optimierungsverfahren scheiterte und nicht an ihrer Architektur. Und die Konfidenzintervalle
machten deutlich, dass das eindrucksvollste Einzelergebnis, nämlich die vollständige Erkennung
aller pathologischen Fälle, die am wenigsten belastbare Zahl des gesamten Projekts ist.

Für einen realen Einsatz ist dieses Modell ein Kandidat für eine Assistenzfunktion mit ärztlicher
Letztentscheidung, kein fertiges Diagnosewerkzeug. Der Weg dorthin führt über mehr und breitere
Daten sowie über eine Validierung an tatsächlichen Geburtsausgängen, nicht über ein größeres
Modell.

---

### Anhang: Kennzahlen im Überblick

| Kenngröße | Wert |
|---|---|
| Aufzeichnungen (nach Duplikatentfernung) | 2.113 |
| Merkmale | 21 (alle numerisch), keine fehlenden Werte |
| Klassenverteilung | Normal 77,9 %, Suspect 13,8 %, Pathological 8,3 % |
| Datenaufteilung | 1.479 / 317 / 317 (stratifiziert 70 / 15 / 15) |
| Verglichene Modelle | 5 (3 klassische, 2 neuronale Netze) |
| Naive Vergleichsbasis | 77,9 % Genauigkeit |
| **Finales Modell** | **Random Forest, ausgeglichene Klassengewichte** |
| **Testergebnis** | **Accuracy 92,7 %, F1 (weighted) 92,8 %** |
| Erkennungsrate Pathological | 26 von 26, Konfidenzintervall [0,871; 1,000] |
| Erkennungsrate Suspect | 33 von 44 (75,0 %) |
| Wichtigste Merkmale | Kurzzeit-Variabilität (14,1 %), Anteil abnormer Langzeit-Variabilität (12,3 %) |
| Trainingsdauer finales Modell | unter 1 Sekunde (keine GPU erforderlich) |
| Dokumentierte Entscheidungen | 18 (vollständig in `ADR.md`) |

---

*Zugehörige Dokumente: `src/fetal_health_modellvergleich.ipynb` (vollständig ausgeführtes Notebook
mit allen sieben Bearbeitungsschritten), `ADR.md` (Entscheidungsregister) und `README.md`
(Projektdokumentation und Einrichtung).*

*Hinweis zur KI-Nutzung: KI-Werkzeuge wurden entsprechend den Vorgaben des Assignments für Code,
Debugging und als Diskussionspartner eingesetzt; alle KI-generierten Codeblöcke sind im Notebook
gekennzeichnet. Bewertung, Empfehlung und Reflexion verantwortet die Gruppe.*
