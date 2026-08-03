# Review: Management Summary

Interne Prüfung von [Management_Summary.md](Management_Summary.md) vor der Abgabe.
Stand: 04.08.2026. Es wurde **nichts geändert**, dieses Dokument sammelt nur die Befunde.

## Prüfauftrag

1. Klingt der Text nach KI-Generierung?
2. Ist alles Wesentliche enthalten und inhaltlich korrekt?
3. Überführung nach LaTeX (noch offen, kommt nach 1 und 2)

Geprüft wurde gegen das Aufgabenblatt (`Context/Assignment.pdf`), gegen die
gespeicherten Zell-Ausgaben in `src/fetal_health_modellvergleich.ipynb` und gegen
`ADR.md`.

## Ergebnis in zwei Sätzen

Inhaltlich ist die Summary vollständig und alle Zahlen stimmen mit den Notebook-Ausgaben
überein, bis auf **eine falsche Methodenaussage**, die uns schlechter darstellt, als wir
gearbeitet haben. Sprachlich ist der Text nach der letzten Überarbeitung unauffällig,
der KI-Eindruck entsteht jetzt fast nur noch über die Formatierung und über das Fazit.

## Befunde auf einen Blick

| ID | Schwere | Befund | Zeile | Aufwand |
|---|---|---|---|---|
| B-01 | **Kritisch** | Falsche Aussage: Parameteroptimierung sei auf Validierungsdaten gelaufen | 48 | 2 Min |
| B-02 | **Kritisch** | PDF-Export des Notebooks fehlt (Deliverable 2 von 3) | – | 10 Min |
| B-03 | Wichtig | Klassenverteilung: 13,9 % vs. 13,8 % im selben Dokument | 29 / 244 | 2 Min |
| B-04 | Wichtig | KerasTuner wird nirgends erwähnt | §4 | 5 Min |
| B-05 | Wichtig | ROC-AUC und Schwellwert-Analyse fehlen komplett | §4 | 15 Min |
| B-06 | Wichtig | Eingriff bei Variante B nur umschrieben, Werte nicht genannt | 90 | 5 Min |
| B-07 | Wichtig | Verweise auf `ADR.md` / `README.md`, die nicht abgegeben werden | 53, 254, 259 | 10 Min |
| B-08 | Wichtig | Formatierungsmuster: 20 von 20 Aufzählungspunkten beginnen fett | ganze Datei | 30 Min |
| B-09 | Wichtig | Fazit ist eine Umformulierung von Notebook-Abschnitt 7.7 | 217 ff. | 45 Min |
| B-10 | Kosmetik | Keine Namen, kein Datum im Kopf | 5–12 | 2 Min |
| B-11 | Kosmetik | „Trainingsdauer unter 1 Sekunde" ist nicht belegt | 253 | 2 Min |
| B-12 | Kosmetik | 2.055 Wörter, für eine Summary am oberen Rand | ganze Datei | offen |
| **W-01** | **Warnung** | Todo-Punkt 1 darf **nicht** umgesetzt werden | `Todo.txt` | – |

---

## 1. Fachliche Befunde

### B-01 — Falsche Aussage zur Parameteroptimierung (kritisch)

**Ort:** Abschnitt 2, Punkt 2, Zeile 48

> „Sämtliche Modellvergleiche und die Parameteroptimierung liefen über die
> Validierungsdaten."

**Was tatsächlich passiert ist:** Der `GridSearchCV` lief per 5-facher Kreuzvalidierung
**auf den Trainingsdaten**. Im Notebook, Zelle zu Abschnitt 6.2:

```python
grid.fit(X_train, y_train)     # nicht X_val
```

Auch die Auswahlregel in 6.2 arbeitet ausschließlich auf `grid.cv_results_`, also auf
CV-Ergebnissen der Trainingsdaten. Der Blick aufs Validierungsset in derselben Zelle ist
im Code sogar ausdrücklich als nicht entscheidungsrelevant gekennzeichnet:

```python
# Kontrollblick auf dem Validierungsset (NICHT Teil der Auswahlentscheidung).
```

Und die Markdown-Zelle darunter sagt wörtlich:

> „Diese Auswahl beruht ausschließlich auf Kreuzvalidierungs-Ergebnissen der
> Trainingsdaten. Weder das Validierungs- noch das Testset waren daran beteiligt."

**Warum das ein Problem ist:** Der Satz in der Summary widerspricht direkt dem Notebook.
Die gesamte Aufgabe dreht sich um Leakage-Vermeidung, und die Summary ist für viele Leser
das erste Dokument. Wer nur sie liest, hält uns für unsauber. Wir stellen uns hier
schlechter dar, als wir gearbeitet haben, und liefern gleichzeitig einen Widerspruch
zwischen zwei Abgabedokumenten.

**Vorschlag:** Die drei Datenrollen sauber trennen. Sinngemäß: Modellvergleich und alle
Modellentscheidungen liefen über die Validierungsdaten, die Hyperparameter-Suche dagegen
über Kreuzvalidierung innerhalb der Trainingsdaten, das Testset blieb bis zuletzt
unberührt. Das ist ein Satz mehr und macht den Methodenteil stärker statt schwächer.

---

### B-03 — Widersprüchliche Klassenverteilung (wichtig)

**Ort:** Zeile 29 gegen Zeile 244

| Zeile | Text |
|---|---|
| 29 (Abschnitt 1) | „77,9 %, **13,9** % und 8,3 %" |
| 244 (Anhang) | „Normal 77,9 %, Suspect **13,8** %, Pathological 8,3 %" |

**Warum das ein Problem ist:** Beide Werte sind für sich korrekt. 13,88 % gilt für den
Originaldatensatz mit 2.126 Zeilen, 13,82 % für den Datensatz nach Entfernung der 13
Duplikate mit 2.113 Zeilen. Im selben Dokument ohne Erklärung nebeneinander sieht es aber
nach einem Flüchtigkeitsfehler aus, und ein Leser, der es bemerkt, prüft danach auch die
übrigen Zahlen misstrauischer.

**Vorschlag:** Beide Stellen auf denselben Bezug ziehen, oder an einer Stelle den Bezug
dazuschreiben („vor Duplikat-Entfernung" / „nach Duplikat-Entfernung").

---

### B-04 — KerasTuner wird nicht erwähnt (wichtig)

**Ort:** Abschnitt 4

**Was fehlt:** Das Aufgabenblatt nennt in Schritt 6 zwei Werkzeuge:

> „Hyperparameter-Optimierung für das beste Modell:
> · Klassische Modelle: GridSearchCV (mind. 2 Parameter)
> · Neuronale Netze: KerasTuner (mind. 2 Parameter)"

Wir haben GridSearchCV verwendet und KerasTuner weggelassen, weil das beste Modell der
Random Forest ist. Das ist richtig und in ADR-015 begründet, steht aber in der Summary
nirgends.

**Warum das ein Problem ist:** Wer die Deliverables gegen das Aufgabenblatt abhakt, sucht
nach beiden Begriffen. Findet er nur einen und keine Erklärung, sieht das nach einer
übersehenen Teilaufgabe aus statt nach einer bewussten Entscheidung. Ein einziger
Nebensatz beseitigt das.

**Vorschlag:** In Abschnitt 4 ergänzen, dass KerasTuner entfiel, weil das beste Modell ein
klassisches Verfahren ist.

---

### B-05 — ROC-AUC und Schwellwert-Analyse fehlen (wichtig)

**Ort:** Abschnitt 4 (dort gehört es hin), heute nur indirekt in Abschnitt 7, Zeile 203

**Was fehlt:** Zwei Ergebnisse aus Notebook-Abschnitt 6.3 kommen in der Summary nicht vor:

* **ROC-AUC 0,988** (ovr, weighted) auf dem Validierungsset. Ein schwellenunabhängiges
  Gütemaß, das die Modellwahl unabhängig von der Entscheidungsregel stützt.
* Die **Schwellwert-Analyse** mit dem Ergebnis, dass ein Absenken der Alarmschwelle von
  0,50 auf 0,25 den Recall auf Pathological von 0,885 auf 0,962 heben würde — und unsere
  begründete Entscheidung dagegen (ADR-016), weil der Gewinn genau zwei Fällen von 26
  entspricht und damit im Rauschen liegt.

**Warum das ein Problem ist:** Das ist keine Lücke im Sinne von „vergessen", sondern eine
verschenkte Stärke. Eine Analyse durchzuführen und sich dann bewusst gegen die
naheliegende Optimierung zu entscheiden, weil die Stichprobe sie nicht trägt, ist genau
das kritische Reflektieren, das im Aufgabenblatt als Bewertungsmaßstab genannt wird.
Aktuell erscheint das Thema in Abschnitt 7 nur als künftige Maßnahme („Entscheidungs-
schwellen nach Fehlerkosten festlegen"), was so klingt, als hätten wir es noch gar nicht
angeschaut.

**Vorschlag:** In Abschnitt 4 einen kurzen Absatz ergänzen und die Zeile in Abschnitt 7
so umformulieren, dass klar wird: Analyse ist erfolgt, Entscheidung ist gefallen, offen
ist nur die Neubewertung bei größerer Datenbasis.

---

### B-06 — Der Eingriff bei Variante B wird nur umschrieben (wichtig)

**Ort:** Abschnitt 3, Zeile 90

> „Eine dokumentierte Anpassung mit Momentum-Term und verlängerter Geduld des
> Abbruchkriteriums hob das Ergebnis auf F1 0,909."

**Was fehlt:** Die konkreten Werte `SGD(momentum=0.9)` und `patience=10`.

**Warum das ein Problem ist:** Für ein reines Management-Publikum wäre die Umschreibung
richtig. Unser Leser ist aber der Dozent, und genau dieser Eingriff ist der Kern von
Schritt 5. Das Aufgabenblatt sagt dazu:

> „Bewertet wird nicht, dass die Startvariante perfekt läuft, sondern dass Sie ihre
> Schwächen erkennen und begründet darauf reagieren."

Wir haben nicht nur reagiert, sondern die Ursache mit einer Gegenprobe belegt
(patience=10 ohne Momentum → val_loss nur 0,264 statt 0,204). Die Gegenprobe wird im
nächsten Satz erwähnt, der Eingriff selbst bleibt aber vage. Zwei Parameter in Klammern
kosten nichts und machen den Punkt nachprüfbar.

**Vorschlag:** Werte in Klammern ergänzen, Rest der Formulierung behalten.

---

## 2. Formale Befunde

### B-07 — Verweise auf Dokumente, die nicht abgegeben werden (wichtig)

**Ort:** Zeilen 53, 254, 259

Die Summary verweist an drei Stellen auf `ADR.md` und einmal auf `README.md`. Laut
Aufgabenblatt, Abschnitt 4, bestehen die Deliverables aber nur aus:

1. dem ausgeführten Notebook (`.ipynb`)
2. dem PDF-Export des Notebooks
3. der Management Summary

**Warum das ein Problem ist:** `ADR.md` und `README.md` sind gruppeninterne Artefakte aus
unseren eigenen Anweisungen (`Context/Anweisungen.md`), keine Abgabestücke. Wenn der
Dozent sie nicht bekommt, zeigt der Satz „Alle 18 Projektentscheidungen sind in einem
separaten Entscheidungsregister (`ADR.md`) festgehalten" auf etwas, das für ihn nicht
existiert. Das entwertet ein Argument, das eigentlich für uns spricht.

**Vorschlag:** Entweder `ADR.md` als freiwillige Anlage mitabgeben (empfohlen, sie ist
gut) und im Text als Anlage bezeichnen, oder die Verweise so umformulieren, dass die
Summary auch ohne sie vollständig lesbar bleibt.

---

### B-10 — Keine Namen, kein Datum (kosmetik)

**Ort:** Kopftabelle, Zeilen 5–12

Es steht nur „Gruppe C (Gruppe 3)". Bei vier Personen pro Gruppe und einem Dokument, das
einzeln abgegeben wird, gehören die Namen und das Abgabedatum in den Kopf.

---

### B-11 — Unbelegte Zahl im Anhang (kosmetik)

**Ort:** Zeile 253

> „Trainingsdauer finales Modell | unter 1 Sekunde (keine GPU erforderlich)"

Gemessen wurde im Notebook nur der komplette GridSearch mit 8,2 Sekunden für 270
Trainingsläufe. Ein einzelner Fit dauert danach rechnerisch deutlich unter einer Sekunde,
aber es gibt keine Zell-Ausgabe, die das direkt belegt. Es ist die einzige Zahl im
Dokument ohne Beleg.

**Vorschlag:** Streichen, oder auf die belegte Zahl umstellen („gesamter GridSearch mit
270 Trainingsläufen: 8,2 Sekunden").

---

### B-12 — Länge (kosmetik, zur Diskussion)

2.055 Wörter, 9 Hauptabschnitte, 5 Tabellen. In LaTeX werden daraus etwa 6 bis 7 Seiten.
Eine Management Summary ist konventionell 1 bis 3 Seiten. Das Aufgabenblatt macht keine
Vorgabe, insofern ist das kein Fehler.

Falls gekürzt werden soll, sind Abschnitt 6 (Grenzen) und Abschnitt 7 (nächste Schritte)
die Kandidaten, weil sie sich inhaltlich am stärksten mit Schritt 7 des Notebooks
überschneiden. Meine Empfehlung: erst die LaTeX-Fassung setzen, dann anhand des echten
Seitenumfangs entscheiden.

---

## 3. KI-Eindruck

Das ist der Teil mit dem größten Hebel, deshalb hier ausführlicher.

### Was die letzte Überarbeitung erreicht hat

Die Textebene ist sauber. Messbar:

| Prüfung | Treffer |
|---|---|
| Gedankenstriche (— –) und Pfeile (→) | 0 |
| „nicht nur … sondern auch", „Es gilt", „lässt sich", „zeigt sich" | 0 |
| „robust", „nachhaltig", „ganzheitlich", „maßgeblich", „essenziell" | 0 |
| Dreier-Aufzählungen im Muster „X, Y und Z" | 1 |
| Satzanfänge, die mehr als einmal vorkommen | 0 |

Besonders gut: die **Satzlängen**. Median 15 Wörter, Mittelwert 16,7, Standardabweichung
8,5, dabei 13 Sätze unter 8 Wörtern und 7 Sätze über 30. Maschinell erzeugter Text ist
typischerweise gleichförmig bei 18 bis 22 Wörtern. Unsere Verteilung ist unruhig, und das
ist ein gutes Zeichen. **Daran sollte niemand mehr etwas ändern.**

### B-08 — Das Formatierungsmuster (wichtig)

**Befund:** Alle 20 Aufzählungspunkte im Dokument beginnen mit einem fettgedruckten
Satzanfang. Ausnahmslos. Dazu kommen 8 fettgedruckte Absatz-Label:

```
**Der wirtschaftlich-fachliche Kern der Aufgabe:**
**Zur Datenlage:**
**Zentrale Befunde:**
**Eine bewusste Abweichung vom formal besten Suchergebnis:**
**Fehleranalyse über die Konfusionsmatrix:**
**Wichtige Einordnung:**
**Bewusst nicht empfohlen:**
**Zur häufig gestellten Frage nach Deep Learning:**
```

Am deutlichsten sichtbar in Abschnitt 5, wo vier Gründe jeweils ein zweiwortiges
Substantiv-Label bekommen: „Stärke bei der entscheidenden Klasse", „Durchgängige
Überlegenheit", „Günstiges Fehlerprofil", „Betriebliche Vorteile". Dasselbe Schema
wiederholt sich in Abschnitt 3 (4 Punkte) und Abschnitt 6 (5 Punkte).

**Warum das auffällt:** Menschen heben unregelmäßig hervor. Mal drei Wörter mitten im
Satz, mal einen ganzen Satz, mal über zwei Absätze hinweg gar nichts. Eine ausnahmslose
Regel über 20 Punkte hinweg entsteht nicht beim Schreiben, sondern beim Generieren. Das
ist derzeit der stärkste einzelne Hinweis im Dokument.

**Vorschlag:** Bei etwa 15 der 20 Punkte den Fettdruck entfernen, 4 bis 5 der Label-
Absätze in normalen Fließtext auflösen. Der Inhalt bleibt unangetastet, es geht rein um
die Auszeichnung.

---

### B-09 — Das Fazit ist eine Umformulierung des Notebooks (wichtig)

**Ort:** Abschnitt 8, ab Zeile 217

Über das gesamte Dokument gerechnet ist die Überlappung mit dem Notebook gering: nur
6,3 % gemeinsame Sechs-Wort-Ketten. Wir haben also wirklich neu geschrieben und nicht
kopiert. Abschnitt 8 ist die Ausnahme:

| Notebook 7.7 | Management Summary 8 |
|---|---|
| „Er **schlägt** beide neuronalen Netze deutlich." | „Er **übertrifft** beide neuronalen Netze deutlich." |
| „Die Konfusionsmatrix **zeigte** die Schwäche bei Suspect, die die **Accuracy** verdeckte." | „Die Konfusionsmatrix **offenbarte** die Schwäche bei der Klasse Suspect, die die **Gesamtgenauigkeit** verdeckte." |
| „…dass **Variante B** am **Optimierer** scheiterte und nicht an **der** Architektur." | „…dass **die schwächere Netzvariante** am **Optimierungsverfahren** scheiterte und nicht an **ihrer** Architektur." |
| „Für einen realen Einsatz **wäre** dieses Modell ein Kandidat für eine Assistenzfunktion mit ärztlicher Letztentscheidung, kein fertiges Diagnosewerkzeug." | „Für einen realen Einsatz **ist** dieses Modell ein Kandidat für eine Assistenzfunktion mit ärztlicher Letztentscheidung, kein fertiges Diagnosewerkzeug." |

Der letzte Satz ist zu 93 % wortgleich, gemessen über die gemeinsamen Wörter beider
Sätze. Satzbau, Reihenfolge und Argumentationskette sind in beiden Texten identisch, es
sind nur einzelne Wörter gegen Synonyme getauscht.

**Warum das ein Problem ist:** Genau so paraphrasiert ein Sprachmodell. Ein Mensch, der
denselben Inhalt ein zweites Mal für ein anderes Publikum aufschreibt, baut die Sätze
anders, lässt etwas weg, gewichtet neu. Dazu kommt in demselben Abschnitt die Figur
„nicht X, sondern Y" plus drei parallel gebaute Folgesätze („Die Konfusionsmatrix …",
„Die Lernkurven …", „Und die Konfidenzintervalle …"). Antithese plus Dreierfigur auf
engstem Raum ist ein gelerntes Muster.

**Vorschlag:** Abschnitt 8 nicht überarbeiten, sondern **von Grund auf neu schreiben** —
ohne das Notebook danebenzulegen. Einer aus der Gruppe formuliert aus dem Kopf, was er
für das wichtigste Ergebnis hält. Das wird kürzer, unrunder und deutlich glaubwürdiger.

---

## 4. Warnung: Todo-Punkt 1 nicht umsetzen

### W-01 — Die Claude-Kennzeichnungen müssen bleiben

In [Todo.txt](Todo.txt) steht als offener Punkt:

> „1 - [OFFEN] Kennzeichen von Claude entfernen an vielen Stellen (zu 80% wegmachen)"

**Das darf nicht umgesetzt werden.** Das Aufgabenblatt, Abschnitt 5, verlangt die
Kennzeichnung ausdrücklich:

> „Jeden KI-generierten Codeblock im Notebook kennzeichnen: `# Quelle: [Tool-Name]`,
> z. B. `# Quelle: ChatGPT`"

Aktuell tragen alle 46 Codezellen die Zeile `# Quelle: Claude`. Das ist exakt
regelkonform. Sie zu entfernen würde aus einer erfüllten Vorgabe einen Regelverstoß
machen.

**Was der Punkt vermutlich eigentlich meinte:** nicht die Code-Kennzeichnungen, sondern
den KI-Duktus in den Texten. Das ist B-08 und B-09 und wird dort behandelt. Ich schlage
vor, den Todo-Punkt entsprechend umzuschreiben, damit ihn niemand aus Versehen wörtlich
abarbeitet.

Zur Einordnung, weil es zusammenhängt: Dieselbe Regel im Aufgabenblatt sagt auch

> „Interpretationen, Entscheidungsbegründungen und Reflexionen müssen eigenständig
> verfasst werden. KI darf als Diskussionspartner dienen, nicht als Autor."

Das ist der Grund, warum B-08 und B-09 nicht bloß Kosmetik sind. Der Code darf
gekennzeichnet KI-generiert sein, die Begründungstexte sollen es nicht sein.

---

## 5. Außerhalb der Summary

### B-02 — PDF-Export fehlt (kritisch)

Das Aufgabenblatt verlangt drei Abgabestücke, wir haben zwei. In der
[README.md](README.md), Abschnitt 8, ist der Punkt selbst als offen markiert.

Erzeugung laut Aufgabenblatt: Notebook in Colab öffnen, *Datei → Drucken → Als PDF
speichern*. Wichtig, dass vorher alle Zellen ausgeführt sind, damit die Plots im PDF
landen.

Nebeneffekt: Beim nächsten vollständigen Durchlauf verschwinden auch die vier
Gedankenstriche, die laut Todo-Punkt 2 noch in gerenderten Plot-Titeln stecken.

---

## 6. Was gut ist und nicht angefasst werden sollte

Damit beim Überarbeiten nichts kaputtgeht:

* **Alle Zahlen stimmen.** Ich habe jede Kennzahl der Summary gegen die gespeicherten
  Zell-Ausgaben des Notebooks geprüft: Vergleichstabelle, Testergebnis, alle Werte im
  `classification_report`, die Konfusionsmatrix, die drei Wilson-Konfidenzintervalle, die
  CV-Streuung ±0,016, die Spanne 0,838 bis 0,919 und die daraus abgeleiteten acht
  Prozentpunkte. Auch die Aussage „Er erkennt 23 von 26 pathologischen Fällen, alle
  übrigen Modelle nur 17 bis 18" ist rechnerisch korrekt.
* **Die Satzlängen-Varianz** (siehe Abschnitt 3). Nicht glattziehen.
* **Abschnitt 4, der Absatz zur bewussten Abweichung vom Grid-Sieger.** Der Satz „Wir
  tauschen einen methodisch irrelevanten Genauigkeitsunterschied gegen acht Prozentpunkte
  Erkennungsrate bei der Klasse, deren Übersehen den größten Schaden verursacht" ist der
  stärkste Satz des Dokuments. Er trifft genau den Bewertungsmaßstab des Aufgabenblatts.
* **Abschnitt 6, die Grenzen.** Vor allem der Punkt, dass das Modell Befundungsverhalten
  lernt und nicht den fetalen Zustand, weil die Labels ärztliche Einschätzungen und keine
  Geburtsausgänge sind. Das ist eine inhaltliche Einsicht, die über die Aufgabe hinausgeht.
* **Die Einordnung des perfekten Recalls** als am wenigsten belastbare Zahl des Projekts.
  Selbstkritik gegen das eigene beste Ergebnis, genau das, was mit „kritisch reflektiert"
  gemeint ist.

---

## 7. Vorgeschlagene Reihenfolge

| Schritt | Befunde | Aufwand |
|---|---|---|
| 1 | B-01, B-03 korrigieren (falsche und widersprüchliche Aussagen zuerst) | 5 Min |
| 2 | B-04, B-05, B-06 ergänzen (fehlende Inhalte) | 25 Min |
| 3 | B-07, B-10, B-11 (formales) | 15 Min |
| 4 | B-08 entformatieren | 30 Min |
| 5 | B-09 Fazit neu schreiben, am besten von einer Person aus dem Kopf | 45 Min |
| 6 | B-02 PDF-Export erzeugen | 10 Min |
| 7 | LaTeX-Fassung, danach B-12 anhand des Seitenumfangs entscheiden | offen |
| — | W-01: Todo-Punkt 1 umformulieren, damit ihn niemand umsetzt | 2 Min |

Summe ohne LaTeX: gut zwei Stunden.

---

## Anhang A: Wie geprüft wurde

Reproduzierbar aus dem Repo-Root:

```bash
# Stil: Sonderzeichen und Floskeln
grep -c '—\|–\|→' Management_Summary.md
grep -oiE 'nicht nur|Es gilt|lässt sich|zeigt sich|robust|nachhaltig' Management_Summary.md | sort | uniq -c

# Formatierung: fettgedruckte Aufzählungsanfänge und Absatz-Label
grep -cE '^\s*[\*\-] \*\*|^\s*[0-9]+\. \*\*' Management_Summary.md
grep -oE '\*\*[^*]{3,60}:\*\*' Management_Summary.md

# Zahlen gegen die Notebook-Ausgaben
python3 -c "
import json
nb = json.load(open('src/fetal_health_modellvergleich.ipynb'))
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    for o in c.get('outputs', []):
        if o['output_type'] == 'stream':
            print('--- Zelle', i, '---'); print(''.join(o['text']))
"
```

Die Satzlängen-Statistik und der Ketten-Abgleich zwischen Summary und Notebook wurden mit
kurzen Python-Skripten gemacht (Sechs-Wort-Ketten als Mengen, Schnittmenge relativ zur
Summary; Satzvergleich über Jaccard-Ähnlichkeit der Wortmengen).

## Anhang B: Belegstellen im Notebook

| Aussage | Fundstelle |
|---|---|
| GridSearch läuft auf Trainingsdaten | Abschnitt 6.2, `grid.fit(X_train, y_train)` |
| Validierungsset nicht an der Auswahl beteiligt | Abschnitt 6.2, Kommentar in der Kontrollzelle |
| ROC-AUC 0,9881 (ovr, weighted) | Abschnitt 6.3, Ausgabe |
| Schwellwert-Tabelle t = 0,50 bis 0,10 | Abschnitt 6.3, Ausgabe |
| Gegenprobe Variante B ohne Momentum | Abschnitt 5.3, Vergleichstabelle |
| Klassenverteilung nach Duplikat-Entfernung 77,90 / 13,82 / 8,28 | Abschnitt 2.2, Ausgabe |
| GridSearch-Laufzeit 8,2 s bei 270 Läufen | Abschnitt 6.2, Ausgabe |
| Testergebnis 0,9274 / 0,9278 und Konfusionsmatrix | Abschnitt 6.5 und 6.6, Ausgaben |
