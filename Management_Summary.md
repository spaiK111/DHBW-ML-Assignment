# Management Summary

## Modellvergleich zur Einstufung des fetalen Zustands aus CTG-Daten

Diese Zusammenfassung gehört zum Assignment „Systematischer Modellvergleich mit klassischen
Modellen und neuronalen Netzen, Mehrklassen-Klassifikation“ aus dem Kurs Maschinelles Lernen mit
Python, scikit-learn und Keras.

Bearbeitet von: *[Namen der Gruppenmitglieder eintragen]*

## 1. Worum es geht

Ein Kardiotokogramm zeichnet die Herzfrequenz des Fötus und die Wehentätigkeit der Mutter auf. Aus
21 Kennzahlen, die aus so einer Aufzeichnung berechnet werden, soll ein Modell den Zustand des
Fötus in drei Klassen einordnen: Normal, Suspect (also verdächtig) und Pathological (also
pathologisch). Bei den Kennzahlen handelt es sich zum Beispiel um die Baseline-Herzfrequenz, um
Akzelerationen und Dezelerationen, um verschiedene Maße für die Variabilität und um Statistiken
über das Herzfrequenz-Histogramm. Die Klassen selbst haben Geburtsmediziner vergeben, die die
Aufzeichnungen befundet haben.

Der Datensatz stammt von Kaggle und enthält 2.126 Aufzeichnungen. Nach dem Entfernen von 13
exakten Duplikaten bleiben davon 2.113 übrig. Vorgegeben war, fünf Modelle zu vergleichen, drei
klassische Verfahren und zwei Varianten eines neuronalen Netzes.

Wichtig für das Verständnis aller folgenden Zahlen ist die Verteilung der Klassen. Etwa 77,9
Prozent der Fälle sind normal, 13,8 Prozent verdächtig und 8,3 Prozent pathologisch. Ein Modell,
das einfach immer „Normal“ vorhersagt, kommt damit schon auf 77,9 Prozent Genauigkeit, ohne einen
einzigen kritischen Fall zu finden. Als Erfolgsmaß taugt die Accuracy hier also nicht. Deshalb
haben wir bereits vor dem ersten Training festgelegt, woran wir die Modelle messen. An erster
Stelle steht der gewichtete F1-Score, an zweiter der Recall der Klasse Pathological, und die
Accuracy nehmen wir nur noch zur Einordnung dazu. Der teuerste Fehler ist ein übersehener
pathologischer Fall, und genau das bildet diese Reihenfolge ab.

## 2. Wie wir vorgegangen sind

Die sieben vorgegebenen Schritte haben wir der Reihe nach abgearbeitet. Vier Punkte waren uns
dabei besonders wichtig.

Der erste ist, dass keine Information aus den Testdaten ins Training durchsickert. Die Skalierung
der Merkmale haben wir nur auf den Trainingsdaten angepasst, und innerhalb der Kreuzvalidierung
steckt sie in einer Pipeline, damit sie für jeden Datenteil neu gelernt wird. Die 13 Duplikate
haben wir vor dem Aufteilen entfernt, sonst hätte dieselbe Zeile gleichzeitig zum Trainieren und
zum Bewerten gedient.

Der zweite ist die saubere Trennung der drei Datenrollen. Aufgeteilt haben wir stratifiziert, also
unter Beibehaltung der Klassenverteilung: 70 Prozent der Daten gehen ins Training, die restlichen
30 Prozent zu gleichen Teilen in Validierung und Test. In Fällen gerechnet sind das 1.479 für das
Training und jeweils 317 für Validierung und Test. Der Modellvergleich und die Entscheidung,
welches Modell wir weiterverfolgen, liefen über die Validierungsdaten. Die spätere Suche nach den
besten Einstellungen lief per fünffacher Kreuzvalidierung nur innerhalb der Trainingsdaten, das
Validierungsset war daran nicht beteiligt.

Der dritte Punkt ist, dass wir das Testset genau einmal angefasst haben, ganz am Ende und nach
allen Entscheidungen. Damit ist das berichtete Ergebnis eine ehrliche Schätzung für neue Daten und
kein Wert, auf den wir vorher hin optimiert haben.

Und viertens haben wir jede Entscheidung schriftlich begründet, insgesamt 18 Stück, jeweils mit
den geprüften Alternativen und den Konsequenzen.

Die Daten selbst machten wenig Arbeit. Es fehlen keine Werte, es gibt keine Textspalten, alles ist
numerisch. Bei der Merkmalsanalyse sind uns die Variabilitätsmaße und das Fehlen von
Akzelerationen als stärkste Warnzeichen aufgefallen. Das passt zu dem, was in der Geburtsmedizin
ohnehin als kritisch gilt, und wir haben das als gutes Zeichen für die Datenqualität gewertet.

## 3. Der Vergleich der fünf Modelle

Zuerst haben wir alle fünf Modelle mit den Standardeinstellungen trainiert, per fünffacher
stratifizierter Kreuzvalidierung geprüft und danach auf den 317 Validierungsfällen bewertet. Das
kam dabei heraus:

- Die logistische Regression erreicht 89,3 Prozent Accuracy bei einem gewichteten F1-Score von
  0,890 und erkennt 65,4 Prozent der pathologischen Fälle.
- Der Random Forest kommt auf 94,6 Prozent Accuracy, einen F1-Score von 0,945 und einen Recall von
  88,5 Prozent auf der Klasse Pathological.
- Die Support Vector Machine mit RBF-Kernel liegt bei 90,2 Prozent Accuracy, einem F1-Score von
  0,896 und einem Recall von 69,2 Prozent.
- Variante A des neuronalen Netzes schafft 89,0 Prozent Accuracy, einen F1-Score von 0,886 und
  einen Recall von 65,4 Prozent.
- Variante B des neuronalen Netzes kommt nach der Optimierung auf 90,2 Prozent Accuracy, einen
  F1-Score von 0,899 und einen Recall von 69,2 Prozent.

Zum Vergleich noch einmal der Referenzpunkt: Ein Modell, das immer Normal vorhersagt, erreicht
77,9 Prozent Accuracy und einen Recall von null auf der kritischen Klasse.

Der Random Forest führt in allen drei Metriken gleichzeitig, und am deutlichsten bei der Klasse,
auf die es wirklich ankommt. Er erkennt 23 der 26 pathologischen Validierungsfälle, alle anderen
Modelle nur 17 oder 18. Zusätzlich haben wir für ihn die ROC-AUC berechnet, weil dieses Maß
unabhängig von der gewählten Entscheidungsschwelle ist. Sie liegt bei 0,988, berechnet nach dem
One vs Rest Verfahren und nach Klassenhäufigkeit gewichtet. Der Vorsprung hängt also nicht an einer
zufällig günstigen Schwelle.

Dass die neuronalen Netze nicht gewinnen, hat uns am Anfang gewundert. Bei 1.479 Trainingsbeispielen
und 21 fertig aufbereiteten Kennzahlen können sie ihre eigentliche Stärke aber gar nicht ausspielen,
nämlich aus großen und unstrukturierten Datenmengen selbst brauchbare Repräsentationen zu lernen.
Für tabellarische Daten in dieser Größenordnung sind Baumensembles einfach das passendere Werkzeug.

Am meisten gelernt haben wir bei Variante B, also der vorgegebenen tieferen Architektur. Sie war
zunächst schlechter als die einfachere Variante A, mit einem F1-Score von 0,877 gegenüber 0,886. An
der Architektur lag das aber nicht. Die Lernkurven zeigten, dass das Training in einem flachen
Bereich hängen geblieben ist und das Abbruchkriterium diesen Stillstand für Konvergenz gehalten
hat. Wir haben daraufhin dem Optimierer SGD ein Momentum von 0,9 mitgegeben und die Geduld des
Abbruchkriteriums von fünf auf zehn Epochen erhöht. Danach stieg der F1-Score auf 0,899, und der
Fehler auf den Validierungsdaten fiel von 0,273 auf 0,208. Um sicherzugehen, woran es lag, haben
wir eine Gegenprobe gemacht: allein mit der längeren Geduld und ohne Momentum kommt das Netz nur
auf 0,264. Entscheidend war also das Momentum.

Bei den beiden Modellen, die wir frei wählen durften, haben wir uns für den Random Forest
entschieden und damit gegen einen einzelnen Entscheidungsbaum und gegen XGBoost, außerdem für die
Support Vector Machine und damit gegen k-Nächste-Nachbarn. Beides haben wir vorher aus der
Datenanalyse begründet und nicht im Nachhinein danach ausgesucht, was am besten abgeschnitten hat.

## 4. Die Optimierung des Random Forest

Für den Random Forest haben wir eine Gittersuche über vier Einstellungen laufen lassen. Das sind 54
Kombinationen, jede davon auf fünf Datenteilen, also 270 Trainingsläufe in gut 32 Sekunden.
Gerechnet wurde per Kreuzvalidierung auf den Trainingsdaten. KerasTuner, das im Aufgabenblatt für
neuronale Netze vorgesehen ist, haben wir nicht gebraucht, weil das beste Modell ein klassisches
Verfahren ist.

An einer Stelle sind wir bewusst vom formal besten Suchergebnis abgewichen. Die fünf besten
Kombinationen unterscheiden sich im F1-Score nur um 0,001. Da die Kreuzvalidierung ohnehin um etwa
0,016 schwankt, ist so ein Unterschied bedeutungslos. Im Recall der kritischen Klasse liegen sie
dagegen weit auseinander, nämlich zwischen 0,838 und 0,919. Nach unserer vorher festgelegten
Reihenfolge haben wir deshalb die Variante mit ausgeglichenen Klassengewichten genommen. Wir
tauschen damit einen Genauigkeitsunterschied, der methodisch keine Rolle spielt, gegen acht
Prozentpunkte Erkennungsrate bei der Klasse, deren Übersehen den größten Schaden anrichtet. Damit
unsere Begründung später nicht von der tatsächlich verwendeten Einstellung abweichen kann, steht
diese Auswahlregel im Notebook als ausführbarer Code und nicht als abgetippte Liste von Werten.

Geprüft haben wir außerdem, ob es sich lohnt, die Alarmschwelle abzusenken. Wenn ein Fall schon ab
einer Wahrscheinlichkeit von 0,25 statt der üblichen 0,50 als pathologisch gemeldet wird, steigt
der Recall auf dieser Klasse von 0,885 auf 0,962. Dagegen entschieden haben wir uns trotzdem, und
zwar aus drei Gründen. Der Gewinn entspricht zwei Fällen von 26 und liegt damit im Rauschen. Eine
Schwelle, die an 26 Fällen kalibriert wurde, wäre mit ziemlicher Sicherheit nur an diese eine
Stichprobe angepasst. Und die Klassengewichtung zieht bereits in dieselbe Richtung, wir würden den
Effekt also doppelt anwenden.

## 5. Das Ergebnis auf den Testdaten

Beim einmaligen und abschließenden Blick auf das Testset kam eine Accuracy von 92,7 Prozent heraus,
bei einem gewichteten F1-Score von 92,8 Prozent. Aufgeschlüsselt nach Klassen sieht es so aus:

- Bei Normal, immerhin 247 Fälle, liegt die Precision bei 0,963 und der Recall bei 0,951, der
  F1-Score beträgt 0,957.
- Bei Suspect, 44 Fälle, liegt die Precision bei 0,733 und der Recall bei 0,750, der F1-Score
  beträgt 0,742.
- Bei Pathological, 26 Fälle, liegt die Precision bei 0,929 und der Recall bei 1,000, der F1-Score
  beträgt 0,963.

Von den 317 Testfällen wurden 23 falsch eingeordnet. Kein einziger pathologischer Fall wurde dabei
als normal eingestuft, die schlimmste Fehlerart ist also überhaupt nicht aufgetreten. 21 der 23
Fehler entfallen auf die Verwechslung von Normal und Suspect, davon 12 in die eine und 9 in die
andere Richtung. Diesen Graubereich zwischen „noch normal“ und „schon verdächtig“ gibt es auch
klinisch. Am schlechtesten erkennt das Modell die Klasse Suspect, dort trifft es 33 von 44 Fällen.

Beim perfekten Recall auf Pathological muss man aber ehrlich bleiben. Wir haben dafür ein
95-Prozent-Konfidenzintervall nach Wilson berechnet, und das reicht von 0,871 bis 1,000. Bei nur 26
Fällen ist „26 von 26 erkannt“ also auch mit einer wahren Erkennungsrate von 90 Prozent gut
vereinbar. Von einer perfekten Erkennung sollte deshalb niemand reden. Es ist die am wenigsten
belastbare Zahl im ganzen Projekt, obwohl sie auf den ersten Blick die schönste ist.

## 6. Unsere Empfehlung

Wir empfehlen den Random Forest mit ausgeglichenen Klassengewichten, und zwar ausschließlich als
Assistenzsystem, bei dem am Ende ein Arzt entscheidet. Als eigenständiges Diagnosewerkzeug ist er
nicht geeignet.

Dafür sprechen vier Gründe. Erstens wurde kein pathologischer Fall als unauffällig durchgewinkt,
und das ist der einzige Fehlertyp, der wirklich gefährlich werden kann. Zweitens lag das Modell in
jeder einzelnen Auswertung vorn, in der Kreuzvalidierung, auf den Validierungsdaten und auf dem
Testset. Unsere Empfehlung hängt also nicht an einem einzelnen Zahlenvorsprung. Drittens sind die
übrig gebliebenen Fehler überwiegend die vorsichtige Sorte. Sie erzeugen zusätzlichen Prüfaufwand,
aber keinen medizinischen Schaden. Und viertens ist der Betrieb unkompliziert. Es braucht keine
spezielle Hardware und keinen Skalierungsschritt, das ist eine Fehlerquelle weniger. Über die
Merkmalswichtigkeiten kann man außerdem zumindest teilweise nachvollziehen, worauf eine Einstufung
beruht. Gerade in der Medizin ist das ein echtes Argument, weil erklärbar sein muss, warum ein
Alarm ausgelöst wird.

Von den neuronalen Netzen raten wir ab. Auf dieser Art von Daten bringen sie keinen Vorteil bei der
Leistung, kosten aber deutlich mehr Trainingsaufwand, sind schlechter erklärbar und ziehen eine
schwergewichtige technische Abhängigkeit in den Betrieb. Wirtschaftlich ist das ein schlechtes
Tauschgeschäft.

## 7. Grenzen und Risiken

Die Klasse Suspect erkennt das Modell nur zu 75,0 Prozent, und von den als verdächtig gemeldeten
Fällen stimmen 73,3 Prozent. Ausgerechnet an dieser Klasse hängt aber die praktisch wichtigste
Frage, nämlich ob weiter beobachtet wird oder nicht. Neun von 44 verdächtigen Fällen würden aus der
Nachbeobachtung herausfallen. Das ist aus unserer Sicht die wichtigste offene Baustelle.

Grundsätzlich ist ein zweiter Punkt. Das Modell lernt Befundungsverhalten und nicht den Zustand
des Fötus. Die Zielwerte sind ärztliche Einschätzungen des CTG und nicht der tatsächliche Ausgang
der Geburt. Im besten Fall reproduziert das Modell also die Urteilsqualität derjenigen, die
befundet haben, samt deren systematischer Fehler. Um etwas über den klinischen Nutzen sagen zu
können, bräuchte man Ergebnisdaten wie den Apgar-Score oder den Nabelschnur-pH, und die liegen
schlicht nicht vor.

Dazu kommen noch drei Einschränkungen:

- Die Datenbasis ist zu klein für ein Qualitätsversprechen. Auf 26 pathologischen Testfällen kann
  man keine Leistungsgarantie aufbauen.
- Ob das Modell übertragbar ist, wissen wir nicht. Alle Daten stammen aus einer Quelle mit
  einheitlicher Befundungspraxis. Wie es auf anderen Geräten, in anderen Kliniken oder bei anderen
  Patientinnengruppen aussieht, haben wir nie geprüft. Erfahrungsgemäß ist genau das die häufigste
  Stelle, an der medizinische KI-Systeme scheitern.
- So ein System wäre ein Medizinprodukt und damit zulassungspflichtig. Außerdem besteht die Gefahr,
  dass ein unauffälliges Modellergebnis die eigene kritische Prüfung ersetzt. Beides spricht dafür,
  es wirklich nur als Zweitmeinung einzusetzen.

## 8. Was als Nächstes sinnvoll wäre

Der Engpass sind die Daten und die fachliche Validierung, nicht die Technik. Die komplette
Optimierung hat gut eine halbe Minute gerechnet, mehr Rechenleistung würde am Ergebnis nichts
ändern. Sinnvoll wäre aus unserer Sicht Folgendes, ungefähr in dieser Reihenfolge:

1. Mehr Daten sammeln, vor allem pathologische Fälle. Damit die Aussage auf etwa fünf Prozentpunkte
   genau wird, bräuchte man rund 73 pathologische Testfälle. Hochgerechnet sind das etwa 5.900
   Aufzeichnungen statt der heutigen 2.113.
2. Daten aus mehreren Kliniken einbeziehen, weil man die Übertragbarkeit sonst gar nicht prüfen
   kann.
3. Gegen echte Geburtsausgänge validieren statt gegen Befunde. Damit verschiebt sich der Maßstab
   von „stimmt mit der ärztlichen Einschätzung überein“ auf den tatsächlichen klinischen Nutzen.
4. Gezielt an der Klasse Suspect arbeiten, zum Beispiel mit einer zweistufigen Klassifikation.
5. Die Frage nach der Alarmschwelle neu bewerten, sobald mehr pathologische Fälle vorliegen. Die
   Analyse dazu haben wir in Abschnitt 4 bereits gemacht, nur trägt die heutige Datenmenge die
   Entscheidung noch nicht.
6. Methodisch nachschärfen, etwa mit wiederholter oder geschachtelter Kreuzvalidierung, mit einer
   Kalibrierung der Wahrscheinlichkeiten und mit XGBoost als weiterem Vergleichspunkt.



## 9. Fazit

Am meisten gelernt haben wir nicht beim Trainieren, sondern beim genauen Hinsehen, was die Modelle
eigentlich falsch machen. Die 92,7 Prozent Accuracy sind ein ordentliches Ergebnis, für sich
genommen sagen sie aber wenig. Erst die Konfusionsmatrix hat gezeigt, dass die Klasse Suspect das
eigentliche Problem ist, und erst das Konfidenzintervall hat klargemacht, dass unsere schönste Zahl
gleichzeitig unsere unsicherste ist.

Der Random Forest hat beide neuronalen Netze klar geschlagen. Das war für uns die zweite
Überraschung, weil man bei einer medizinischen Fragestellung zuerst an neuronale Netze denkt. Hier
hätten sie vor allem Aufwand gekostet.

Bleibt die Einordnung für einen echten Einsatz. Ein größeres oder komplizierteres Modell hätte an
unseren Ergebnissen wenig geändert. Was fehlt, sind mehr Daten und vor allem Zielwerte, die den
tatsächlichen Ausgang einer Geburt abbilden und nicht das Urteil eines Arztes über ein CTG. Solange
das so bleibt, ist dieses Modell ein Kandidat für eine Assistenzfunktion und kein fertiges
Diagnosewerkzeug.

## Die wichtigsten Zahlen

- Der Datensatz enthält 2.126 Aufzeichnungen, nach dem Entfernen von 13 Duplikaten noch 2.113.
- Es gibt 21 Merkmale, alle numerisch und ohne fehlende Werte.
- Die Klassen verteilen sich auf 77,9 Prozent Normal, 13,8 Prozent Suspect und 8,3 Prozent
  Pathological.
- Aufgeteilt wurde stratifiziert in 1.479 Trainingsfälle sowie je 317 Validierungs- und Testfälle.
- Verglichen wurden fünf Modelle, drei klassische und zwei neuronale Netze.
- Die naive Vergleichsbasis liegt bei 77,9 Prozent Accuracy und einem Recall von null auf
  Pathological.
- Das finale Modell ist ein Random Forest mit ausgeglichenen Klassengewichten.
- Auf dem Testset erreicht es 92,7 Prozent Accuracy und einen gewichteten F1-Score von 92,8
  Prozent.
- Von 26 pathologischen Testfällen erkennt es alle 26, das Konfidenzintervall reicht von 0,871 bis
  1,000.
- Von 44 verdächtigen Fällen erkennt es 33, also 75,0 Prozent.
- Die wichtigsten Merkmale sind der Anteil auffälliger Kurzzeit-Variabilität mit 14,1 Prozent und
  der Anteil auffälliger Langzeit-Variabilität mit 12,3 Prozent.
- Die Gittersuche umfasste 54 Kombinationen und 270 Trainingsläufe und brauchte 32,2 Sekunden



