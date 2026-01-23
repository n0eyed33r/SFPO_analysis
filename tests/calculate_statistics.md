# Überlegung für CalculateStatistics

Welche Werte sind wichtig herauszusuchen oder zu berechnen

## Gegebene Parameter

F_max:
     der höchste Wert der erreicht werden kann. Meist findet dieser beim initialen Risswachstum statt. Es kann passieren, dass eine Kraft beim Herausziehen, nach der initialen Rissentwicklung, größer ist. Dies wäre dann eine sehr hohe Reib-Kraft und entspricht nicht der "Ablösekraft".
Einbettlänge:
    ist der gesamte Auszugsweg, welcher zwar computergestützt vorgegeben ist (meist 1000 µm), aber real nicht eingehalten werden kann und somit abweicht.
    Deshalb ist es auch wichtig in DataWrangler sinnlose Einbettlängen rauszurechnen.

## Berechnete Werte

### Absolutwerte

IFSS (Grenzflächenscherfestigkeit, en. interfacial shear strength)
    ist die Normierung der eingebetteten Mantelfläche über F_max

Pull-Out Rate
    Gibt prozentual die erfolgreichen SFPO einer Messreihe im Vergleich mit der Gesamtanzahl an Messungen einer Messreihe wieder. 

#### Arbeit

Gesamte Auszugsarbeit
    gibt die benötigte Arbeit wieder des gesamten Auszugs.
Ablösearbeit
    gibt die benötigte Arbeit der initialen Rissentwicklnug wieder.
Reibarbeit
    gibt die Arbeit beim Auszug nach der initialen Rissentwicklung wieder.


### Statistik

Es sollen Mittelwerte und Standardabweichung sowie Boxplots:
     von IFSS, der Ablösearbeit, der gesamten verrichteten Auszugsarbeit.

Es soll ein Boots-Trapping angewandt werden, um Statistiken mit und ohne Boots-Trapping zu vergleichen
    vor allem Messreihen sehr geringer Pull-Out Rate

Durch Boots-Trapping soll dann eine ANOVA Analyse durchgeführt werden und der Einfluss der Beschichtungen verglichen werden.

Kaplan-Meier-Analyse
    soll genutzt werden um sehr kleine Pull-Out Raten qualitativ beschreiben zu können. (bei welcher Kraft welche Probe gebrochen ist)