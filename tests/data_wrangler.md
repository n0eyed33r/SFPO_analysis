# Überlegung für DataWrangler

Pro Messung werden unterschiedlichste Parameter in einer .txt Datei festgehalten.

Die wichtigsten sind der Name (ID), daraus erfolgreich oder nicht-erfolgreicher SFPO und die Messwertpaare Weg sowie Kraft. Weitere Daten wären dann der Pfad (fürs programmieren wichtig)

## Struktur

MeasurementSeries (Messreihe)
├── name: "CF1_MMT1_160C_3wt"
├── path: "/pfad/zum/ordner"
├── measurements: [Measurement, Measurement, Measurement, ...]
│   │
│   ├── Measurement 1
│   │   ├── id: "01"
│   │   ├── fiber_diameter: 7.0
│   │   ├── displacement: [0.0, 0.1, 0.2, ...]
│   │   ├── force: [0.001, 0.002, 0.003, ...]
│   │   └── is_successful: True
│   │
│   ├── Measurement 2
│   │   └── ...

## Erste Datenbereinigung

Dies ist wichtig, um Messedaten im Ursprung (0,0) "loslaufen" zu lassen. Durch manuelle- (menschliche Handhabung) oder systematische Abweichungen:
    - beginnt die Messung manchmal nicht im Ursprung
    - wird die Messung nicht bei zu geringen Auszugskräften gestoppt und hinterfragt, wie tief die reale Einbettiefe ist/war (Kraftniveau - schwankt um 0 ... )