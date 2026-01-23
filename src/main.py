# src/main.py
from sfpo.io.loader import FileLoader

def main():
    """
    main of the single fiber pull-out analyzer
    """
    
    # step 1: load data
    files = load_files()
    if not files:
        return
    
    # Schritt 2: Daten verarbeiten
    measurements = parse_measurements(files)
    
    # Schritt 3: Berechnungen durchführen
    results = analyze(measurements)
    
    # Schritt 4: Ergebnisse ausgeben
    export_results(results)


def load_files():
    """Öffnet den Loader-Dialog und gibt die ausgewählten Dateien zurück"""
    loader = FileLoader()
    return loader.run()


def parse_measurements(files):
    """Liest die Messdaten aus den Dateien"""
    # ... hier kommt später der Parser
    pass


def analyze(measurements):
    """Führt alle Berechnungen durch"""
    # ... hier kommen später die Berechnungen
    pass


def export_results(results):
    """Exportiert die Ergebnisse"""
    # ... hier kommt später der Export
    pass


if __name__ == "__main__":
    main()