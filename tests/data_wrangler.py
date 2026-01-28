# tests/data_wrangler.py
from pathlib import Path
import numpy as np
import pandas as pd

def clean_measurement_pairs():
    """
    Bereinigt SFPO-Messdaten durch:
    1. Entfernen des Anfahrwegs (vor displacement ≈ 0)
    2. Entfernen der Daten nach dem Pull-Out (nach Kraft ≈ 0, nach Maximum)
    3. Korrektur des Kraft-Offsets am Nullpunkt
    
    Returns:
        tuple: (bereinigte_displacement, bereinigte_force)
    """
    
    # Daten laden
    df = np.loadtxt(
        fname="tests\\05a_PO_C3B2_CF1_F1_45wt_160C_1dc_14d_1000.txt", 
        delimiter="\t",
        skiprows=40,
        encoding="latin-1",
        usecols=(1, 2)
    )
    
    displacement = df[:, 0]  # first column
    force = df[:, 1]         # second column
    
    print(f"=== Original Data ===")
    print(f"Data points: {len(displacement)}")
    print(f"Displacement range: {displacement.min():.4f} to {displacement.max():.4f} µm")
    print(f"Force range: {force.min():.6f} to {force.max():.6f} N")
    
    # SCHRITT 1: Trimming am Anfang - finde displacement ≈ 0
    tolerance_disp = 1e-4  # 0.0001 µm Toleranz
    
    indices_displacement_zero = np.where(np.abs(displacement) < tolerance_disp)[0]
    
    if len(indices_displacement_zero) > 0:
        first_zero_disp_index = indices_displacement_zero[0]
        print(f"\n=== Step 1: Trim Beginning ===")
        print(f"First index with displacement ≈ 0: {first_zero_disp_index}")
        print(f"Actual value: {displacement[first_zero_disp_index]:.6f} µm")
        
        displacement = displacement[first_zero_disp_index:]
        force = force[first_zero_disp_index:]
        
        print(f"Removed {first_zero_disp_index} points at the beginning")
        print(f"Remaining data points: {len(displacement)}")
    else:
        print(f"\n=== Step 1: No exact zero found ===")
        first_zero_disp_index = np.argmin(np.abs(displacement))
        print(f"Using closest to zero at index {first_zero_disp_index}")
        print(f"Value: {displacement[first_zero_disp_index]:.6f} µm")
        
        displacement = displacement[first_zero_disp_index:]
        force = force[first_zero_disp_index:]
    
    # SCHRITT 2: Trimming am Ende - finde Kraft ≈ 0 NACH dem Maximum
    
    # Finde Maximum
    max_force = np.max(force)
    max_force_index = np.argmax(force)
    
    print(f"\n=== Step 2: Trim End ===")
    print(f"Maximum force: {max_force:.4f} N at index {max_force_index}")
    
    # Toleranz für Kraft (ähnlich klein wie für displacement)
    tolerance_force = 1e-4  # 0.0001 N Toleranz
    print(f"Force tolerance: {tolerance_force:.6f} N")
    
    # Nur im Bereich NACH dem Maximum suchen
    force_after_max = force[max_force_index:]
    
    # Finde alle Indizes wo Kraft ≈ 0 (im Bereich nach Maximum)
    indices_force_zero = np.where(np.abs(force_after_max) < tolerance_force)[0]
    
    if len(indices_force_zero) > 0:
        # Erster Index im Teil-Array (nach Maximum)
        first_zero_force_index_local = indices_force_zero[0]
        # Umrechnen auf ursprünglichen Index
        first_zero_force_index = max_force_index + first_zero_force_index_local
        
        print(f"First index with force ≈ 0 (after maximum): {first_zero_force_index}")
        print(f"Actual value: {force[first_zero_force_index]:.6f} N")
        
        # Schneide ab (bis einschließlich diesem Punkt)
        displacement = displacement[:first_zero_force_index + 1]
        force = force[:first_zero_force_index + 1]
        
        points_removed = len(df) - first_zero_disp_index - len(displacement)
        print(f"Removed {points_removed} points at the end")
        print(f"Remaining data points: {len(displacement)}")
    else:
        print("Warning: No force ≈ 0 found after maximum, keeping all data")
    
    # SCHRITT 3: Kraft-Offset-Korrektur
    force_at_zero_displacement = force[0]
    
    print(f"\n=== Step 3: Force Offset Correction ===")
    print(f"Force at displacement ≈ 0: {force_at_zero_displacement:.6f} N")
    
    if force_at_zero_displacement > 0:
        print(f"Correcting positive offset: subtracting {force_at_zero_displacement:.6f} N")
        force = force - force_at_zero_displacement
    elif force_at_zero_displacement < 0:
        print(f"Correcting negative offset: adding {abs(force_at_zero_displacement):.6f} N")
        force = force + abs(force_at_zero_displacement)
    else:
        print("No offset correction needed")
    
    # Final check
    print(f"\n=== Final Cleaned Data ===")
    print(f"Data points: {len(displacement)}")
    print(f"Displacement: {displacement.min():.2f} to {displacement.max():.2f} µm")
    print(f"Force: {force.min():.6f} to {force.max():.6f} N")
    print(f"Force at start (should be ≈ 0): {force[0]:.6f} N")
    
    return displacement, force


if __name__ == "__main__":
    cleaned_disp, cleaned_force = clean_measurement_pairs()
    
    # Optional: Zeige erste und letzte Werte
    print("\n=== First 5 Data Pairs ===")
    for i in range(min(5, len(cleaned_disp))):
        print(f"  {i+1}: disp={cleaned_disp[i]:.4f} µm, force={cleaned_force[i]:.6f} N")
    
    print("\n=== Last 5 Data Pairs ===")
    for i in range(max(0, len(cleaned_disp)-5), len(cleaned_disp)):
        print(f"  {i+1}: disp={cleaned_disp[i]:.4f} µm, force={cleaned_force[i]:.6f} N")