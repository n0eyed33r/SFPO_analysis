# src/sfpo/models/measurement.py
from pathlib import Path
import numpy as np


class DataCleanUp:
    """
    Cleans SFPO measurement data through:
    1. Removing approach path (before displacement ≈ 0)
    2. Removing data after pull-out (after force ≈ 0, after maximum)
    3. Correcting force offset at zero displacement
    """

    def clean_measurement_pairs(self, file_path: str) -> list[tuple[float, float]]:
        """
        Cleans a single SFPO measurement file.
        
        Args:
            file_path: Path to the measurement file (as string)
            
        Returns:
            List of (displacement, force) tuples
        """
        
        # Load data
        try:
            df = np.loadtxt(
                fname=file_path, 
                delimiter="\t",
                skiprows=40,
                encoding="latin-1",
                usecols=(1, 2)
            )
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return []
        
        displacement = df[:, 0]
        force = df[:, 1]
        
        print(f"=== Original Data: {Path(file_path).name} ===")
        print(f"Data points: {len(displacement)}")
        print(f"Displacement range: {displacement.min():.4f} to {displacement.max():.4f} µm")
        print(f"Force range: {force.min():.6f} to {force.max():.6f} N")
        
        # STEP 1: Trim beginning - find displacement ≈ 0
        tolerance_disp = 1e-4  # 0.0001 µm tolerance
        indices_displacement_zero = np.where(np.abs(displacement) < tolerance_disp)[0]
        
        if len(indices_displacement_zero) > 0:
            first_zero_disp_index = indices_displacement_zero[0]
            print(f"\n=== Step 1: Trim Beginning ===")
            print(f"First index with displacement ≈ 0: {first_zero_disp_index}")
            
            displacement = displacement[first_zero_disp_index:]
            force = force[first_zero_disp_index:]
            
            print(f"Removed {first_zero_disp_index} points at the beginning")
        else:
            first_zero_disp_index = np.argmin(np.abs(displacement))
            displacement = displacement[first_zero_disp_index:]
            force = force[first_zero_disp_index:]
        
        # STEP 2: Trim end - find force ≈ 0 AFTER maximum
        max_force = np.max(force)
        max_force_index = np.argmax(force)
        
        print(f"\n=== Step 2: Trim End ===")
        print(f"Maximum force: {max_force:.4f} N at index {max_force_index}")
        
        tolerance_force = 1e-4  # 0.0001 N tolerance
        force_after_max = force[max_force_index:]
        indices_force_zero = np.where(np.abs(force_after_max) < tolerance_force)[0]
        
        if len(indices_force_zero) > 0:
            first_zero_force_index_local = indices_force_zero[0]
            first_zero_force_index = max_force_index + first_zero_force_index_local
            
            print(f"First index with force ≈ 0 (after maximum): {first_zero_force_index}")
            
            displacement = displacement[:first_zero_force_index + 1]
            force = force[:first_zero_force_index + 1]
        else:
            print("Warning: No force ≈ 0 found after maximum")
        
        # STEP 3: Force offset correction
        force_at_zero_displacement = force[0]
        
        print(f"\n=== Step 3: Force Offset Correction ===")
        print(f"Force at displacement ≈ 0: {force_at_zero_displacement:.6f} N")
        
        if force_at_zero_displacement > 0:
            print(f"Correcting positive offset")
            force = force - force_at_zero_displacement
        elif force_at_zero_displacement < 0:
            print(f"Correcting negative offset")
            force = force + abs(force_at_zero_displacement)
        
        print(f"\n=== Final Cleaned Data ===")
        print(f"Data points: {len(displacement)}")
        print(f"Force at start: {force[0]:.6f} N (should be ≈ 0)")
        
        return list(zip(displacement, force))