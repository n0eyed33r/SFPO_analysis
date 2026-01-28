# src/sfpo/analysis/mechanics.py
import numpy as np
from typing import List, Tuple
from pathlib import Path


class MechanicsCalculator:
    """
    Performs mechanical calculations on SFPO measurement data.
    """
    
    @staticmethod
    def read_fiber_diameter(file_path: str) -> float:
        """
        Reads the fiber diameter from line 20 of the measurement file.
        
        Format in line 20: "Faserdurchm. [Âµm]:\t7.2"
        
        Args:
            file_path: Path to the measurement file
            
        Returns:
            Fiber diameter [Âµm]
        """
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()
                
                if len(lines) >= 20:
                    line_20 = lines[19]  # Index 19 = Line 20
                    # Split by tab and take the second part
                    parts = line_20.split('\t')
                    if len(parts) >= 2:
                        diameter = float(parts[1].strip())
                        return diameter
        except Exception as e:
            print(f"Error reading fiber diameter from {file_path}: {e}")
            return 0.0
        
        return 0.0
    
    @staticmethod
    def calculate_f_max(measurement: List[Tuple[float, float]]) -> float:
        """
        Determines the maximum force in a measurement.
        
        Args:
            measurement: List of (displacement, force) tuples
            
        Returns:
            Maximum force value [N]
        """
        if not measurement:
            return 0.0
        
        forces = [point[1] for point in measurement]
        return max(forces)
    
    @staticmethod
    def get_f_max_index(measurement: List[Tuple[float, float]]) -> int:
        """
        Finds the index where F_max occurs.
        
        Args:
            measurement: List of (displacement, force) tuples
            
        Returns:
            Index of maximum force
        """
        if not measurement:
            return 0
        
        forces = [point[1] for point in measurement]
        return forces.index(max(forces))
    
    @staticmethod
    def calculate_embedding_length(measurement: List[Tuple[float, float]]) -> float:
        """
        Determines the maximum embedding length (after data cleaning).
        This is the last displacement value in the cleaned data.
        
        Args:
            measurement: List of (displacement, force) tuples
            
        Returns:
            Maximum embedding length [Âµm]
        """
        if not measurement:
            return 0.0
        
        # Last displacement value after cleaning
        return measurement[-1][0]
    
    @staticmethod
    def calculate_ifss(f_max: float, diameter: float, embedding_length: float) -> float:
        """
        Calculates the Interface Shear Strength (IFSS).
        
        Formula: IFSS = F_max / (Ï€ Ã— d Ã— l_e)
        
        Args:
            f_max: Maximum force [N]
            diameter: Fiber diameter [Âµm]
            embedding_length: Embedding length [Âµm]
            
        Returns:
            IFSS [MPa]
        """
        if diameter <= 0 or embedding_length <= 0:
            return 0.0
        
        # Calculate surface area: Ï€ Ã— d Ã— l_e [ÂµmÂ²]
        surface_area = np.pi * diameter * embedding_length
        
        # IFSS = F_max / surface_area
        # F_max is in [N], surface_area in [ÂµmÂ²]
        # Result: N/ÂµmÂ² = MPa (because 1 N/ÂµmÂ² = 1 MPa)
        ifss = f_max / surface_area * 1e6  # Convert to MPa
        
        return ifss
    
    @staticmethod
    def calculate_work_total(measurement: List[Tuple[float, float]]) -> float:
        """
        Calculates the total work (complete area under the curve).
        
        Uses trapezoidal integration.
        
        Args:
            measurement: List of (displacement, force) tuples
            
        Returns:
            Total work [ÂµJ] (micro-Joules)
        """
        if len(measurement) < 2:
            return 0.0
        
        displacements = np.array([point[0] for point in measurement])
        forces = np.array([point[1] for point in measurement])
        
        # Trapezoidal integration
        work = np.trapezoid(forces, displacements)
        
        return float(work)
    
    @staticmethod
    def calculate_work_debonding(measurement: List[Tuple[float, float]]) -> float:
        """
        Calculates the debonding work (area under curve up to F_max).
        
        This is the energy required to debond the fiber from the matrix.
        
        Args:
            measurement: List of (displacement, force) tuples
            
        Returns:
            Debonding work [ÂµJ]
        """
        if len(measurement) < 2:
            return 0.0
        
        # Find index of F_max
        f_max_index = MechanicsCalculator.get_f_max_index(measurement)
        
        # Take only data up to F_max (inclusive)
        measurement_debonding = measurement[:f_max_index + 1]
        
        displacements = np.array([point[0] for point in measurement_debonding])
        forces = np.array([point[1] for point in measurement_debonding])
        
        # Trapezoidal integration
        work = np.trapezoid(forces, displacements)
        
        return float(work)
    
    @staticmethod
    def calculate_work_friction(measurement: List[Tuple[float, float]]) -> float:
        """
        Calculates the friction work (area under curve after F_max).
        
        This is the energy dissipated through friction during pull-out.
        
        Args:
            measurement: List of (displacement, force) tuples
            
        Returns:
            Friction work [ÂµJ]
        """
        if len(measurement) < 2:
            return 0.0
        
        # Find index of F_max
        f_max_index = MechanicsCalculator.get_f_max_index(measurement)
        
        # Take only data after F_max
        if f_max_index >= len(measurement) - 1:
            # F_max is at the end, no friction work
            return 0.0
        
        measurement_friction = measurement[f_max_index:]
        
        displacements = np.array([point[0] for point in measurement_friction])
        forces = np.array([point[1] for point in measurement_friction])
        
        # Trapezoidal integration
        work = np.trapezoid(forces, displacements)
        
        return float(work)
    
    @staticmethod
    def calculate_mean_std(values: List[float]) -> Tuple[float, float]:
        """
        Calculates mean and standard deviation.
        
        Args:
            values: List of numerical values
            
        Returns:
            Tuple of (mean, std_deviation)
        """
        if not values:
            return 0.0, 0.0
        
        arr = np.array(values)
        return float(np.mean(arr)), float(np.std(arr, ddof=1))