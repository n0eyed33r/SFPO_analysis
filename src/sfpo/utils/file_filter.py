# src/sfpo/utils/file_filter.py
import re
from pathlib import Path
from typing import Tuple, List


class MeasurementFileFilter:
    """
    Filters and categorizes measurement files based on naming convention.
    
    Naming convention:
    - ZZa_...txt â†’ successful SFPO measurements
    - ZZxa_...txt â†’ SFPO measurements with fiber break
    
    Where ZZ are digits (e.g., 01, 02, 03, etc.)
    """
    
    @staticmethod
    def categorize_files(file_paths: List[str]) -> Tuple[List[str], List[str]]:
        """
        Categorizes measurement files into successful and broken.
        
        Args:
            file_paths: List of file paths as strings
            
        Returns:
            Tuple of (successful_measurements, broken_measurements)
        """
        successful = []
        broken = []
        
        for file_path in file_paths:
            filename = Path(file_path).name
            
            if MeasurementFileFilter.is_successful(filename):
                successful.append(file_path)
            elif MeasurementFileFilter.is_broken(filename):
                broken.append(file_path)
            else:
                print(f"Warning: File doesn't match naming convention: {filename}")
        
        return successful, broken
    
    @staticmethod
    def is_successful(filename: str) -> bool:
        """
        Check if filename matches successful SFPO pattern: ZZa_...txt
        
        Pattern: starts with 2 digits, followed by 'a', then '_'
        Example: 01a_test.txt, 15a_measurement.txt
        """
        pattern = r'^\d{2}a_.*\.txt$'
        return bool(re.match(pattern, filename))
    
    @staticmethod
    def is_broken(filename: str) -> bool:
        """
        Check if filename matches broken fiber pattern: ZZxa_...txt
        
        Pattern: starts with 2 digits, followed by 'xa', then '_'
        Example: 01xa_test.txt, 15xa_measurement.txt
        """
        pattern = r'^\d{2}xa_.*\.txt$'
        return bool(re.match(pattern, filename))
    
    @staticmethod
    def extract_number(filename: str) -> int:
        """
        Extracts the measurement number from the filename.
        
        Example: "01a_test.txt" â†’ 1, "15xa_test.txt" â†’ 15
        """
        match = re.match(r'^(\d{2})', filename)
        if match:
            return int(match.group(1))
        return 0