# src/sfpo/io/exporter.py
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime


class DataExporter:
    """
    Exports SFPO analysis results to various formats.
    """
    
    @staticmethod
    def save_cleaned_data_txt(
        measurement_data: List[Tuple[float, float]],
        original_file_path: str,
        output_folder: Path
    ) -> Path:
        """
        Saves cleaned measurement data to a TXT file.
        
        Format: Same as original (tab-separated), but with cleaned data.
        Filename: Original_name_Clean.txt
        
        Args:
            measurement_data: List of (displacement, force) tuples
            original_file_path: Path to the original file
            output_folder: Folder to save the cleaned file
            
        Returns:
            Path to the saved file
        """
        # Create output folder if it doesn't exist
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # Generate output filename
        original_name = Path(original_file_path).stem  # without extension
        output_filename = f"{original_name}_Clean.txt"
        output_path = output_folder / output_filename
        
        # Write data
        with open(output_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("# Cleaned SFPO Measurement Data\n")
            f.write(f"# Original file: {Path(original_file_path).name}\n")
            f.write(f"# Cleaned on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("#\n")
            f.write("# Displacement [Âµm]\tForce [N]\n")
            f.write("#" + "="*50 + "\n")
            
            # Data
            for displacement, force in measurement_data:
                f.write(f"{displacement:.6f}\t{force:.6f}\n")
        
        return output_path
    
    @staticmethod
    def save_single_series_to_excel(
        results: dict,
        series_name: str,
        output_folder: Path
    ) -> Path:
        """
        Saves results from a single measurement series to Excel.
        
        Creates an Excel file with multiple sheets:
        - Summary: Statistical overview
        - Individual_Measurements: All measurements with calculated values
        - Raw_Data_X: One sheet per measurement with cleaned data
        
        Args:
            results: Dictionary with analysis results
            series_name: Name of the measurement series
            output_folder: Folder to save the Excel file
            
        Returns:
            Path to the saved Excel file
        """
        # Create output folder if it doesn't exist
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{series_name}_Analysis_{timestamp}.xlsx"
        output_path = output_folder / filename
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Sheet 1: Summary Statistics
            summary_data = {
                'Parameter': [
                    'Fiber Diameter [Âµm]',
                    'F_max [N]',
                    'Embedding Length [Âµm]',
                    'IFSS [MPa]',
                    'Work Total [ÂµJ]',
                    'Work Debonding [ÂµJ]',
                    'Work Friction [ÂµJ]'
                ],
                'Mean': [
                    f"{pd.Series(results['fiber_diameters']).mean():.4f}",
                    f"{pd.Series(results['f_max_values']).mean():.4f}",
                    f"{pd.Series(results['embedding_lengths']).mean():.4f}",
                    f"{pd.Series(results['ifss_values']).mean():.4f}",
                    f"{pd.Series(results['work_total']).mean():.4f}",
                    f"{pd.Series(results['work_debonding']).mean():.4f}",
                    f"{pd.Series(results['work_friction']).mean():.4f}"
                ],
                'Std Dev': [
                    f"{pd.Series(results['fiber_diameters']).std(ddof=1):.4f}",
                    f"{pd.Series(results['f_max_values']).std(ddof=1):.4f}",
                    f"{pd.Series(results['embedding_lengths']).std(ddof=1):.4f}",
                    f"{pd.Series(results['ifss_values']).std(ddof=1):.4f}",
                    f"{pd.Series(results['work_total']).std(ddof=1):.4f}",
                    f"{pd.Series(results['work_debonding']).std(ddof=1):.4f}",
                    f"{pd.Series(results['work_friction']).std(ddof=1):.4f}"
                ],
                'n': [len(results['fiber_diameters'])] * 7
            }
            
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 2: Individual Measurements
            individual_data = {
                'Measurement': [Path(m['file_path']).name for m in results['measurements']],
                'Fiber Diameter [Âµm]': results['fiber_diameters'],
                'F_max [N]': results['f_max_values'],
                'Embedding Length [Âµm]': results['embedding_lengths'],
                'IFSS [MPa]': results['ifss_values'],
                'Work Total [ÂµJ]': results['work_total'],
                'Work Debonding [ÂµJ]': results['work_debonding'],
                'Work Friction [ÂµJ]': results['work_friction']
            }
            
            df_individual = pd.DataFrame(individual_data)
            df_individual.to_excel(writer, sheet_name='Individual_Measurements', index=False)
            
            # Sheets 3+: Raw cleaned data for each measurement
            for i, measurement in enumerate(results['measurements'], 1):
                sheet_name = f"Raw_Data_{i}"
                if len(sheet_name) > 31:  # Excel sheet name limit
                    sheet_name = f"Data_{i}"
                
                # Create DataFrame from measurement data
                df_raw = pd.DataFrame(
                    measurement['data'],
                    columns=['Displacement [Âµm]', 'Force [N]']
                )
                
                df_raw.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return output_path
    
    @staticmethod
    def save_multiple_series_to_excel(
        all_results: Dict[str, dict],
        output_folder: Path
    ) -> Path:
        """
        Saves results from multiple measurement series to a single Excel file.
        
        Creates an Excel file with:
        - Comparison: Summary statistics for all series
        - SeriesName_Summary: Individual summary for each series
        - SeriesName_Data: Individual measurements for each series
        
        Args:
            all_results: Dictionary with {series_name: results_dict}
            output_folder: Folder to save the Excel file
            
        Returns:
            Path to the saved Excel file
        """
        # Create output folder if it doesn't exist
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"SFPO_MultiSeries_Analysis_{timestamp}.xlsx"
        output_path = output_folder / filename
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Sheet 1: Comparison of all series
            comparison_data = {
                'Series': [],
                'n': [],
                'Fiber Ã˜ [Âµm]': [],
                'Fiber Ã˜ Std': [],
                'F_max [N]': [],
                'F_max Std': [],
                'Emb. Length [Âµm]': [],
                'Emb. Length Std': [],
                'IFSS [MPa]': [],
                'IFSS Std': [],
                'Work Total [ÂµJ]': [],
                'Work Total Std': []
            }
            
            for series_name, results in all_results.items():
                comparison_data['Series'].append(series_name)
                comparison_data['n'].append(len(results['f_max_values']))
                
                # Fiber diameter
                comparison_data['Fiber Ã˜ [Âµm]'].append(
                    f"{pd.Series(results['fiber_diameters']).mean():.2f}"
                )
                comparison_data['Fiber Ã˜ Std'].append(
                    f"{pd.Series(results['fiber_diameters']).std(ddof=1):.2f}"
                )
                
                # F_max
                comparison_data['F_max [N]'].append(
                    f"{pd.Series(results['f_max_values']).mean():.4f}"
                )
                comparison_data['F_max Std'].append(
                    f"{pd.Series(results['f_max_values']).std(ddof=1):.4f}"
                )
                
                # Embedding length
                comparison_data['Emb. Length [Âµm]'].append(
                    f"{pd.Series(results['embedding_lengths']).mean():.2f}"
                )
                comparison_data['Emb. Length Std'].append(
                    f"{pd.Series(results['embedding_lengths']).std(ddof=1):.2f}"
                )
                
                # IFSS
                comparison_data['IFSS [MPa]'].append(
                    f"{pd.Series(results['ifss_values']).mean():.2f}"
                )
                comparison_data['IFSS Std'].append(
                    f"{pd.Series(results['ifss_values']).std(ddof=1):.2f}"
                )
                
                # Work Total
                comparison_data['Work Total [ÂµJ]'].append(
                    f"{pd.Series(results['work_total']).mean():.4f}"
                )
                comparison_data['Work Total Std'].append(
                    f"{pd.Series(results['work_total']).std(ddof=1):.4f}"
                )
            
            df_comparison = pd.DataFrame(comparison_data)
            df_comparison.to_excel(writer, sheet_name='Comparison', index=False)
            
            # Individual sheets for each series
            for series_name, results in all_results.items():
                # Sanitize sheet name (max 31 chars, no special chars)
                sheet_name_base = series_name[:25]  # Leave room for suffix
                
                # Summary sheet
                summary_sheet = f"{sheet_name_base}_Sum"
                summary_data = {
                    'Parameter': [
                        'Fiber Diameter [Âµm]',
                        'F_max [N]',
                        'Embedding Length [Âµm]',
                        'IFSS [MPa]',
                        'Work Total [ÂµJ]',
                        'Work Debonding [ÂµJ]',
                        'Work Friction [ÂµJ]'
                    ],
                    'Mean': [
                        pd.Series(results['fiber_diameters']).mean(),
                        pd.Series(results['f_max_values']).mean(),
                        pd.Series(results['embedding_lengths']).mean(),
                        pd.Series(results['ifss_values']).mean(),
                        pd.Series(results['work_total']).mean(),
                        pd.Series(results['work_debonding']).mean(),
                        pd.Series(results['work_friction']).mean()
                    ],
                    'Std Dev': [
                        pd.Series(results['fiber_diameters']).std(ddof=1),
                        pd.Series(results['f_max_values']).std(ddof=1),
                        pd.Series(results['embedding_lengths']).std(ddof=1),
                        pd.Series(results['ifss_values']).std(ddof=1),
                        pd.Series(results['work_total']).std(ddof=1),
                        pd.Series(results['work_debonding']).std(ddof=1),
                        pd.Series(results['work_friction']).std(ddof=1)
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name=summary_sheet, index=False)
                
                # Data sheet
                data_sheet = f"{sheet_name_base}_Data"
                data_dict = {
                    'Measurement': [Path(m['file_path']).name for m in results['measurements']],
                    'Fiber Ã˜ [Âµm]': results['fiber_diameters'],
                    'F_max [N]': results['f_max_values'],
                    'Emb. Len [Âµm]': results['embedding_lengths'],
                    'IFSS [MPa]': results['ifss_values'],
                    'Work Tot [ÂµJ]': results['work_total'],
                    'Work Deb [ÂµJ]': results['work_debonding'],
                    'Work Fric [ÂµJ]': results['work_friction']
                }
                df_data = pd.DataFrame(data_dict)
                df_data.to_excel(writer, sheet_name=data_sheet, index=False)
        
        return output_path