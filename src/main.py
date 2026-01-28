# src/main.py
from typing import List, Dict, Optional
from pathlib import Path
from tkinter import filedialog
import tkinter as tk

from sfpo.io.loader import FileLoader
from sfpo.io.exporter import DataExporter
from sfpo.models.measruement import DataCleanUp
from sfpo.utils.file_filter import MeasurementFileFilter
from sfpo.analysis.mechanics import MechanicsCalculator
from sfpo.analysis.statistics import StatisticsVisualizer
from sfpo.analysis.anova import IFSSAnalyzer


def main():
    """
    Main function of the Single Fiber Pull-Out Analyzer
    """
    
    # Step 1: Load data
    loader_result = load_files()
    if not loader_result:
        print("No files selected. Exiting...")
        return

    # Step 2: Process based on mode
    # Mode 3 now returns (dict, anova_enabled) tuple
    if isinstance(loader_result, tuple):
        # Mode 3: Multiple series with potential ANOVA
        files, anova_enabled = loader_result
        
        if not files:
            print("No files found. Exiting...")
            return
        
        print(f"\n{'='*60}")
        print(f"Found {len(files)} measurement series")
        if anova_enabled:
            print("ANOVA + Bootstrap analysis: ENABLED")
        print(f"{'='*60}")
        
        # Ask once for plots save location
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        
        plots_save_folder_str = filedialog.askdirectory(
            title="Select folder to save ALL plots (for all series)"
        )
        root.destroy()
        
        plots_save_folder: Optional[Path] = None
        if not plots_save_folder_str:
            print("No save location selected for plots, skipping plot generation")
            plots_save_folder = None
        else:
            plots_save_folder = Path(plots_save_folder_str)
            print(f"All plots will be saved to: {plots_save_folder}")
        
        # Process all series and collect results
        all_results = {}
        for series_name, file_list in files.items():
            print(f"\n{'='*60}")
            print(f"Processing series: {series_name}")
            print(f"{'='*60}")
            
            results = process_single_series(
                file_list, 
                series_name=series_name,
                plots_save_folder=plots_save_folder
            )
            if results:
                all_results[series_name] = results
        
        # Create comparison analysis and export
        if all_results:
            process_multiple_series(all_results, anova_enabled=anova_enabled)
            
    elif isinstance(loader_result, list):
        # Mode 2: Single measurement series
        results = process_single_series(loader_result)
        
        if results:
            export_single_series(results, series_name="SFPO_analysis")
            
    elif isinstance(loader_result, str):
        # Mode 1: Single file
        print("\n=== Single File Mode ===")
        results = process_single_series([loader_result], series_name="single_measurement")
        
        if results:
            export_single_series(results, series_name="single_measurement")
    else:
        print("Unknown file format")
        return


def load_files():
    """Opens the loader dialog and returns the selected files"""
    loader = FileLoader()
    return loader.run()


def analyze_broken_fibers(file_paths: List[str], series_name: str) -> Dict[str, List]:
    """
    Analyzes broken fibers (ZZxa_ files) for Kaplan-Meier survival analysis.
    
    Args:
        file_paths: List of all file paths in the series
        series_name: Name of the series
    
    Returns:
        Dictionary with broken_forces and broken_labels
    """
    broken_forces = []
    broken_labels = []
    
    for file_path in file_paths:
        filename = Path(file_path).stem
        
        # Check if this is a broken fiber (pattern: contains 'xa_' or ends with 'xa')
        if 'xa_' in filename.lower() or filename.lower().endswith('xa'):
            try:
                cleaner = DataCleanUp()
                cleaned_data = cleaner.clean_measurement_pairs(file_path)
                
                if not cleaned_data or len(cleaned_data) == 0:
                    continue
                
                forces = [point[1] for point in cleaned_data]
                if forces:
                    f_max = max(forces)
                    broken_forces.append(f_max)
                    broken_labels.append(filename)
                    
            except Exception as e:
                print(f"  Warning: Could not process broken fiber {filename}: {str(e)}")
                continue
    
    if broken_forces:
        print(f"\n  Found {len(broken_forces)} broken fibers for Kaplan-Meier analysis")
    
    return {
        'broken_forces': broken_forces,
        'broken_labels': broken_labels
    }


def export_single_series(results: dict, series_name: str = "measurement_series"):
    """
    Exports results from a single measurement series.
    
    Args:
        results: Dictionary with analysis results
        series_name: Name of the measurement series
    """
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes('-topmost', True)
    
    save_folder = filedialog.askdirectory(
        title="Select folder to save export files"
    )
    
    if not save_folder:
        print("No save location selected, skipping export")
        return
    
    save_path = Path(save_folder)
    
    print(f"\n{'='*60}")
    print("=== Exporting Results ===")
    print(f"Export folder: {save_path}")
    print("="*60)
    
    exporter = DataExporter()
    
    # Create subfolders
    cleaned_folder = save_path / f"{series_name}_cleaned_data"
    excel_folder = save_path / f"{series_name}_excel"
    
    # Export cleaned TXT files
    print("\nExporting cleaned measurement data...")
    for measurement in results['measurements']:
        output_path = exporter.save_cleaned_data_txt(
            measurement_data=measurement['data'],
            original_file_path=measurement['file_path'],
            output_folder=cleaned_folder
        )
        print(f"  Saved: {output_path.name}")
    
    # Export to Excel
    print("\nExporting to Excel...")
    excel_path = exporter.save_single_series_to_excel(
        results=results,
        series_name=series_name,
        output_folder=excel_folder
    )
    print(f"  Saved: {excel_path.name}")
    
    print(f"\n✓ All exports saved to: {save_path}")
    print("="*60)


def save_visualizations(
    results: dict, 
    series_name: str = "measurement_series",
    predefined_save_folder: Optional[Path] = None
):
    """
    Creates and saves all visualizations (boxplots).
    
    Args:
        results: Dictionary with all calculated values
        series_name: Name of the measurement series
        predefined_save_folder: If provided, use this folder instead of asking user
    """
    if predefined_save_folder is None:
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        
        save_folder = filedialog.askdirectory(
            title="Select folder to save plots"
        )
        
        if not save_folder:
            print("No save location selected, skipping visualization export")
            return
        
        save_path = Path(save_folder)
    else:
        save_path = predefined_save_folder
        print(f"Using predefined save folder: {save_path}")
    
    plots_folder = save_path / f"{series_name}_plots"
    plots_folder.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*60}")
    print("=== Creating Visualizations ===")
    print(f"Saving plots to: {plots_folder}")
    print("="*60)
    
    visualizer = StatisticsVisualizer()
    
    # Create complete overview (all 6 plots in one figure)
    print("\nCreating complete overview...")
    visualizer.create_all_boxplots_grid(
        f_max=results['f_max_values'],
        embedding_length=results['embedding_lengths'],
        ifss=results['ifss_values'],
        work_total=results['work_total'],
        work_debonding=results['work_debonding'],
        work_friction=results['work_friction'],
        output_path=plots_folder / "complete_overview.png"
    )
    
    # Create individual boxplots
    print("\nCreating individual boxplots...")
    
    visualizer.create_boxplot(
        data=results['f_max_values'],
        title='Maximum Force (F_max)',
        ylabel='Force [N]',
        output_path=plots_folder / "boxplot_fmax.png"
    )
    
    visualizer.create_boxplot(
        data=results['embedding_lengths'],
        title='Embedding Length',
        ylabel='Length [µm]',
        output_path=plots_folder / "boxplot_embedding_length.png"
    )
    
    visualizer.create_boxplot(
        data=results['ifss_values'],
        title='Interface Shear Strength (IFSS)',
        ylabel='IFSS [MPa]',
        output_path=plots_folder / "boxplot_ifss.png"
    )
    
    visualizer.create_boxplot(
        data=results['work_total'],
        title='Total Work',
        ylabel='Work [µJ]',
        output_path=plots_folder / "boxplot_work_total.png"
    )
    
    visualizer.create_boxplot(
        data=results['work_debonding'],
        title='Debonding Work',
        ylabel='Work [µJ]',
        output_path=plots_folder / "boxplot_work_debonding.png"
    )
    
    visualizer.create_boxplot(
        data=results['work_friction'],
        title='Friction Work',
        ylabel='Work [µJ]',
        output_path=plots_folder / "boxplot_work_friction.png"
    )
    
    # Create Force-Displacement plot
    if 'measurements_data' in results and results['measurements_data']:
        print("\nCreating Force-Displacement diagram...")
        visualizer.create_force_displacement_plot(
            measurements_data=results['measurements_data'],
            title=f"Force-Displacement Curves: {series_name}",
            output_path=plots_folder / "force_displacement_curves.png"
        )
    
    # Create Kaplan-Meier plot for broken fibers
    if 'broken_forces' in results and results['broken_forces']:
        print(f"\nCreating Kaplan-Meier survival plot ({len(results['broken_forces'])} broken fibers)...")
        visualizer.create_kaplan_meier_plot(
            broken_forces=results['broken_forces'],
            broken_labels=results['broken_labels'],
            output_path=plots_folder / "kaplan_meier_survival.png"
        )
    
    print(f"\n✓ All visualizations saved to: {plots_folder}")
    print("="*60)


def save_comparison_visualizations(all_results: Dict[str, dict], output_folder: Path):
    """
    Creates comparison visualizations for multiple measurement series.
    
    Args:
        all_results: Dictionary with {series_name: results_dict}
        output_folder: Folder to save the comparison plots
    """
    print(f"\n{'='*60}")
    print("=== Creating Comparison Visualizations ===")
    print(f"Saving plots to: {output_folder}")
    print("="*60)
    
    visualizer = StatisticsVisualizer()
    
    # Prepare data dictionaries for combined plots
    f_max_dict = {}
    embedding_dict = {}
    ifss_dict = {}
    work_total_dict = {}
    work_debonding_dict = {}
    work_friction_dict = {}
    
    for series_name, results in all_results.items():
        f_max_dict[series_name] = results['f_max_values']
        embedding_dict[series_name] = results['embedding_lengths']
        ifss_dict[series_name] = results['ifss_values']
        work_total_dict[series_name] = results['work_total']
        work_debonding_dict[series_name] = results['work_debonding']
        work_friction_dict[series_name] = results['work_friction']
    
    # Create combined boxplots
    print("\nCreating comparison boxplots...")
    
    visualizer.create_combined_boxplots(
        data_dict=f_max_dict,
        title='Comparison: Maximum Force (F_max)',
        ylabel='Force [N]',
        output_path=output_folder / "comparison_fmax.png"
    )
    
    visualizer.create_combined_boxplots(
        data_dict=embedding_dict,
        title='Comparison: Embedding Length',
        ylabel='Length [µm]',
        output_path=output_folder / "comparison_embedding_length.png"
    )
    
    visualizer.create_combined_boxplots(
        data_dict=ifss_dict,
        title='Comparison: Interface Shear Strength (IFSS)',
        ylabel='IFSS [MPa]',
        output_path=output_folder / "comparison_ifss.png"
    )
    
    visualizer.create_combined_boxplots(
        data_dict=work_total_dict,
        title='Comparison: Total Work',
        ylabel='Work [µJ]',
        output_path=output_folder / "comparison_work_total.png"
    )
    
    visualizer.create_combined_boxplots(
        data_dict=work_debonding_dict,
        title='Comparison: Debonding Work',
        ylabel='Work [µJ]',
        output_path=output_folder / "comparison_work_debonding.png"
    )
    
    visualizer.create_combined_boxplots(
        data_dict=work_friction_dict,
        title='Comparison: Friction Work',
        ylabel='Work [µJ]',
        output_path=output_folder / "comparison_work_friction.png"
    )
    
    print(f"\n✓ All comparison plots saved!")
    print("="*60)


def perform_anova_analysis(
    all_results: Dict[str, dict], 
    output_folder: Path
) -> Optional[Dict]:
    """
    Performs ANOVA analysis with Bootstrap for IFSS comparison.
    
    Creates a separate folder for ANOVA results with:
    - Bootstrap distribution plots
    - Confidence interval comparison plot
    - ANOVA results table
    - Complete overview figure
    - Text file with detailed results
    
    Args:
        all_results: Dictionary with {series_name: results_dict}
        output_folder: Base folder for saving results
        
    Returns:
        Dictionary with ANOVA analysis results, or None if failed
    """
    import matplotlib.pyplot as plt
    
    print(f"\n{'='*60}")
    print("=== ANOVA + Bootstrap Analysis for IFSS ===")
    print("="*60)
    
    # Create ANOVA output folder
    anova_folder = output_folder / "anova_bootstrap_analysis"
    try:
        anova_folder.mkdir(exist_ok=True, parents=True)
        print(f"ANOVA results folder created: {anova_folder}")
    except Exception as e:
        print(f"ERROR: Could not create ANOVA folder: {e}")
        return None
    
    # Prepare IFSS data dictionary
    ifss_data = {}
    success_rates = {}
    excluded_groups = []
    
    for series_name, results in all_results.items():
        n_successful = len(results['ifss_values'])
        n_broken = len(results.get('broken_forces', []))
        total = n_successful + n_broken
        
        # Check minimum sample size requirement
        if n_successful < 2:
            excluded_groups.append((series_name, n_successful))
            print(f"  {series_name}: n={n_successful} - EXCLUDED (min. 2 required)")
            continue
        
        ifss_data[series_name] = results['ifss_values']
        
        if total > 0:
            success_rates[series_name] = n_successful / total
        else:
            success_rates[series_name] = 1.0
        
        print(f"  {series_name}: n={n_successful}, success rate={success_rates[series_name]:.1%}")
    
    # Report excluded groups
    if excluded_groups:
        print(f"\n  ⚠ {len(excluded_groups)} groups excluded due to insufficient samples:")
        for name, n in excluded_groups:
            print(f"    - {name} (n={n})")
    
    # Check minimum requirements
    if len(ifss_data) < 2:
        print(f"\nWarning: ANOVA requires at least 2 valid groups. Only {len(ifss_data)} remaining. Skipping analysis.")
        return None
    
    # Perform analysis
    print(f"\nPerforming ANOVA + Bootstrap analysis on {len(ifss_data)} groups...")
    analyzer = IFSSAnalyzer(alpha=0.05, confidence_level=0.95)
    
    try:
        results = analyzer.analyze(ifss_data, success_rates)
        print("  Analysis completed successfully")
    except Exception as e:
        print(f"ERROR during ANOVA analysis: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Print results to console
    print(analyzer.format_results_text(results))
    
    # Save text results
    print("\nSaving analysis results...")
    try:
        analyzer.save_results_to_file(
            results, 
            anova_folder / "anova_results.txt"
        )
        print(f"  ✓ Text file saved")
    except Exception as e:
        print(f"  ERROR saving text file: {e}")
    
    # Create visualizations
    visualizer = StatisticsVisualizer()
    
    # 1. Bootstrap distribution plots
    print("Creating bootstrap distribution plots...")
    try:
        plt.close('all')  # Close any existing figures
        visualizer.create_bootstrap_distribution_plot(
            bootstrap_results=results['bootstrap'],
            output_path=anova_folder / "bootstrap_distributions.png"
        )
        print(f"  ✓ Bootstrap distributions saved")
    except Exception as e:
        print(f"  ERROR creating bootstrap plot: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Confidence interval comparison
    print("Creating confidence interval plot...")
    try:
        plt.close('all')
        visualizer.create_ifss_confidence_interval_plot(
            bootstrap_results=results['bootstrap'],
            output_path=anova_folder / "ifss_confidence_intervals.png"
        )
        print(f"  ✓ CI plot saved")
    except Exception as e:
        print(f"  ERROR creating CI plot: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. ANOVA results table
    print("Creating ANOVA results table...")
    try:
        plt.close('all')
        visualizer.create_anova_results_table(
            anova_result=results['anova'],
            games_howell_results=results['games_howell'],
            output_path=anova_folder / "anova_results_table.png"
        )
        print(f"  ✓ ANOVA table saved")
    except Exception as e:
        print(f"  ERROR creating ANOVA table: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Complete overview
    print("Creating complete ANOVA overview...")
    try:
        plt.close('all')
        visualizer.create_complete_anova_overview(
            analysis_results=results,
            output_path=anova_folder / "anova_complete_overview.png"
        )
        print(f"  ✓ Complete overview saved")
    except Exception as e:
        print(f"  ERROR creating overview: {e}")
        import traceback
        traceback.print_exc()
    
    # Verify files were created
    print("\n--- Verifying saved files ---")
    expected_files = [
        "anova_results.txt",
        "bootstrap_distributions.png",
        "ifss_confidence_intervals.png",
        "anova_results_table.png",
        "anova_complete_overview.png"
    ]
    
    for filename in expected_files:
        filepath = anova_folder / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {filename}: {size_kb:.1f} KB")
        else:
            print(f"  ✗ {filename}: NOT FOUND!")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"ANOVA Analysis Summary:")
    print(f"  Groups analyzed: {len(ifss_data)}")
    if excluded_groups:
        print(f"  Groups excluded (n<2): {len(excluded_groups)}")
    print(f"  Results folder: {anova_folder}")
    print("="*60)
    
    return results


def process_multiple_series(all_results: Dict[str, dict], anova_enabled: bool = False):
    """
    Processes multiple measurement series together for comparison.
    
    Args:
        all_results: Dictionary with {series_name: results_dict}
        anova_enabled: Whether to perform ANOVA analysis
    """
    print(f"\n{'='*60}")
    print(f"=== Multi-Series Analysis ===")
    print(f"Comparing {len(all_results)} measurement series")
    if anova_enabled:
        print("ANOVA + Bootstrap: ENABLED")
    print(f"{'='*60}")
    
    # Print comparison summary
    print("\nComparison Summary:")
    print("-" * 60)
    print(f"{'Series':<30} {'n':>6} {'F_max [N]':>12} {'IFSS [MPa]':>12}")
    print("-" * 60)
    
    for series_name, results in all_results.items():
        n = len(results['f_max_values'])
        f_max_mean = sum(results['f_max_values']) / n
        ifss_mean = sum(results['ifss_values']) / n
        print(f"{series_name:<30} {n:>6} {f_max_mean:>12.4f} {ifss_mean:>12.2f}")
    
    print("-" * 60)
    
    # Ask user for save location
    root = tk.Tk()
    root.withdraw()
    root.lift()
    root.attributes('-topmost', True)
    
    save_folder = filedialog.askdirectory(
        title="Select folder to save multi-series analysis"
    )
    
    if not save_folder:
        print("No save location selected, skipping multi-series export")
        return
    
    save_path = Path(save_folder)
    
    # Create subfolders
    plots_folder = save_path / "comparison_plots"
    plots_folder.mkdir(exist_ok=True, parents=True)
    
    excel_folder = save_path / "comparison_excel"
    excel_folder.mkdir(exist_ok=True, parents=True)
    
    # Create comparison visualizations
    save_comparison_visualizations(all_results, plots_folder)
    
    # Perform ANOVA analysis if enabled
    if anova_enabled:
        anova_results = perform_anova_analysis(all_results, save_path)
        
        # Optionally add ANOVA results to Excel export
        if anova_results:
            print("\nANOVA analysis completed successfully!")
    
    # Export to Excel
    print("\nExporting multi-series Excel file...")
    exporter = DataExporter()
    excel_path = exporter.save_multiple_series_to_excel(
        all_results=all_results,
        output_folder=excel_folder
    )
    print(f"  Saved: {excel_path.name}")
    
    print(f"\n✓ All multi-series exports saved to: {save_path}")
    print("="*60)


def process_single_series(
    file_paths: List[str], 
    series_name: str = "SFPO_analysis",
    plots_save_folder: Optional[Path] = None
):
    """
    Processes a single measurement series.
    
    Args:
        file_paths: List of file paths in the series
        series_name: Name for this series (used in plot filenames)
        plots_save_folder: Optional predefined folder for saving plots
    """
    # Step 1: Categorize files
    print("\n=== Step 1: Categorizing Files ===")
    successful_files, broken_files = MeasurementFileFilter.categorize_files(file_paths)
    
    print(f"Found {len(successful_files)} successful measurements (ZZa_)")
    print(f"Found {len(broken_files)} broken fiber measurements (ZZxa_)")
    
    if not successful_files:
        print("No successful measurements found!")
        return None
    
    # Step 2: Clean and parse successful measurements
    print("\n=== Step 2: Cleaning Data ===")
    cleaner = DataCleanUp()
    cleaned_measurements = []
    
    for file_path in successful_files:
        try:
            cleaned = cleaner.clean_measurement_pairs(file_path)
            if cleaned:
                cleaned_measurements.append({
                    'file_path': file_path,
                    'data': cleaned
                })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print(f"Successfully cleaned {len(cleaned_measurements)} measurements")
    
    if not cleaned_measurements:
        print("No valid measurements after cleaning!")
        return None
    
    # Step 3: Calculate mechanical properties
    print("\n=== Step 3: Calculating Mechanical Properties ===")
    calculator = MechanicsCalculator()
    
    # Storage for all calculated values
    f_max_values = []
    embedding_lengths = []
    fiber_diameters = []
    ifss_values = []
    work_total_values = []
    work_debonding_values = []
    work_friction_values = []
    measurements_data = []
    
    for measurement in cleaned_measurements:
        data = measurement['data']
        file_path = measurement['file_path']
        filename = Path(file_path).name
        
        measurements_data.append(data)
        
        diameter = calculator.read_fiber_diameter(file_path)
        fiber_diameters.append(diameter)
        
        f_max = calculator.calculate_f_max(data)
        f_max_values.append(f_max)
        
        embed_length = calculator.calculate_embedding_length(data)
        embedding_lengths.append(embed_length)
        
        ifss = calculator.calculate_ifss(f_max, diameter, embed_length)
        ifss_values.append(ifss)
        
        work_total = calculator.calculate_work_total(data)
        work_debonding = calculator.calculate_work_debonding(data)
        work_friction = calculator.calculate_work_friction(data)
        
        work_total_values.append(work_total)
        work_debonding_values.append(work_debonding)
        work_friction_values.append(work_friction)
        
        print(f"\n  {filename}:")
        print(f"    Fiber diameter: {diameter:.2f} µm")
        print(f"    F_max: {f_max:.4f} N")
        print(f"    Embedding length: {embed_length:.2f} µm")
        print(f"    IFSS: {ifss:.2f} MPa")
        print(f"    Work (total): {work_total:.4f} µJ")
        print(f"    Work (debonding): {work_debonding:.4f} µJ")
        print(f"    Work (friction): {work_friction:.4f} µJ")
    
    # Step 4: Statistical analysis
    print("\n" + "="*60)
    print("=== Step 4: Statistical Analysis ===")
    print("="*60)
    
    diameter_mean, diameter_std = calculator.calculate_mean_std(fiber_diameters)
    print(f"\nFiber Diameter:")
    print(f"  Mean: {diameter_mean:.2f} µm")
    print(f"  Std Dev: {diameter_std:.2f} µm")
    print(f"  n = {len(fiber_diameters)}")
    
    f_max_mean, f_max_std = calculator.calculate_mean_std(f_max_values)
    print(f"\nF_max:")
    print(f"  Mean: {f_max_mean:.4f} N")
    print(f"  Std Dev: {f_max_std:.4f} N")
    print(f"  n = {len(f_max_values)}")
    
    embed_mean, embed_std = calculator.calculate_mean_std(embedding_lengths)
    print(f"\nEmbedding Length:")
    print(f"  Mean: {embed_mean:.2f} µm")
    print(f"  Std Dev: {embed_std:.2f} µm")
    print(f"  n = {len(embedding_lengths)}")
    
    ifss_mean, ifss_std = calculator.calculate_mean_std(ifss_values)
    print(f"\nIFSS:")
    print(f"  Mean: {ifss_mean:.2f} MPa")
    print(f"  Std Dev: {ifss_std:.2f} MPa")
    print(f"  n = {len(ifss_values)}")
    
    work_total_mean, work_total_std = calculator.calculate_mean_std(work_total_values)
    work_debond_mean, work_debond_std = calculator.calculate_mean_std(work_debonding_values)
    work_friction_mean, work_friction_std = calculator.calculate_mean_std(work_friction_values)
    
    print(f"\nWork (Total):")
    print(f"  Mean: {work_total_mean:.4f} µJ")
    print(f"  Std Dev: {work_total_std:.4f} µJ")
    print(f"  n = {len(work_total_values)}")
    
    print(f"\nWork (Debonding):")
    print(f"  Mean: {work_debond_mean:.4f} µJ")
    print(f"  Std Dev: {work_debond_std:.4f} µJ")
    print(f"  n = {len(work_debonding_values)}")
    
    print(f"\nWork (Friction):")
    print(f"  Mean: {work_friction_mean:.4f} µJ")
    print(f"  Std Dev: {work_friction_std:.4f} µJ")
    print(f"  n = {len(work_friction_values)}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    
    # Step 5: Analyze broken fibers for Kaplan-Meier
    print("\n=== Step 5: Broken Fiber Analysis (Kaplan-Meier) ===")
    broken_fiber_results = analyze_broken_fibers(file_paths, series_name)
    
    # Prepare results dictionary
    results = {
        'measurements': cleaned_measurements,
        'measurements_data': measurements_data,
        'fiber_diameters': fiber_diameters,
        'f_max_values': f_max_values,
        'embedding_lengths': embedding_lengths,
        'ifss_values': ifss_values,
        'work_total': work_total_values,
        'work_debonding': work_debonding_values,
        'work_friction': work_friction_values,
        'broken_forces': broken_fiber_results['broken_forces'],
        'broken_labels': broken_fiber_results['broken_labels']
    }
    
    # Create visualizations
    save_visualizations(results, series_name=series_name, predefined_save_folder=plots_save_folder)
    
    return results


def parse_measurements(files):
    """
    Legacy function - now handled in process_single_series.
    Kept for backwards compatibility.
    """
    cleaner = DataCleanUp()
    measurements = []
    
    if isinstance(files, str):
        files = [files]
    
    for file_path in files:
        try:
            cleaned = cleaner.clean_measurement_pairs(file_path)
            measurements.append(cleaned)
        except Exception as e:
            print(f"Error: {e}")
    
    return measurements


def analyze(measurements):
    """
    Performs all calculations.
    Currently integrated into process_single_series.
    """
    pass


def export_results(results):
    """
    Exports the results.
    Now implemented in export_single_series() and process_multiple_series()
    """
    pass


if __name__ == "__main__":
    main()