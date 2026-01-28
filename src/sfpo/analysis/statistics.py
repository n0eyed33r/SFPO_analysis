# src/sfpo/analysis/statistics.py
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Backend für Nicht-GUI Umgebungen


class StatisticsVisualizer:
    """
    Creates statistical visualizations (boxplots, etc.) for SFPO data.
    """
    
    @staticmethod
    def create_boxplot(
        data: List[float],
        title: str,
        ylabel: str,
        output_path: Optional[Path] = None,
        show_plot: bool = False
    ):
        """
        Creates a single boxplot.
        
        Args:
            data: List of values to plot
            title: Plot title
            ylabel: Y-axis label
            output_path: Path to save the plot (if None, plot is not saved)
            show_plot: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create boxplot
        bp = ax.boxplot([data], patch_artist=True, widths=0.5)
        
        # Style the boxplot
        for box in bp['boxes']:
            box.set(facecolor='lightblue', linewidth=2)
        for whisker in bp['whiskers']:
            whisker.set(linewidth=2, color='blue')
        for cap in bp['caps']:
            cap.set(linewidth=2, color='blue')
        for median in bp['medians']:
            median.set(linewidth=2, color='red')
        for flier in bp['fliers']:
            flier.set(marker='o', markerfacecolor='red', markersize=8, alpha=0.5)
        
        # Labels and title
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_xticklabels([''], fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        median_val = np.median(data)
        n = len(data)
        
        stats_text = f'n = {n}\nMean = {mean_val:.4f}\nStd = {std_val:.4f}\nMedian = {median_val:.4f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5), fontsize=11)
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved boxplot: {output_path}")
        
        # Show if requested
        if show_plot:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def create_combined_boxplots(
        data_dict: Dict[str, List[float]],
        title: str,
        ylabel: str,
        output_path: Optional[Path] = None,
        show_plot: bool = False
    ):
        """
        Creates a boxplot with multiple datasets side by side.
        
        Args:
            data_dict: Dictionary with {label: data_list}
            title: Plot title
            ylabel: Y-axis label
            output_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Prepare data
        labels = list(data_dict.keys())
        data_lists = [data_dict[label] for label in labels]
        
        # Create boxplot
        bp = ax.boxplot(data_lists, labels=labels, patch_artist=True, widths=0.6)
        
        # Style the boxplot
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for box, color in zip(bp['boxes'], colors):
            box.set(facecolor=color, linewidth=2, alpha=0.7)
        for whisker in bp['whiskers']:
            whisker.set(linewidth=2, color='blue')
        for cap in bp['caps']:
            cap.set(linewidth=2, color='blue')
        for median in bp['medians']:
            median.set(linewidth=2.5, color='red')
        for flier in bp['fliers']:
            flier.set(marker='o', markerfacecolor='red', markersize=6, alpha=0.5)
        
        # Labels and title
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
        ax.set_xlabel('Measurement Series', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined boxplot: {output_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def create_all_boxplots_grid(
        f_max: List[float],
        embedding_length: List[float],
        ifss: List[float],
        work_total: List[float],
        work_debonding: List[float],
        work_friction: List[float],
        output_path: Optional[Path] = None,
        show_plot: bool = False
    ):
        """
        Creates a grid of 6 boxplots showing all parameters at once.
        
        Args:
            f_max: F_max values
            embedding_length: Embedding length values
            ifss: IFSS values
            work_total: Total work values
            work_debonding: Debonding work values
            work_friction: Friction work values
            output_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SFPO Analysis - Complete Overview', fontsize=20, fontweight='bold')
        
        # Data and labels for each subplot
        data_sets = [
            (f_max, 'F_max [N]', 'Maximum Force'),
            (embedding_length, 'Length [µm]', 'Embedding Length'),
            (ifss, 'IFSS [MPa]', 'Interface Shear Strength'),
            (work_total, 'Work [µJ]', 'Total Work'),
            (work_debonding, 'Work [µJ]', 'Debonding Work'),
            (work_friction, 'Work [µJ]', 'Friction Work')
        ]
        
        # Create each subplot
        for ax, (data, ylabel, title) in zip(axes.flat, data_sets):
            bp = ax.boxplot([data], patch_artist=True, widths=0.5)
            
            # Style
            for box in bp['boxes']:
                box.set(facecolor='lightblue', linewidth=2, alpha=0.7)
            for whisker in bp['whiskers']:
                whisker.set(linewidth=2, color='blue')
            for cap in bp['caps']:
                cap.set(linewidth=2, color='blue')
            for median in bp['medians']:
                median.set(linewidth=2.5, color='red')
            for flier in bp['fliers']:
                flier.set(marker='o', markerfacecolor='red', markersize=8, alpha=0.5)
            
            # Labels
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_xticklabels([''])
            ax.grid(True, alpha=0.3, axis='y')
            
            # Statistics text
            mean_val = np.mean(data)
            std_val = np.std(data, ddof=1)
            median_val = np.median(data)
            n = len(data)
            
            stats_text = f'n={n}\nμ={mean_val:.3f}\nσ={std_val:.3f}\nM={median_val:.3f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5), fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved complete overview: {output_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def create_force_displacement_plot(
        measurements_data: List[List[Tuple[float, float]]],
        title: str,
        output_path: Optional[Path] = None,
        max_displacement: float = 1000.0,
        max_force: float = 0.3,
        show_plot: bool = False
    ):
        """
        Creates a force-displacement plot for all measurements in a series.
        
        Uses the plasma colormap to distinguish between measurements.
        
        Args:
            measurements_data: List of measurements, each is a list of (displacement, force) tuples
            title: Plot title
            output_path: Path to save the plot
            max_displacement: Maximum displacement for x-axis [µm]
            max_force: Maximum force for y-axis [N]
            show_plot: Whether to display the plot
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Use Plasma colormap
        n_measurements = len(measurements_data)
        colors = plt.cm.plasma(np.linspace(0, 1, max(n_measurements, 10)))
        
        # Plot each measurement
        for i, (measurement, color) in enumerate(zip(measurements_data, colors)):
            if measurement:
                distances, forces = zip(*measurement)
                ax.plot(distances, forces, color=color, linewidth=2, label=f'Measurement {i + 1}')
        
        # Set axis limits
        ax.set_xlim(0, max_displacement)
        ax.set_ylim(0, max_force)
        
        # Set ticks
        tick_step_x = max_displacement / 5
        ax.set_xticks(np.arange(0, max_displacement + 1, tick_step_x))
        ax.set_yticks(np.arange(0, max_force + 0.01, 0.05))
        
        # Format ticks
        ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        
        # Labels
        ax.set_xlabel('Displacement [µm]', fontsize=20, fontweight='bold', labelpad=10)
        ax.set_ylabel('Force [N]', fontsize=20, fontweight='bold', labelpad=10)
        ax.set_title(title, fontsize=22, fontweight='bold', pad=20)
        
        # Grid
        ax.grid(True, alpha=0.3, linewidth=1.5)
        
        # Legend (optional, can be removed if too cluttered)
        if n_measurements <= 15:  # Only show legend if not too many measurements
            ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
        
        # Spine width
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved force-displacement plot: {output_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    @staticmethod
    def create_kaplan_meier_plot(
        broken_forces: List[float],
        broken_labels: List[str],
        output_path: Optional[Path] = None,
        show_plot: bool = False
    ):
        """
        Creates a Kaplan-Meier survival plot for broken fibers.
        
        This shows the cumulative probability of fiber survival
        as a function of applied force.
        
        Args:
            broken_forces: F_max values where fibers broke
            broken_labels: Labels for each broken fiber
            output_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        if not broken_forces:
            print("No broken fiber data available for Kaplan-Meier plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort forces
        sorted_forces = sorted(broken_forces)
        n = len(sorted_forces)
        
        # Calculate survival probability at each event
        survival_prob = [(n - i) / n for i in range(n + 1)]
        forces_plot = [0] + sorted_forces
        
        # Create step plot
        ax.step(forces_plot, survival_prob, where='post', linewidth=2, color='blue')
        ax.plot(forces_plot, survival_prob, 'o', markersize=8, color='red', alpha=0.6)
        
        # Labels and title
        ax.set_title('Kaplan-Meier Survival Analysis\nFiber Break Events', 
                     fontsize=16, fontweight='bold')
        ax.set_xlabel('Applied Force [N]', fontsize=14, fontweight='bold')
        ax.set_ylabel('Survival Probability', fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        if sorted_forces:
            median_force = np.median(sorted_forces)
            mean_force = np.mean(sorted_forces)
            stats_text = f'n = {n}\nMedian Force = {median_force:.4f} N\nMean Force = {mean_force:.4f} N'
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=11)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved Kaplan-Meier plot: {output_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    # =========================================================================
    # NEW: ANOVA and Bootstrap Visualization Methods
    # =========================================================================
    
    @staticmethod
    def create_bootstrap_distribution_plot(
        bootstrap_results: Dict,
        output_path: Optional[Path] = None,
        show_plot: bool = False
    ):
        """
        Creates histogram plots of bootstrap distributions for each group.
        
        Shows the distribution of bootstrap means with BCa confidence intervals.
        
        Args:
            bootstrap_results: Dictionary with {group_name: BootstrapResult}
            output_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        n_groups = len(bootstrap_results)
        
        # Determine grid layout
        if n_groups <= 3:
            n_cols = n_groups
            n_rows = 1
        elif n_groups <= 6:
            n_cols = 3
            n_rows = 2
        else:
            n_cols = 3
            n_rows = (n_groups + 2) // 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle('Bootstrap Distributions of IFSS Means (BCa)', 
                     fontsize=16, fontweight='bold', y=1.02)
        
        # Flatten axes for iteration
        if n_groups == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # Colors for groups
        colors = plt.cm.Set2(np.linspace(0, 1, n_groups))
        
        for idx, (name, result) in enumerate(bootstrap_results.items()):
            ax = axes[idx]
            
            # Histogram of bootstrap means
            ax.hist(result.bootstrap_means, bins=50, density=True, 
                    alpha=0.7, color=colors[idx], edgecolor='black', linewidth=0.5)
            
            # Original mean as vertical line
            ax.axvline(result.original_mean, color='red', linewidth=2, 
                       linestyle='-', label=f'Mean = {result.original_mean:.2f}')
            
            # CI bounds
            ax.axvline(result.ci_lower, color='darkgreen', linewidth=2, 
                       linestyle='--', label=f'95% CI lower = {result.ci_lower:.2f}')
            ax.axvline(result.ci_upper, color='darkgreen', linewidth=2, 
                       linestyle='--', label=f'95% CI upper = {result.ci_upper:.2f}')
            
            # Shade CI region
            ax.axvspan(result.ci_lower, result.ci_upper, alpha=0.2, color='green')
            
            # Labels
            ax.set_title(f'{name}\n(n={result.n_samples}, {result.n_iterations} iterations)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('IFSS [MPa]', fontsize=11)
            ax.set_ylabel('Density', fontsize=11)
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_groups, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved bootstrap distribution plot: {output_path}")
        else:
            if show_plot:
                plt.show()
            plt.close(fig)
    
    @staticmethod
    def create_ifss_confidence_interval_plot(
        bootstrap_results: Dict,
        output_path: Optional[Path] = None,
        show_plot: bool = False
    ):
        """
        Creates a plot showing IFSS means with bootstrap confidence intervals.
        
        Error bars show the BCa 95% confidence intervals.
        
        Args:
            bootstrap_results: Dictionary with {group_name: BootstrapResult}
            output_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        names = list(bootstrap_results.keys())
        means = [r.original_mean for r in bootstrap_results.values()]
        ci_lowers = [r.ci_lower for r in bootstrap_results.values()]
        ci_uppers = [r.ci_upper for r in bootstrap_results.values()]
        n_samples = [r.n_samples for r in bootstrap_results.values()]
        
        # Calculate error bar sizes (asymmetric, ensure non-negative)
        lower_errors = [max(0, m - l) for m, l in zip(means, ci_lowers)]
        upper_errors = [max(0, u - m) for u, m in zip(means, ci_uppers)]
        
        # X positions
        x_pos = np.arange(len(names))
        
        # Colors
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        
        # Create bar plot with error bars
        bars = ax.bar(x_pos, means, color=colors, edgecolor='black', 
                      linewidth=1.5, alpha=0.8)
        
        # Error bars (asymmetric)
        ax.errorbar(x_pos, means, yerr=[lower_errors, upper_errors],
                    fmt='none', ecolor='black', capsize=8, capthick=2, 
                    elinewidth=2)
        
        # Add sample size labels on bars
        for i, (bar, n) in enumerate(zip(bars, n_samples)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'n={n}', ha='center', va='center', fontsize=11,
                    fontweight='bold', color='white')
        
        # Labels and title
        ax.set_title('IFSS Comparison with 95% Bootstrap CI (BCa)', 
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('IFSS [MPa]', fontsize=14, fontweight='bold')
        ax.set_xlabel('Measurement Series', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add legend for CI
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor='gray', alpha=0.8, edgecolor='black', label='Mean IFSS'),
            Line2D([0], [0], color='black', linewidth=2, label='95% BCa CI')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved CI plot: {output_path}")
        else:
            if show_plot:
                plt.show()
            plt.close(fig)
    
    @staticmethod
    def create_anova_results_table(
        anova_result,
        games_howell_results: Optional[List] = None,
        output_path: Optional[Path] = None,
        show_plot: bool = False
    ):
        """
        Creates a graphical table showing ANOVA and Games-Howell results.
        
        Args:
            anova_result: ANOVAResult dataclass instance
            games_howell_results: List of GamesHowellResult (optional)
            output_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        # Determine figure height based on content
        base_height = 3
        if games_howell_results:
            base_height += 1 + len(games_howell_results) * 0.5
        
        fig, ax = plt.subplots(figsize=(12, base_height))
        ax.axis('off')
        
        # Title
        title = 'ANOVA Results Summary'
        if anova_result.is_significant:
            title += ' (SIGNIFICANT)'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # ANOVA table data
        anova_data = [
            ['Source', 'SS', 'df', 'MS', 'F', 'p-value'],
            ['Between Groups', 
             f'{anova_result.ss_between:.4f}', 
             f'{anova_result.df_between}',
             f'{anova_result.ms_between:.4f}',
             f'{anova_result.f_statistic:.4f}',
             f'{anova_result.p_value:.4f}'],
            ['Within Groups',
             f'{anova_result.ss_within:.4f}',
             f'{anova_result.df_within}',
             f'{anova_result.ms_within:.4f}',
             '-',
             '-'],
            ['Total',
             f'{anova_result.ss_between + anova_result.ss_within:.4f}',
             f'{anova_result.df_between + anova_result.df_within}',
             '-',
             '-',
             '-']
        ]
        
        # Create ANOVA table
        anova_table = ax.table(
            cellText=anova_data[1:],
            colLabels=anova_data[0],
            loc='upper center',
            cellLoc='center',
            colWidths=[0.2, 0.15, 0.1, 0.15, 0.15, 0.15]
        )
        anova_table.auto_set_font_size(False)
        anova_table.set_fontsize(11)
        anova_table.scale(1.2, 1.8)
        
        # Style header row
        for j in range(len(anova_data[0])):
            anova_table[(0, j)].set_facecolor('#4472C4')
            anova_table[(0, j)].set_text_props(color='white', fontweight='bold')
        
        # Highlight significance
        if anova_result.is_significant:
            anova_table[(1, 5)].set_facecolor('#92D050')
            anova_table[(1, 5)].set_text_props(fontweight='bold')
        
        # Games-Howell table if available
        if games_howell_results:
            # Add some vertical spacing
            y_pos = 0.4 - len(games_howell_results) * 0.08
            
            ax.text(0.5, y_pos + 0.1, 'Games-Howell Post-Hoc Test', 
                    transform=ax.transAxes, fontsize=14, fontweight='bold',
                    ha='center')
            
            gh_data = [['Comparison', 'Mean Diff', 'SE', 't', 'df', 'p-value', 'Sig.']]
            for gh in games_howell_results:
                sig_marker = '✓' if gh.is_significant else ''
                gh_data.append([
                    f'{gh.group1} vs {gh.group2}',
                    f'{gh.mean_diff:.3f}',
                    f'{gh.std_error:.3f}',
                    f'{gh.t_statistic:.3f}',
                    f'{gh.df:.1f}',
                    f'{gh.p_value:.4f}',
                    sig_marker
                ])
            
            gh_table = ax.table(
                cellText=gh_data[1:],
                colLabels=gh_data[0],
                loc='lower center',
                cellLoc='center',
                colWidths=[0.25, 0.12, 0.1, 0.1, 0.08, 0.12, 0.08]
            )
            gh_table.auto_set_font_size(False)
            gh_table.set_fontsize(10)
            gh_table.scale(1.2, 1.6)
            
            # Style header row
            for j in range(len(gh_data[0])):
                gh_table[(0, j)].set_facecolor('#4472C4')
                gh_table[(0, j)].set_text_props(color='white', fontweight='bold')
            
            # Highlight significant comparisons
            for i, gh in enumerate(games_howell_results, 1):
                if gh.is_significant:
                    for j in range(len(gh_data[0])):
                        gh_table[(i, j)].set_facecolor('#FFE699')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved ANOVA results table: {output_path}")
        else:
            if show_plot:
                plt.show()
            plt.close(fig)
    
    @staticmethod
    def create_complete_anova_overview(
        analysis_results: Dict,
        output_path: Optional[Path] = None,
        show_plot: bool = False
    ):
        """
        Creates a complete overview figure with all ANOVA-related visualizations.
        
        Includes:
        - Bootstrap CI comparison
        - ANOVA results summary
        - Games-Howell pairwise comparisons (if significant)
        
        Args:
            analysis_results: Dictionary from IFSSAnalyzer.analyze()
            output_path: Path to save the plot
            show_plot: Whether to display the plot
        """
        has_posthoc = analysis_results['games_howell'] is not None
        
        if has_posthoc:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
            
            ax1 = fig.add_subplot(gs[0, :])  # CI plot spans top row
            ax2 = fig.add_subplot(gs[1, 0])  # ANOVA table
            ax3 = fig.add_subplot(gs[1, 1])  # Post-hoc info
        else:
            fig = plt.figure(figsize=(14, 8))
            gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1])
            
            ax1 = fig.add_subplot(gs[0])  # CI plot
            ax2 = fig.add_subplot(gs[1])  # ANOVA table
            ax3 = None
        
        fig.suptitle('IFSS ANOVA Analysis - Complete Overview', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # ===== Plot 1: CI comparison =====
        bootstrap_results = analysis_results['bootstrap']
        names = list(bootstrap_results.keys())
        means = [r.original_mean for r in bootstrap_results.values()]
        ci_lowers = [r.ci_lower for r in bootstrap_results.values()]
        ci_uppers = [r.ci_upper for r in bootstrap_results.values()]
        n_samples = [r.n_samples for r in bootstrap_results.values()]
        
        # Calculate error bar sizes (asymmetric, ensure non-negative)
        lower_errors = [max(0, m - l) for m, l in zip(means, ci_lowers)]
        upper_errors = [max(0, u - m) for u, m in zip(means, ci_uppers)]
        
        x_pos = np.arange(len(names))
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        
        bars = ax1.bar(x_pos, means, color=colors, edgecolor='black', 
                       linewidth=1.5, alpha=0.8)
        ax1.errorbar(x_pos, means, yerr=[lower_errors, upper_errors],
                     fmt='none', ecolor='black', capsize=8, capthick=2, elinewidth=2)
        
        for i, (bar, n) in enumerate(zip(bars, n_samples)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                     f'n={n}', ha='center', va='center', fontsize=11,
                     fontweight='bold', color='white')
        
        ax1.set_title('IFSS Means with 95% Bootstrap CI (BCa)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('IFSS [MPa]', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # ===== Plot 2: ANOVA table =====
        ax2.axis('off')
        anova = analysis_results['anova']
        
        sig_text = '✓ SIGNIFICANT' if anova.is_significant else '✗ NOT SIGNIFICANT'
        sig_color = '#92D050' if anova.is_significant else '#FF6B6B'
        
        anova_text = (
            f"One-Way ANOVA Results\n"
            f"{'─' * 35}\n"
            f"F({anova.df_between}, {anova.df_within}) = {anova.f_statistic:.4f}\n"
            f"p-value = {anova.p_value:.4f}\n"
            f"α = {anova.alpha}\n"
            f"{'─' * 35}\n"
            f"{sig_text}"
        )
        
        ax2.text(0.5, 0.5, anova_text, transform=ax2.transAxes,
                 fontsize=13, family='monospace', ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='white', 
                          edgecolor=sig_color, linewidth=3))
        ax2.set_title('Statistical Test', fontsize=14, fontweight='bold')
        
        # ===== Plot 3: Post-hoc results (if available) =====
        if ax3 and has_posthoc:
            ax3.axis('off')
            
            gh_text = "Games-Howell Post-Hoc\n" + "─" * 30 + "\n"
            for gh in analysis_results['games_howell']:
                sig_mark = "✓" if gh.is_significant else " "
                gh_text += f"{sig_mark} {gh.group1} vs {gh.group2}:\n"
                gh_text += f"   Δ = {gh.mean_diff:.2f}, p = {gh.p_value:.4f}\n"
            
            ax3.text(0.5, 0.5, gh_text, transform=ax3.transAxes,
                     fontsize=11, family='monospace', ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6', 
                              edgecolor='#FFB347', linewidth=2))
            ax3.set_title('Pairwise Comparisons', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved complete ANOVA overview: {output_path}")
        else:
            if show_plot:
                plt.show()
            plt.close(fig)