# src/sfpo/analysis/anova.py
"""
ANOVA analysis with Bootstrap confidence intervals and Games-Howell post-hoc test
for SFPO measurement data.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BootstrapResult:
    """Results from bootstrap analysis for a single group."""
    group_name: str
    original_mean: float
    bootstrap_means: np.ndarray
    ci_lower: float
    ci_upper: float
    std_error: float
    n_iterations: int
    n_samples: int


@dataclass
class ANOVAResult:
    """Results from one-way ANOVA analysis."""
    f_statistic: float
    p_value: float
    df_between: int
    df_within: int
    ss_between: float
    ss_within: float
    ms_between: float
    ms_within: float
    is_significant: bool
    alpha: float = 0.05


@dataclass
class GamesHowellResult:
    """Results from Games-Howell post-hoc comparison."""
    group1: str
    group2: str
    mean_diff: float
    std_error: float
    t_statistic: float
    df: float
    p_value: float
    ci_lower: float
    ci_upper: float
    is_significant: bool


class BootstrapAnalyzer:
    """
    Performs BCa (Bias-Corrected and Accelerated) bootstrap analysis.
    """
    
    def __init__(self, confidence_level: float = 0.95, random_seed: int = 42):
        """
        Initialize bootstrap analyzer.
        
        Args:
            confidence_level: Confidence level for intervals (default 0.95)
            random_seed: Random seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def determine_iterations(self, n_samples: int, success_rate: float = 1.0) -> int:
        """
        Determines the number of bootstrap iterations based on sample size and success rate.
        
        Args:
            n_samples: Number of samples in the group
            success_rate: Proportion of successful measurements (0.0 to 1.0)
            
        Returns:
            Number of iterations (1000 or 10000)
        """
        if success_rate <= 0.30:
            return 10000
        return 1000
    
    def bootstrap_mean(
        self, 
        data: np.ndarray, 
        n_iterations: int = 1000
    ) -> Tuple[np.ndarray, float, float]:
        """
        Performs BCa bootstrap for the mean.
        
        Args:
            data: Array of values
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Tuple of (bootstrap_means, ci_lower, ci_upper)
        """
        n = len(data)
        original_mean = float(np.mean(data))
        
        # Generate bootstrap samples
        bootstrap_means = np.zeros(n_iterations)
        for i in range(n_iterations):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means[i] = np.mean(sample)
        
        # BCa correction
        ci_lower, ci_upper = self._bca_interval(
            data, bootstrap_means, original_mean, self.confidence_level
        )
        
        return bootstrap_means, ci_lower, ci_upper
    
    def _bca_interval(
        self, 
        data: np.ndarray, 
        bootstrap_means: np.ndarray,
        original_mean: float,
        confidence_level: float
    ) -> Tuple[float, float]:
        """
        Calculates BCa (Bias-Corrected and Accelerated) confidence interval.
        
        Args:
            data: Original data array
            bootstrap_means: Array of bootstrap mean estimates
            original_mean: Mean of original data
            confidence_level: Confidence level (e.g., 0.95)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        n = len(data)
        alpha = 1 - confidence_level
        
        # Bias correction factor (z0)
        proportion_below = float(np.mean(bootstrap_means < original_mean))
        # Avoid extreme values
        proportion_below = float(np.clip(proportion_below, 0.0001, 0.9999))
        z0 = float(stats.norm.ppf(proportion_below))
        
        # Acceleration factor (a) using jackknife
        jackknife_means = np.zeros(n)
        for i in range(n):
            jackknife_sample = np.delete(data, i)
            jackknife_means[i] = np.mean(jackknife_sample)
        
        jackknife_mean = float(np.mean(jackknife_means))
        numerator = float(np.sum((jackknife_mean - jackknife_means) ** 3))
        denominator = 6 * (float(np.sum((jackknife_mean - jackknife_means) ** 2)) ** 1.5)
        
        if denominator == 0:
            a = 0.0
        else:
            a = numerator / denominator
        
        # Adjusted percentiles
        z_alpha_lower = float(stats.norm.ppf(alpha / 2))
        z_alpha_upper = float(stats.norm.ppf(1 - alpha / 2))
        
        # BCa adjusted percentiles
        def adjusted_percentile(z_alpha: float) -> float:
            num = z0 + z_alpha
            denom = 1 - a * (z0 + z_alpha)
            if denom == 0:
                return float(stats.norm.cdf(z_alpha))
            return float(stats.norm.cdf(z0 + num / denom))
        
        alpha_lower = adjusted_percentile(z_alpha_lower)
        alpha_upper = adjusted_percentile(z_alpha_upper)
        
        # Ensure percentiles are within bounds
        alpha_lower = float(np.clip(alpha_lower, 0.001, 0.999))
        alpha_upper = float(np.clip(alpha_upper, 0.001, 0.999))
        
        ci_lower = float(np.percentile(bootstrap_means, alpha_lower * 100))
        ci_upper = float(np.percentile(bootstrap_means, alpha_upper * 100))
        
        return ci_lower, ci_upper
    
    def analyze_group(
        self, 
        group_name: str, 
        data: List[float],
        success_rate: float = 1.0
    ) -> BootstrapResult:
        """
        Performs complete bootstrap analysis for a single group.
        
        Args:
            group_name: Name of the group
            data: List of values
            success_rate: Success rate of measurements
            
        Returns:
            BootstrapResult with all analysis results
        """
        data_array = np.array(data)
        n_iterations = self.determine_iterations(len(data), success_rate)
        
        bootstrap_means, ci_lower, ci_upper = self.bootstrap_mean(
            data_array, n_iterations
        )
        
        return BootstrapResult(
            group_name=group_name,
            original_mean=float(np.mean(data_array)),
            bootstrap_means=bootstrap_means,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_error=float(np.std(bootstrap_means)),
            n_iterations=n_iterations,
            n_samples=len(data)
        )


class ANOVAAnalyzer:
    """
    Performs one-way ANOVA analysis with Games-Howell post-hoc test.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize ANOVA analyzer.
        
        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha
    
    def one_way_anova(self, groups: Dict[str, List[float]]) -> ANOVAResult:
        """
        Performs one-way ANOVA.
        
        Args:
            groups: Dictionary with {group_name: values_list}
            
        Returns:
            ANOVAResult with F-statistic, p-value, and other statistics
        """
        # Convert to list of arrays
        group_data = [np.array(values) for values in groups.values()]
        
        # Calculate overall mean
        all_data = np.concatenate(group_data)
        grand_mean = float(np.mean(all_data))
        
        # Calculate group statistics
        n_groups = len(group_data)
        n_total = len(all_data)
        
        # Sum of squares between groups
        ss_between = 0.0
        for data in group_data:
            n_i = len(data)
            mean_i = float(np.mean(data))
            ss_between += n_i * (mean_i - grand_mean) ** 2
        
        # Sum of squares within groups
        ss_within = 0.0
        for data in group_data:
            mean_i = float(np.mean(data))
            ss_within += float(np.sum((data - mean_i) ** 2))
        
        # Degrees of freedom
        df_between = n_groups - 1
        df_within = n_total - n_groups
        
        # Mean squares
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        
        # F-statistic
        f_statistic = ms_between / ms_within
        
        # P-value
        p_value = float(1 - stats.f.cdf(f_statistic, df_between, df_within))
        
        return ANOVAResult(
            f_statistic=f_statistic,
            p_value=p_value,
            df_between=df_between,
            df_within=df_within,
            ss_between=ss_between,
            ss_within=ss_within,
            ms_between=ms_between,
            ms_within=ms_within,
            is_significant=p_value < self.alpha,
            alpha=self.alpha
        )
    
    def games_howell(self, groups: Dict[str, List[float]]) -> List[GamesHowellResult]:
        """
        Performs Games-Howell post-hoc test for all pairwise comparisons.
        
        Games-Howell is appropriate when:
        - Variances are unequal (heteroscedasticity)
        - Sample sizes are unequal
        
        Args:
            groups: Dictionary with {group_name: values_list}
            
        Returns:
            List of GamesHowellResult for each pairwise comparison
        """
        results: List[GamesHowellResult] = []
        group_names = list(groups.keys())
        n_groups = len(group_names)
        
        # Calculate statistics for each group
        group_stats: Dict[str, Dict[str, float]] = {}
        for name, values in groups.items():
            data = np.array(values)
            group_stats[name] = {
                'mean': float(np.mean(data)),
                'var': float(np.var(data, ddof=1)),
                'n': float(len(data))
            }
        
        # Pairwise comparisons
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                name1, name2 = group_names[i], group_names[j]
                stats1, stats2 = group_stats[name1], group_stats[name2]
                
                # Mean difference
                mean_diff = stats1['mean'] - stats2['mean']
                
                # Standard error (Games-Howell uses pooled variance)
                se = float(np.sqrt(stats1['var'] / stats1['n'] + stats2['var'] / stats2['n']))
                
                # t-statistic
                t_stat = mean_diff / se if se > 0 else 0.0
                
                # Welch-Satterthwaite degrees of freedom
                num = (stats1['var'] / stats1['n'] + stats2['var'] / stats2['n']) ** 2
                denom = (
                    (stats1['var'] / stats1['n']) ** 2 / (stats1['n'] - 1) +
                    (stats2['var'] / stats2['n']) ** 2 / (stats2['n'] - 1)
                )
                df = num / denom if denom > 0 else 1.0
                
                # P-value (two-tailed)
                p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
                
                # Confidence interval
                # Use Tukey's studentized range distribution approximation
                # For Games-Howell, we use q* from studentized range
                # Simplified: use t-distribution with Bonferroni-like correction
                n_comparisons = n_groups * (n_groups - 1) / 2
                alpha_adjusted = self.alpha / n_comparisons
                t_crit = float(stats.t.ppf(1 - alpha_adjusted / 2, df))
                
                ci_lower = mean_diff - t_crit * se
                ci_upper = mean_diff + t_crit * se
                
                results.append(GamesHowellResult(
                    group1=name1,
                    group2=name2,
                    mean_diff=mean_diff,
                    std_error=se,
                    t_statistic=t_stat,
                    df=df,
                    p_value=p_value,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    is_significant=p_value < self.alpha
                ))
        
        return results


class IFSSAnalyzer:
    """
    Complete IFSS analysis with ANOVA, Bootstrap, and Games-Howell.
    """
    
    def __init__(
        self, 
        alpha: float = 0.05, 
        confidence_level: float = 0.95,
        random_seed: int = 42
    ):
        """
        Initialize IFSS analyzer.
        
        Args:
            alpha: Significance level for ANOVA
            confidence_level: Confidence level for bootstrap intervals
            random_seed: Random seed for reproducibility
        """
        self.bootstrap_analyzer = BootstrapAnalyzer(confidence_level, random_seed)
        self.anova_analyzer = ANOVAAnalyzer(alpha)
        self.alpha = alpha
        self.confidence_level = confidence_level
    
    def analyze(
        self, 
        ifss_data: Dict[str, List[float]],
        success_rates: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Performs complete IFSS analysis.
        
        Args:
            ifss_data: Dictionary with {series_name: ifss_values_list}
            success_rates: Optional dictionary with {series_name: success_rate}
            
        Returns:
            Dictionary with all analysis results
        """
        if success_rates is None:
            success_rates = {name: 1.0 for name in ifss_data.keys()}
        
        # Bootstrap analysis for each group
        bootstrap_results: Dict[str, BootstrapResult] = {}
        for name, values in ifss_data.items():
            rate = success_rates.get(name, 1.0)
            bootstrap_results[name] = self.bootstrap_analyzer.analyze_group(
                name, values, rate
            )
        
        # ANOVA
        anova_result = self.anova_analyzer.one_way_anova(ifss_data)
        
        # Games-Howell post-hoc (only if ANOVA is significant)
        games_howell_results: Optional[List[GamesHowellResult]] = None
        if anova_result.is_significant:
            games_howell_results = self.anova_analyzer.games_howell(ifss_data)
        
        return {
            'bootstrap': bootstrap_results,
            'anova': anova_result,
            'games_howell': games_howell_results,
            'groups': list(ifss_data.keys()),
            'n_groups': len(ifss_data),
            'alpha': self.alpha,
            'confidence_level': self.confidence_level
        }
    
    def format_results_text(self, results: Dict) -> str:
        """
        Formats analysis results as readable text.
        
        Args:
            results: Dictionary from analyze() method
            
        Returns:
            Formatted text string
        """
        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("IFSS ANALYSIS RESULTS - ANOVA with Bootstrap")
        lines.append("=" * 70)
        
        # Bootstrap results per group
        lines.append("\n--- Bootstrap Results (BCa) ---")
        lines.append(f"Confidence Level: {self.confidence_level * 100:.0f}%\n")
        
        for name, br in results['bootstrap'].items():
            lines.append(f"  {name}:")
            lines.append(f"    n = {br.n_samples}")
            lines.append(f"    Mean IFSS = {br.original_mean:.2f} MPa")
            lines.append(f"    95% CI = [{br.ci_lower:.2f}, {br.ci_upper:.2f}] MPa")
            lines.append(f"    Bootstrap SE = {br.std_error:.3f}")
            lines.append(f"    Iterations = {br.n_iterations}")
            lines.append("")
        
        # ANOVA results
        anova = results['anova']
        lines.append("\n--- One-Way ANOVA ---")
        lines.append(f"  F({anova.df_between}, {anova.df_within}) = {anova.f_statistic:.3f}")
        lines.append(f"  p-value = {anova.p_value:.4f}")
        lines.append(f"  α = {anova.alpha}")
        
        if anova.is_significant:
            lines.append(f"  Result: SIGNIFICANT (p < {anova.alpha})")
            lines.append("  → At least one group mean differs significantly")
        else:
            lines.append(f"  Result: NOT SIGNIFICANT (p ≥ {anova.alpha})")
            lines.append("  → No significant differences between group means")
        
        # Games-Howell results
        if results['games_howell']:
            lines.append("\n--- Games-Howell Post-Hoc Test ---")
            lines.append("  Pairwise Comparisons:\n")
            
            for gh in results['games_howell']:
                sig_marker = "*" if gh.is_significant else ""
                lines.append(f"  {gh.group1} vs {gh.group2}:{sig_marker}")
                lines.append(f"    Mean Diff = {gh.mean_diff:.2f} MPa")
                lines.append(f"    95% CI = [{gh.ci_lower:.2f}, {gh.ci_upper:.2f}]")
                lines.append(f"    t = {gh.t_statistic:.3f}, df = {gh.df:.1f}")
                lines.append(f"    p = {gh.p_value:.4f}")
                lines.append("")
            
            lines.append("  * = significant at α = 0.05")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def save_results_to_file(self, results: Dict, output_path: Path) -> None:
        """
        Saves analysis results to a text file.
        
        Args:
            results: Dictionary from analyze() method
            output_path: Path to save the text file
        """
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        text = self.format_results_text(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Saved ANOVA results to: {output_path}")