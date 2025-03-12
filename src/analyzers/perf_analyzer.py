"""
Performance Analyzer Module

This module analyzes performance test results and code changes to identify
performance regressions and their likely causes.
"""

import logging
import statistics
import numpy as np
from scipy import stats
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

class PerformanceAnalyzer:
    """
    Analyzes performance metrics and code changes to identify performance regressions
    and their root causes.
    """
    
    def __init__(self, git_collector, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance analyzer.
        
        Args:
            git_collector: An instance of GitCollector for accessing repository data
            config: Configuration dictionary containing performance thresholds and settings
        """
        self.git_collector = git_collector
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        self.thresholds = {
            'execution_time': 0.05,  # 5% slower
            'memory_usage': 0.10,    # 10% more memory
            'cpu_usage': 0.08,       # 8% more CPU
            'response_time': 0.05    # 5% slower response
        }
        
        if config and 'thresholds' in config:
            self.thresholds.update(config['thresholds'])

        self.regression_threshold = config.get('regression_threshold', 0.05)
        self.significance_level = config.get('significance_level', 0.05)
        self.min_samples = config.get('min_samples', 3)
        
        self.perf_patterns = [
            (r'for\s*\([^)]+\)', 'Loop construct'),
            (r'while\s*\([^)]+\)', 'Loop construct'),
            (r'synchronized', 'Synchronization'),
            (r'Lock\s*\([^)]*\)', 'Locking mechanism'),
            (r'new\s+Thread', 'Thread creation'),
            (r'new\s+.*\[\s*\d+\s*\]', 'Array allocation'),
            (r'Collections\.sort', 'Sorting operation'),
            (r'Stream\.collect', 'Stream operation'),
            (r'new\s+HashMap\s*\(', 'Map creation'),
            (r'new\s+ArrayList\s*\(', 'List creation'),
            (r'select\s+.*\s+from', 'Database query'),
            (r'join\s+.*\s+on', 'Database join'),
            (r'group\s+by', 'Data aggregation'),
            (r'order\s+by', 'Data sorting'),
            (r'@Cacheable', 'Caching'),
            (r'@Transactional', 'Database transaction')
        ]
        
        self.logger.info("Performance analyzer initialized")

    def detect_regression(self, baseline_metrics: List[float], 
                         current_metrics: List[float]) -> Tuple[bool, float, Dict]:
        """
        Detect if there's a statistically significant performance regression.
        
        Args:
            baseline_metrics: List of performance measurements from baseline
            current_metrics: List of performance measurements from current version
            
        Returns:
            Tuple of (regression_detected, percentage_change, statistics)
        """
        if len(baseline_metrics) < self.min_samples or len(current_metrics) < self.min_samples:
            self.logger.warning("Insufficient samples for regression analysis")
            return False, 0.0, {}
            
        try:
            baseline_mean = np.mean(baseline_metrics)
            current_mean = np.mean(current_metrics)
            
            percent_change = (current_mean - baseline_mean) / baseline_mean
            
            t_stat, p_value = stats.ttest_ind(baseline_metrics, current_metrics)
            
            baseline_ci = stats.t.interval(0.95, len(baseline_metrics)-1,
                                         loc=np.mean(baseline_metrics),
                                         scale=stats.sem(baseline_metrics))
            current_ci = stats.t.interval(0.95, len(current_metrics)-1,
                                        loc=np.mean(current_metrics),
                                        scale=stats.sem(current_metrics))
            
            stats_dict = {
                'baseline_mean': baseline_mean,
                'current_mean': current_mean,
                'baseline_std': np.std(baseline_metrics),
                'current_std': np.std(current_metrics),
                'p_value': p_value,
                't_statistic': t_stat,
                'baseline_ci': baseline_ci,
                'current_ci': current_ci
            }
            
            is_regression = (percent_change > self.regression_threshold and 
                           p_value < self.significance_level)
            
            if is_regression:
                self.logger.warning(
                    f"Performance regression detected: {percent_change:.2%} change "
                    f"(p={p_value:.4f})"
                )
            
            return is_regression, percent_change, stats_dict
            
        except Exception as e:
            self.logger.error(f"Error in regression detection: {str(e)}")
            return False, 0.0, {}

    def analyze_code_changes(self, commit_hash: str) -> Dict[str, Any]:
        """
        Analyze code changes in a commit for performance-sensitive patterns.
        
        Args:
            commit_hash: Hash of the commit to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            results = {
                'patterns_found': [],
                'high_risk_files': [],
                'risk_score': 0.0,
                'details': {}
            }
            
            commit_info = self.git_collector.get_commit_info(commit_hash)
            modified_files = self.git_collector.get_files_modified_in_commit(commit_hash)
            
            total_patterns = 0
            for file_path in modified_files:
                if not self._is_source_file(file_path):
                    continue
                
                if commit_info.get('parent_hashes'):
                    parent_hash = commit_info['parent_hashes'][0]
                    diff = self.git_collector.get_diff_between_commits(
                        parent_hash, commit_hash, file_path
                    )
                    
                    file_results = self._analyze_diff_patterns(diff)
                    
                    if file_results['pattern_count'] > 0:
                        results['details'][file_path] = file_results
                        total_patterns += file_results['pattern_count']
                        
                        if file_results['risk_score'] > 0.7:
                            results['high_risk_files'].append({
                                'file': file_path,
                                'score': file_results['risk_score'],
                                'patterns': file_results['patterns']
                            })
            
            results['risk_score'] = self._calculate_risk_score(
                total_patterns, 
                len(results['high_risk_files']),
                len(modified_files)
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing code changes: {str(e)}")
            return {'error': str(e)}

    def _analyze_diff_patterns(self, diff: str) -> Dict[str, Any]:
        """
        Analyze a diff for performance-sensitive patterns.
        
        Args:
            diff: The diff content to analyze
            
        Returns:
            Dictionary containing analysis results for the diff
        """
        results = {
            'pattern_count': 0,
            'patterns': [],
            'risk_score': 0.0
        }
        
        added_lines = [line[1:] for line in diff.split('\n') if line.startswith('+')]
        
        for pattern, description in self.perf_patterns:
            for line in added_lines:
                if re.search(pattern, line):
                    results['pattern_count'] += 1
                    if description not in results['patterns']:
                        results['patterns'].append(description)
        
        if results['pattern_count'] > 0:
            results['risk_score'] = min(1.0, results['pattern_count'] * 0.2)
        
        return results

    def _calculate_risk_score(self, total_patterns: int, high_risk_files: int, 
                            total_files: int) -> float:
        """
        Calculate overall risk score for a commit.
        
        Args:
            total_patterns: Total number of performance patterns found
            high_risk_files: Number of high-risk files
            total_files: Total number of files changed
            
        Returns:
            Risk score between 0.0 and 1.0
        """
        if total_files == 0:
            return 0.0
            
        pattern_score = min(1.0, total_patterns * 0.1)
        file_score = high_risk_files / total_files
        
        return (pattern_score * 0.7 + file_score * 0.3)

    def _is_source_file(self, file_path: str) -> bool:
        """
        Check if a file is a source code file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is a source file, False otherwise
        """
        source_extensions = {'.java', '.cpp', '.c', '.h', '.py', '.js', '.go'}
        return any(file_path.endswith(ext) for ext in source_extensions)

    def generate_report(self, commit_hash: str, perf_analysis: Dict[str, Any], 
                       code_analysis: Dict[str, Any]) -> str:
        """
        Generate a detailed performance analysis report.
        
        Args:
            commit_hash: The analyzed commit hash
            perf_analysis: Results from performance regression analysis
            code_analysis: Results from code change analysis
            
        Returns:
            Formatted report string
        """
        report = [
            "Performance Analysis Report",
            "=" * 25,
            f"Commit: {commit_hash}",
            ""
        ]
        
        if perf_analysis.get('is_regression'):
            report.extend([
                "⚠️ PERFORMANCE REGRESSION DETECTED",
                f"Change: {perf_analysis['percent_change']:.2%}",
                f"Statistical Significance: p={perf_analysis['stats']['p_value']:.4f}",
                f"Confidence Interval: {perf_analysis['stats']['current_ci']}",
                ""
            ])
        
        report.extend([
            "Code Analysis:",
            f"Risk Score: {code_analysis['risk_score']:.2f}",
            f"High-risk Files: {len(code_analysis['high_risk_files'])}",
            ""
        ])
        
        if code_analysis['high_risk_files']:
            report.append("High Risk Files:")
            for file_info in code_analysis['high_risk_files']:
                report.extend([
                    f"  • {file_info['file']}",
                    f"    Risk Score: {file_info['score']:.2f}",
                    f"    Patterns: {', '.join(file_info['patterns'])}",
                    ""
                ])
        
        return "\n".join(report)

    def analyze_performance_regression(self, good_results: List[float], bad_results: List[float], threshold: float = 0.05) -> Dict[str, Any]:
        """Analyze performance regression between good and bad results"""
        try:
            t_stat, p_value = stats.ttest_ind(good_results, bad_results)
            mean_diff = np.mean(bad_results) - np.mean(good_results)
            percent_change = (mean_diff / np.mean(good_results)) * 100
            
            return {
                'is_regression': p_value < 0.05 and percent_change > threshold * 100,
                'percent_change': percent_change,
                'p_value': p_value,
                'confidence': 1 - p_value
            }
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {str(e)}")
            return {}

    def analyze_performance(self, good_metrics: Dict[str, float], 
                          bad_metrics: Dict[str, float], 
                          threshold: float = 0.05) -> Dict[str, Any]:
        """Analyze performance difference between good and bad commits."""
        try:
            results = {
                "regression_found": False,
                "metrics_comparison": {},
                "regressions": [],
                "improvements": [],
                "charts": {},
                "summary": "",
                "timestamp": datetime.now().isoformat()
            }

            for metric in good_metrics:
                if metric not in bad_metrics:
                    continue

                good_value = float(good_metrics[metric])
                bad_value = float(bad_metrics[metric])
                
                if good_value == 0:
                    continue

                change_pct = (bad_value - good_value) / good_value * 100
                threshold_pct = self.thresholds.get(metric, threshold) * 100

                comparison = {
                    "metric": metric,
                    "good_value": good_value,
                    "bad_value": bad_value,
                    "change_percentage": change_pct,
                    "threshold": threshold_pct,
                    "is_regression": False
                }

                if metric in ['execution_time', 'memory_usage', 'cpu_usage', 'response_time']:
                    is_regression = change_pct >= threshold_pct
                else:
                    is_regression = change_pct <= -threshold_pct

                if is_regression:
                    results["regression_found"] = True
                    comparison["is_regression"] = True
                    results["regressions"].append(comparison)
                elif change_pct < 0:
                    results["improvements"].append(comparison)

                results["metrics_comparison"][metric] = comparison

                chart_data = self._generate_metric_chart(
                    metric, good_value, bad_value, threshold_pct
                )
                results["charts"][metric] = chart_data

            results["summary"] = self._generate_summary(results)
            
            return results

        except Exception as e:
            self.logger.error(f"Error in performance analysis: {str(e)}")
            raise

    def _generate_metric_chart(self, metric_name: str, good_value: float, 
                             bad_value: float, threshold: float) -> str:
        """Generate a base64 encoded chart image."""
        try:
            plt.figure(figsize=(8, 4))
            plt.clf()

            bars = plt.bar(['Good Build', 'Bad Build'], [good_value, bad_value])
            bars[0].set_color('green')
            bars[1].set_color('red' if bad_value > good_value else 'blue')

            plt.title(f'{metric_name} Comparison')
            plt.ylabel('Value')
            
            threshold_value = good_value * (1 + threshold/100)
            plt.axhline(y=threshold_value, color='r', linestyle='--', 
                       label=f'Threshold ({threshold:.1f}%)')
            plt.legend()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()
            
            return base64.b64encode(image_png).decode()

        except Exception as e:
            self.logger.error(f"Error generating chart: {str(e)}")
            return ""

    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable summary."""
        lines = []
        
        if results["regression_found"]:
            lines.append("⚠️ Performance regressions detected!")
            for reg in results["regressions"]:
                lines.append(
                    f"\n- {reg['metric']}: {reg['change_percentage']:.1f}% degradation "
                    f"(Threshold: {reg['threshold']:.1f}%)"
                )
        else:
            lines.append("✅ No significant performance regressions found.")

        if results["improvements"]:
            lines.append("\n🎉 Performance improvements:")
            for imp in results["improvements"]:
                lines.append(
                    f"\n- {imp['metric']}: {abs(imp['change_percentage']):.1f}% improvement"
                )

        return "\n".join(lines)

    def collect_performance_metrics(self, commit_hash: str) -> Dict[str, float]:
        """Collect performance metrics for a specific commit."""
        try:
            self.git_collector.checkout_commit(commit_hash)
            
            metrics = {
                'execution_time': self._measure_execution_time(),
                'memory_usage': self._measure_memory_usage(),
                'cpu_usage': self._measure_cpu_usage(),
                'response_time': self._measure_response_time()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics for commit {commit_hash}: {str(e)}")
            return {}

    def _measure_execution_time(self) -> float:
        """Measure execution time of test suite."""
        return np.random.uniform(0.8, 1.2)  # Random value between 0.8 and 1.2 seconds

    def _measure_memory_usage(self) -> float:
        """Measure memory usage."""
        return np.random.uniform(100, 200)  # Random value between 100-200 MB

    def _measure_cpu_usage(self) -> float:
        """Measure CPU usage."""
        return np.random.uniform(20, 80)  # Random value between 20-80%

    def _measure_response_time(self) -> float:
        """Measure response time."""
        return np.random.uniform(50, 150)  # Random value between 50-150ms