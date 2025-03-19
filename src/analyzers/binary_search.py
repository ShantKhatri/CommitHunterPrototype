"""
Binary Search Analyzer Module

This module implements an enhanced binary search approach to identify problematic commits,
particularly useful for performance regressions and test failures. It incorporates
risk-based prioritization to find problematic commits more efficiently, using techniques
inspired by Commit Guru.
"""

import os
import math
import numpy as np
from typing import Dict, Any, List, Callable, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from scipy import stats


class BinarySearchAnalyzer:
    """
    Enhanced binary search analyzer that identifies problematic commits using
    risk-based prioritization and statistical analysis.
    """
    
    def __init__(self, git_collector, config: Dict[str, Any] = None):
        """
        Initialize the binary search analyzer.
        
        Args:
            git_collector: The git collector instance to use for retrieving commit data
            config: Configuration options for the analyzer
        """
        self.git_collector = git_collector
        self.config = config or {}
        self.logger = self.config.get('logger', None)
        self.perf_logger = self.config.get('perf_logger', None)
        self.test_threshold = self.config.get('test_threshold', 5)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.75)
        
        self._commit_info_cache = {}
        self._risk_score_cache = {}
        self._author_stats = None
        self._subsystem_stats = None
        
        self.tests_run = 0
        self.high_risk_commits_tested = 0
        
        self._initialize_historical_data()
    
    def _initialize_historical_data(self):
        """Initialize historical data for risk analysis"""
        self.problematic_commits = []
        self.safe_commits = []
        
        self.metrics_by_category = {
            'problematic': {},
            'safe': {}
        }
        
        self._author_stats = defaultdict(lambda: {'commits': 0, 'bugs': 0})
        self._subsystem_stats = defaultdict(lambda: {'commits': 0, 'bugs': 0})
    
    def analyze_commit_range(self, good_commit: str, bad_commit: str, 
                             test_commit_func: Callable[[str], bool]) -> Dict[str, Any]:
        """
        Use enhanced binary search to identify problematic commits between good and bad commits.
        
        Args:
            good_commit: The known good commit (tests pass)
            bad_commit: The known bad commit (tests fail)
            test_commit_func: Function to test if a commit passes or fails
            
        Returns:
            Dict with analysis results, including identified problematic commits and confidence scores
        """
        self._log(f"Starting binary search analysis between {good_commit} and {bad_commit}")
        
        self.tests_run = 0
        self.high_risk_commits_tested = 0
        
        start_time = datetime.now()
        commit_range = self.git_collector.get_commits_between(good_commit, bad_commit)
        if not commit_range:
            return {'success': False, 'error': 'No commits found between good and bad', 'results': []}
        
        self._log(f"Found {len(commit_range)} commits to analyze")
        
        self._precache_commit_info(commit_range)
        
        lower = -1 
        upper = len(commit_range)
        
        self._log(f"Verifying good commit {good_commit}")
        if not test_commit_func(good_commit):
            return {'success': False, 'error': 'Good commit does not pass tests', 'results': []}
        
        self._log(f"Verifying bad commit {bad_commit}")
        if test_commit_func(bad_commit):
            return {'success': False, 'error': 'Bad commit does not fail tests', 'results': []}
        
        self.tests_run += 2 
        
        tested_commits = {
            good_commit: {'result': True, 'commit': good_commit},
            bad_commit: {'result': False, 'commit': bad_commit}
        }
        
        while upper - lower > 1:
            risk_scores = []
            for i, commit in enumerate(commit_range):
                if commit['hash'] not in tested_commits:
                    risk_score = self._calculate_enhanced_risk_score(commit)
                    risk_scores.append((i, risk_score, commit['hash']))
            
            if risk_scores:
                risk_scores.sort(key=lambda x: x[1], reverse=True)
                
                for idx, score, commit_hash in risk_scores:
                    if lower < idx < upper:
                        mid = idx
                        self._log(f"Testing high-risk commit {commit_hash} with score {score:.3f}")
                        self.high_risk_commits_tested += 1
                        break
                else:
                    mid = (lower + upper) // 2
            else:
                mid = (lower + upper) // 2
            
            commit_to_test = commit_range[mid]['hash']
            self._log(f"Testing commit {commit_to_test} ({mid+1}/{len(commit_range)})")
            result = test_commit_func(commit_to_test)
            self.tests_run += 1
            
            tested_commits[commit_to_test] = {'result': result, 'commit': commit_to_test}
            
            if result:
                lower = mid
            else:
                upper = mid
            
            commit_info = self._get_commit_info(commit_to_test)
            if result:
                self.safe_commits.append(commit_info)
            else:
                self.problematic_commits.append(commit_info)
                
            if self.tests_run >= self.test_threshold:
                self._log(f"Reached test threshold of {self.test_threshold}")
                break
        
        probable_commit = None
        if upper - lower <= 1:
            problematic_idx = upper
            if 0 <= problematic_idx < len(commit_range):
                probable_commit = commit_range[problematic_idx]['hash']
                self._log(f"Identified problematic commit: {probable_commit}")
        
        significant_metrics = {}
        if len(self.problematic_commits) >= 2 and len(self.safe_commits) >= 2:
            significant_metrics = self._calculate_statistical_significance()
        
        results = []
        if probable_commit:
            commit_info = self._get_commit_info(probable_commit)
            confidence = self._calculate_confidence(commit_info, significant_metrics)
            
            results.append({
                'commit': probable_commit,
                'confidence': confidence,
                'category': 'Likely Problematic',
                'reason': self._generate_reason(commit_info, significant_metrics),
                'test_result': False
            })
            
            related_commits = self._find_related_commits(probable_commit, commit_range)
            for related_commit in related_commits:
                commit_info = self._get_commit_info(related_commit)
                confidence = self._calculate_confidence(commit_info, significant_metrics) * 0.8  # Reduced confidence
                
                results.append({
                    'commit': related_commit,
                    'confidence': confidence,
                    'category': 'Potentially Related',
                    'reason': "Related to primary problematic commit",
                    'test_result': None  # Not directly tested
                })
        
        duration = datetime.now() - start_time
        efficiency = 0
        if len(commit_range) > 0:
            efficiency = (len(commit_range) - (upper - lower)) / self.tests_run if self.tests_run > 0 else 0
        
        metrics = {
            'total_commits': len(commit_range),
            'tests_run': self.tests_run,
            'high_risk_tested': self.high_risk_commits_tested,
            'duration': duration.total_seconds(),
            'algorithm': 'risk_binary_search',
            'efficiency': efficiency,
            'statistical_significance': significant_metrics
        }
        
        return {'success': True, 'results': results, 'metrics': metrics}
    
    def _precache_commit_info(self, commit_range: List[Dict[str, Any]]) -> None:
        """Pre-cache commit info for all commits in the range"""
        for commit in commit_range:
            if commit['hash'] not in self._commit_info_cache:
                info = self.git_collector.get_commit_info(commit['hash'])
                self._commit_info_cache[commit['hash']] = info
    
    def _get_commit_info(self, commit_hash: str) -> Dict[str, Any]:
        """Get or fetch commit info"""
        if commit_hash not in self._commit_info_cache:
            self._commit_info_cache[commit_hash] = self.git_collector.get_commit_info(commit_hash)
        return self._commit_info_cache[commit_hash]
    
    def _calculate_enhanced_risk_score(self, commit: Dict[str, Any]) -> float:
        """
        Calculate a risk score for a commit using advanced metrics from Commit Guru.
        
        The risk score is based on:
        1. Commit size (lines added/deleted, files changed)
        2. Commit diffusion (subsystems, directories affected)
        3. Developer experience
        4. Code complexity (entropy)
        5. File type risk
        
        Returns:
            Float between 0 and 1 representing risk score
        """
        commit_hash = commit['hash']
        if commit_hash in self._risk_score_cache:
            return self._risk_score_cache[commit_hash]
        
        commit_info = commit
        if 'files_changed' not in commit_info:
            commit_info = self._get_commit_info(commit_hash)
        
        score = 0.0
        
        # 1. Size factors - 35% of score
        lines_added = commit_info.get('lines_added', 0)
        lines_deleted = commit_info.get('lines_deleted', 0)
        files_changed = len(commit_info.get('files_changed', []))
        
        normalized_lines_added = min(1.0, lines_added / 500)  # Cap at 500 lines
        normalized_lines_deleted = min(1.0, lines_deleted / 200)  # Cap at 200 lines
        normalized_files_changed = min(1.0, files_changed / 10)  # Cap at 10 files
        
        size_score = (0.5 * normalized_lines_added + 
                     0.3 * normalized_lines_deleted + 
                     0.2 * normalized_files_changed)
        
        score += 0.35 * size_score
        
        # 2. Diffusion factors - 25% of score
        files = commit_info.get('files_changed', [])
        if files:
            subsystems = self._get_subsystems(files)
            directories = self._get_directories(files)
            
            normalized_subsystems = min(1.0, len(subsystems) / 3)  # Cap at 3 subsystems
            normalized_directories = min(1.0, len(directories) / 5)  # Cap at 5 directories
            
            diffusion_score = 0.6 * normalized_subsystems + 0.4 * normalized_directories
            score += 0.25 * diffusion_score
        
        # 3. Developer experience - 15% of score
        author = commit_info.get('author_name', commit_info.get('author', ''))
        experience_score = self._calculate_author_experience(author)
        score += 0.15 * (1.0 - experience_score)  # Inverse of experience (less exp = higher risk)
        
        # 4. Code complexity (entropy) - 15% of score
        entropy_score = self._calculate_entropy(commit_info)
        score += 0.15 * entropy_score
        
        # 5. File type risk - 10% of score
        file_type_risk = self._calculate_file_type_risk(files)
        score += 0.10 * file_type_risk
        
        # Cache and return the final risk score (ensure it's between 0 and 1)
        final_score = min(1.0, max(0.0, score))
        self._risk_score_cache[commit_hash] = final_score
        return final_score
    
    def _calculate_risk_score(self, commit: Dict[str, Any]) -> float:
        """
        Original risk score calculation (simpler version).
        Kept for backward compatibility.
        """
        commit_info = commit
        if 'files_changed' not in commit_info:
            commit_info = self._get_commit_info(commit['hash'])
        
        lines_changed = commit_info.get('lines_added', 0) + commit_info.get('lines_deleted', 0)
        files_changed = len(commit_info.get('files_changed', []))
        
        normalized_lines = min(1.0, lines_changed / 1000)  # Cap at 1000 lines
        normalized_files = min(1.0, files_changed / 10)  # Cap at 10 files
        
        score = 0.7 * normalized_lines + 0.3 * normalized_files
        return min(1.0, score)
    
    def _get_subsystems(self, files: List[str]) -> Set[str]:
        """Extract subsystems from files (top-level directories)"""
        subsystems = set()
        for file_path in files:
            if '/' in file_path:
                subsystems.add(file_path.split('/')[0])
            elif '\\' in file_path:
                subsystems.add(file_path.split('\\')[0])
            else:
                subsystems.add('root')
        return subsystems
    
    def _get_directories(self, files: List[str]) -> Set[str]:
        """Extract all directories from file paths"""
        directories = set()
        for file_path in files:
            directory = os.path.dirname(file_path)
            if directory:
                directories.add(directory)
        return directories
    
    def _calculate_author_experience(self, author: str) -> float:
        """
        Calculate developer experience score based on:
        1. Number of commits by the developer
        2. Historical bug introduction rate
        
        Returns a value between 0 (inexperienced) and 1 (experienced)
        """
        if not self._author_stats or author not in self._author_stats:
            return 0.5
        
        stats = self._author_stats[author]
        total_commits = stats['commits']
        
        if total_commits == 0:
            return 0.0
        
        # Log scale for commit count: log10(1)=0, log10(10)=1, log10(100)=2, etc.
        commit_experience = min(1.0, math.log10(total_commits + 1) / 2.0)
        
        bug_rate = 0.0
        if total_commits > 0:
            bug_rate = stats['bugs'] / total_commits
        
        reliability = 1.0 - min(1.0, bug_rate * 2)
        
        if commit_experience < 0.3:
            return commit_experience
        else:
            return 0.4 * commit_experience + 0.6 * reliability
    
    def _calculate_entropy(self, commit_info: Dict[str, Any]) -> float:
        """
        Calculate code change entropy (complexity) based on distribution of changes.
        Higher entropy means changes are spread across files more evenly.
        
        Returns a value between 0 (low entropy) and 1 (high entropy)
        """
        files = commit_info.get('files_changed', [])
        if not files:
            return 0.0
        
        lines_per_file = {}
        for file_path in files:
            file_stats = commit_info.get('file_stats', {}).get(file_path, {})
            lines_changed = file_stats.get('lines_added', 0) + file_stats.get('lines_deleted', 0)
            lines_per_file[file_path] = max(1, lines_changed)  # Ensure at least 1 line per file
        
        total_lines = sum(lines_per_file.values())
        if total_lines == 0:
            return 0.0
        
        entropy = 0.0
        for file_path in lines_per_file:
            p = lines_per_file[file_path] / total_lines
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Normalize to 0-1 scale
        # Maximum entropy is log2(n) where n is number of files
        max_possible_entropy = math.log2(len(files)) if len(files) > 0 else 1
        normalized_entropy = entropy / max_possible_entropy if max_possible_entropy > 0 else 0
        
        return normalized_entropy
    
    def _calculate_file_type_risk(self, files: List[str]) -> float:
        """
        Calculate risk based on file types being modified.
        Different file types have different risk profiles.
        
        Returns a value between 0 (low risk) and 1 (high risk)
        """
        if not files:
            return 0.0
        
        risk_by_extension = {
            # Core source files - higher risk
            '.java': 0.8,
            '.c': 0.9,
            '.cpp': 0.9,
            '.h': 0.85,
            '.py': 0.75,
            '.js': 0.7,
            
            # Build/configuration files - medium-high risk
            '.xml': 0.65,
            '.gradle': 0.7,
            '.pom': 0.7,
            '.mk': 0.75,
            'makefile': 0.75,
            '.cmake': 0.7,
            
            # Test files - medium risk
            'test.java': 0.5,
            'test.py': 0.5,
            'test.js': 0.5,
            
            # Documentation - low risk
            '.md': 0.1,
            '.txt': 0.1,
            '.rst': 0.1,
            
            # Default for unknown extensions
            '': 0.5
        }
        
        total_risk = 0.0
        for file_path in files:
            # Check for test files first
            if 'test' in file_path.lower():
                if file_path.endswith('.java'):
                    total_risk += risk_by_extension['test.java']
                elif file_path.endswith('.py'):
                    total_risk += risk_by_extension['test.py']
                elif file_path.endswith('.js'):
                    total_risk += risk_by_extension['test.js']
                else:
                    total_risk += 0.4  # Default risk for test files
                continue
            
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()
            
            risk = risk_by_extension.get(ext, risk_by_extension[''])
            total_risk += risk
        
        return total_risk / len(files) if files else 0.0
    
    def _calculate_statistical_significance(self) -> Dict[str, Any]:
        """
        Calculate statistical significance of differences between metrics
        for problematic and safe commits.
        
        Returns a dictionary with metrics and their significance.
        """
        if len(self.problematic_commits) < 2 or len(self.safe_commits) < 2:
            return {}
        
        metrics_to_check = [
            'lines_added', 
            'lines_deleted', 
            'lines_changed',
            'files_changed'
        ]
        
        results = {}
        for metric in metrics_to_check:
            problematic_values = []
            for commit in self.problematic_commits:
                if metric == 'lines_changed':
                    value = commit.get('lines_added', 0) + commit.get('lines_deleted', 0)
                else:
                    value = commit.get(metric, 0)
                problematic_values.append(value)
            
            safe_values = []
            for commit in self.safe_commits:
                if metric == 'lines_changed':
                    value = commit.get('lines_added', 0) + commit.get('lines_deleted', 0)
                else:
                    value = commit.get(metric, 0)
                safe_values.append(value)
            
            try:
                u_stat, p_value = stats.mannwhitneyu(problematic_values, safe_values, alternative='two-sided')
                
                mean_diff = abs(np.mean(problematic_values) - np.mean(safe_values))
                pooled_std = np.sqrt((np.std(problematic_values)**2 + np.std(safe_values)**2) / 2)
                effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
                
                results[metric] = {
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': effect_size,
                    'problematic_mean': np.mean(problematic_values),
                    'safe_mean': np.mean(safe_values),
                    'problematic_median': np.median(problematic_values),
                    'safe_median': np.median(safe_values)
                }
                
                if effect_size >= 0.8:
                    results[metric]['effect_description'] = 'Large'
                elif effect_size >= 0.5:
                    results[metric]['effect_description'] = 'Medium'
                elif effect_size >= 0.2:
                    results[metric]['effect_description'] = 'Small'
                else:
                    results[metric]['effect_description'] = 'Negligible'
                    
            except Exception as e:
                results[metric] = {
                    'error': str(e),
                    'significant': False
                }
                
        return results
    
    def _calculate_confidence(self, commit_info: Dict[str, Any], 
                             significant_metrics: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the identification of a problematic commit
        based on risk factors and statistical significance.
        
        Args:
            commit_info: Information about the commit
            significant_metrics: Dictionary of statistically significant metrics
            
        Returns:
            Float between 0 and 1 representing confidence level
        """
        # Base confidence
        confidence = 0.7  # Start with 70% base confidence
        
        risk_score = self._calculate_enhanced_risk_score(commit_info)
        confidence += 0.15 * risk_score
        
        if significant_metrics:
            significant_count = sum(1 for m in significant_metrics.values() if m.get('significant', False))
            
            if significant_count >= 3:
                confidence += 0.15  # Strong statistical evidence
            elif significant_count >= 1:
                confidence += 0.10  # Some statistical evidence
                
            matching_metrics = 0
            total_checked = 0
            
            for metric, stats in significant_metrics.items():
                if not stats.get('significant', False):
                    continue
                    
                total_checked += 1
                
                if metric == 'lines_changed':
                    commit_value = commit_info.get('lines_added', 0) + commit_info.get('lines_deleted', 0)
                else:
                    commit_value = commit_info.get(metric, 0)
                
                problematic_median = stats.get('problematic_median', 0)
                safe_median = stats.get('safe_median', 0)
                
                if abs(commit_value - problematic_median) < abs(commit_value - safe_median):
                    matching_metrics += 1
            
            if total_checked > 0:
                match_ratio = matching_metrics / total_checked
                confidence += 0.10 * match_ratio  # Up to 10% more confidence
        
        return min(1.0, max(0.1, confidence))
    
    def _generate_reason(self, commit_info: Dict[str, Any], 
                        significant_metrics: Dict[str, Any]) -> str:
        """Generate a human-readable reason for why a commit is considered problematic"""
        reasons = []
        
        # Basic commit info
        author = commit_info.get('author_name', commit_info.get('author', 'Unknown Author'))
        message = commit_info.get('message', '')
        summary = message.split('\n')[0] if message else 'No commit message'
        
        # Get key metrics
        lines_added = commit_info.get('lines_added', 0)
        lines_deleted = commit_info.get('lines_deleted', 0)
        files_changed = len(commit_info.get('files_changed', []))
        
        reasons.append(f"Commit by {author}: \"{summary}\"")
        
        # Add size-based reasons
        if lines_added > 100 or lines_deleted > 50:
            reasons.append(f"Large commit (+{lines_added}/-{lines_deleted} lines across {files_changed} files)")
        
        # Add risk-based reasons
        risk_score = self._calculate_enhanced_risk_score(commit_info)
        risk_level = "high" if risk_score > 0.7 else "medium" if risk_score > 0.4 else "low"
        reasons.append(f"Risk assessment: {risk_level} risk score ({risk_score:.2f})")
        
        # Add diffusion-based reasons
        files = commit_info.get('files_changed', [])
        if files:
            subsystems = self._get_subsystems(files)
            if len(subsystems) > 1:
                reasons.append(f"Changes span {len(subsystems)} different subsystems")
        
        # Add statistical reasons if available
        if significant_metrics:
            for metric, stats in significant_metrics.items():
                if stats.get('significant', False):
                    effect = stats.get('effect_description', 'notable')
                    
                    readable_metric = metric.replace('_', ' ')
                    
                    if metric == 'lines_changed':
                        commit_value = lines_added + lines_deleted
                    else:
                        commit_value = commit_info.get(metric, 0)
                    
                    problematic_median = stats.get('problematic_median', 0)
                    safe_median = stats.get('safe_median', 0)
                    
                    # Check if closer to problematic median
                    if abs(commit_value - problematic_median) < abs(commit_value - safe_median):
                        reasons.append(f"{readable_metric.title()} ({commit_value}) matches pattern of problematic commits")
        
        # Combine all reasons
        if len(reasons) == 1:
            reasons.append("Identified by binary search algorithm")
            
        return ". ".join(reasons)


    def _find_related_commits(self, primary_commit: str, commit_range: List[Dict[str, Any]]) -> List[str]:
        """
        Find commits that might be related to the primary problematic commit.
        This checks:
        1. Commits that modify the same files
        2. Commits by the same author
        3. Commits with similar messages
        
        Returns:
            List of commit hashes that are potentially related
        """
        primary_info = self._get_commit_info(primary_commit)
        primary_author = primary_info.get('author_name', primary_info.get('author', ''))
        primary_files = set(primary_info.get('files_changed', []))
        primary_message = primary_info.get('message', '').lower()
        
        primary_keywords = set(word.lower() for word in primary_message.split() 
                            if len(word) > 3 and word not in {'the', 'and', 'for', 'with'})
        
        related_commits = []
        related_scores = {}
        
        for commit in commit_range:
            commit_hash = commit['hash']
            
            if commit_hash == primary_commit:
                continue
            
            commit_info = self._get_commit_info(commit_hash)
            commit_author = commit_info.get('author_name', commit_info.get('author', ''))
            commit_files = set(commit_info.get('files_changed', []))
            commit_message = commit_info.get('message', '').lower()
            
            commit_keywords = set(word.lower() for word in commit_message.split()
                                if len(word) > 3 and word not in {'the', 'and', 'for', 'with'})
            
            score = 0.0
            
            if commit_author == primary_author:
                score += 0.3
                
            if primary_files and commit_files:
                overlap_files = primary_files.intersection(commit_files)
                if overlap_files:
                    file_score = len(overlap_files) / max(len(primary_files), len(commit_files))
                    score += 0.4 * file_score
                    
            if primary_keywords and commit_keywords:
                keyword_overlap = primary_keywords.intersection(commit_keywords)
                if keyword_overlap:
                    keyword_score = len(keyword_overlap) / max(len(primary_keywords), len(commit_keywords))
                    score += 0.3 * keyword_score
            
            if score > 0.3:  # Threshold for relatedness
                related_scores[commit_hash] = score
        
        sorted_related = sorted(related_scores.items(), key=lambda x: x[1], reverse=True)
        return [commit_hash for commit_hash, _ in sorted_related[:3]]