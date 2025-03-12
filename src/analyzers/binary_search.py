"""
Binary Search Analyzer Module

This module implements an enhanced binary search approach to identify problematic commits,
particularly useful for performance regressions and test failures. It incorporates
risk-based prioritization to find problematic commits more efficiently.
"""

from typing import Dict, Any, List, Callable, Optional, Tuple
import logging
import time
import re
import json
import os
import pickle
import hashlib
from datetime import datetime
import numpy as np
from git import Repo
from collections import defaultdict
import ast
import functools

class BinarySearchAnalyzer:
    def __init__(self, git_collector, config: Dict[str, Any] = None):
        self.git_collector = git_collector
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 5)
        self.history = []
        
        self.cache_dir = self.config.get('cache_dir', '.commithunter_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.test_results_cache = self._load_cache('test_results.pkl')
        self.risk_scores_cache = self._load_cache('risk_scores.pkl')
        
        self.contributor_metrics = self._load_cache('contributor_metrics.pkl', defaultdict(lambda: {
            'commit_count': 0,
            'failure_rate': 0.0,
            'avg_risk_score': 0.0,
            'first_commit_date': None
        }))
        
        self.risk_patterns = [
            r"fix(?:ed|es|ing)?\s+bug",
            r"workaround",
            r"hack",
            r"temporary\s+fix",
            r"FIXME",
            r"TODO",
            r"bypass",
            r"refactor",
            r"performance\s+impact",
            r"race\s+condition",
            r"deadlock",
            r"memory\s+leak",
            r"overflow",
            r"underflow",
            r"null\s+pointer",
            r"exception\s+handling",
            r"regression",
            r"issue\s+\d+",
            r"bug\s+\d+",
            r"crash",
            r"hang",
            r"timeout",
            r"fail(s|ed|ure)?"
        ]
        
        self.api_change_patterns = [
            r"api\s+change",
            r"interface\s+change",
            r"breaking\s+change",
            r"signature\s+change",
            r"method\s+signature",
            r"deprecat(e|ed|ion)",
            r"remov(e|ed|ing)\s+(method|function|api|parameter)",
            r"chang(e|ed|ing)\s+(method|function|api|parameter)",
            r"renam(e|ed|ing)"
        ]
        
        self.threading_patterns = [
            r"thread",
            r"concurren(t|cy)",
            r"sync(hroniz|hronous|hronization)",
            r"atomic",
            r"mutex",
            r"semaphore",
            r"lock",
            r"deadlock",
            r"race\s+condition",
            r"async(hronous)?",
            r"parallel"
        ]
        
        self.risk_paths = [
            'src/', 'core/', 'lib/', 'engine/', 'kernel/', 
            'common/', 'utils/', 'main/', 'runtime/'
        ]
        
        self.risky_extensions = ['.c', '.cpp', '.h', '.py', '.java', '.js', '.go', '.rs']
        
        self.test_paths = ['test/', 'tests/', 'spec/', 'specs/', 'unittest/']

        self.knowledge_base = self._load_cache('knowledge_base.pkl', {
            'risky_patterns': set(self.risk_patterns),
            'successful_fixes': set(),
            'failure_correlations': defaultdict(int)
        })
        
        self.use_classifier = self.config.get('use_classifier', False)
        if self.use_classifier:
            self._initialize_classifier()

    def find_problematic_commit(self, good_commit: str, bad_commit: str, 
                              test_runner: Callable[[str], bool], 
                              test_name: str) -> Dict[str, Any]:
        """
        Find the first bad commit using enhanced binary search with risk analysis.
        
        Args:
            good_commit: Known good commit
            bad_commit: Known bad commit
            test_runner: Function to run tests on a commit
            test_name: Name of the test being analyzed
            
        Returns:
            Dictionary with information about the first problematic commit
        """
        try:
            self.logger.info(f"Starting enhanced binary search for test '{test_name}' between {good_commit} and {bad_commit}")
            
            commits = self.git_collector.get_commits_between(good_commit, bad_commit)
            self.logger.info(f"Found {len(commits)} commits to analyze")
            
            self.logger.info("Verifying endpoints")
            if not self._verify_commit(good_commit, test_runner, True):
                return {
                    "status": "error",
                    "message": "Good commit fails tests",
                    "commits_tested": 1
                }
                
            if not self._verify_commit(bad_commit, test_runner, False):
                return {
                    "status": "error",
                    "message": "Bad commit passes tests",
                    "commits_tested": 1
                }

            self.logger.info("Calculating risk scores for commits")
            for commit in commits:
                commit['risk_score'] = self._calculate_risk_score(commit)
            
            self.logger.info("Testing high-risk commits first")
            commits_by_risk = sorted(commits, key=lambda x: x['risk_score'], reverse=True)
            high_risk_commits = commits_by_risk[:min(5, len(commits))]
            
            for commit in high_risk_commits:
                self.logger.info(f"Testing high-risk commit: {commit['hash'][:8]} (score: {commit['risk_score']:.2f})")
                test_passed = self._test_commit(commit['hash'], test_runner)
                
                if not test_passed:
                    self.logger.info(f"Found bad high-risk commit: {commit['hash']}")
                    first_bad = self._bisect_range(good_commit, commit['hash'], commits, test_runner)
                    return self._prepare_result(first_bad, commits, test_name)
            
            self.logger.info("No high-risk commit failed, performing weighted binary search")
            first_bad = self._weighted_binary_search(commits, test_runner)
            return self._prepare_result(first_bad, commits, test_name)
            
        except Exception as e:
            self.logger.error(f"Error in enhanced binary search: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "commits_tested": len(self.history)
            }

    def _calculate_risk_score(self, commit: Dict[str, Any]) -> float:
        """
        Calculate a risk score for a commit based on various factors.
        
        Args:
            commit: Commit information dictionary
            
        Returns:
            Risk score between 0.0 and 1.0
        """
        cache_key = commit['hash']
        if cache_key in self.risk_scores_cache:
            return self.risk_scores_cache[cache_key]
            
        try:
            risk_score = 0.0
            
            commit_msg = commit.get('message', '').lower()
            for pattern in self.risk_patterns:
                if re.search(pattern, commit_msg, re.IGNORECASE):
                    risk_score += 0.2
                    break
            
            stats = commit.get('stats', {})
            file_count = stats.get('files_changed', 0)
            
            if 5 <= file_count <= 20:
                risk_score += 0.15
            elif file_count < 5:  
                risk_score += 0.1
            else: 
                risk_score += 0.05
                
            lines_changed = stats.get('insertions', 0) + stats.get('deletions', 0)
            if 20 <= lines_changed <= 200:
                risk_score += 0.15
            elif lines_changed < 20:  
                risk_score += 0.1
            else:  
                risk_score += 0.05
                
            
            try:
                modified_files = self.git_collector.get_files_modified_in_commit(commit['hash'])
                for file_path in modified_files:
                    for risky_path in self.risk_paths:
                        if risky_path in file_path:
                            risk_score += 0.1
                            break
                            
                    for ext in self.risky_extensions:
                        if file_path.endswith(ext):
                            risk_score += 0.05
                            break
            except Exception as e:
                self.logger.warning(f"Error getting modified files: {str(e)}")
                
            try:
                commit_date = datetime.fromisoformat(commit.get('date', '').replace('Z', '+00:00'))
                hour = commit_date.hour
                
                if 22 <= hour or hour <= 6:
                    risk_score += 0.1
            except Exception:
                pass
                
            risk_score += self._analyze_commit_metadata(commit)
                
            static_analysis = self._analyze_code_changes(commit['hash'])
            if static_analysis['risk_level'] == 'high':
                risk_score += 0.3
            elif static_analysis['risk_level'] == 'medium':
                risk_score += 0.15
                
            if self.use_classifier and len(self.training_data['labels']) >= 20:
                ml_risk = self._predict_commit_risk(commit)
                risk_score = 0.7 * risk_score + 0.3 * ml_risk
                
            final_score = min(1.0, risk_score)  
            self.risk_scores_cache[cache_key] = final_score
            self._save_cache(self.risk_scores_cache, 'risk_scores.pkl')
            
            return final_score
            
        except Exception as e:
            self.logger.warning(f"Error calculating risk score: {str(e)}")
            return 0.1  

    def _analyze_commit_metadata(self, commit: Dict[str, Any]) -> float:
        """
        Analyze commit metadata to calculate additional risk factors.
        
        Args:
            commit: Commit information dictionary
            
        Returns:
            Metadata risk score between 0.0 and 1.0
        """
        try:
            risk_score = 0.0
            
            author = commit.get('author', 'unknown')
            author_metrics = self.contributor_metrics[author]
            
            if author_metrics['commit_count'] < 5:
                risk_score += 0.2
            elif author_metrics['commit_count'] < 20:
                risk_score += 0.1
            
            if author_metrics['failure_rate'] > 0.3:  
                risk_score += 0.25
            elif author_metrics['failure_rate'] > 0.1:  
                risk_score += 0.15
            
            author_metrics['commit_count'] += 1
            if not author_metrics['first_commit_date']:
                author_metrics['first_commit_date'] = commit.get('date')
            
            message = commit.get('message', '')
            
            if len(message.split()) < 5:
                risk_score += 0.1
            
            if 'revert' in message.lower():
                risk_score += 0.2
            
            self._save_cache(self.contributor_metrics, 'contributor_metrics.pkl')
            
            return min(1.0, risk_score)
            
        except Exception as e:
            self.logger.debug(f"Error analyzing commit metadata: {str(e)}")
            return 0.0

    def _weighted_binary_search(self, commits: List[Dict[str, Any]], 
                              test_runner: Callable[[str], bool]) -> Dict[str, Any]:
        """
        Perform a weighted binary search based on risk scores.
        
        Args:
            commits: List of commit dictionaries with risk scores
            test_runner: Function to run tests on a commit
            
        Returns:
            First bad commit found, or None
        """
        if not commits:
            return None
            
        left = 0
        right = len(commits) - 1
        first_bad = None
        
        while left <= right:
            weights = [commits[i]['risk_score'] for i in range(left, right + 1)]
            total_weight = sum(weights)
            
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                cum_weights = np.cumsum(weights)
                split_idx = np.searchsorted(cum_weights, 0.5)
                mid = left + split_idx
            else:
                mid = (left + right) // 2
                
            current_commit = commits[mid]
            self.logger.info(f"Testing commit {current_commit['hash'][:8]} (risk score: {current_commit['risk_score']:.2f})")
            
            test_passed = self._test_commit(current_commit['hash'], test_runner)
            
            if test_passed:
                self.logger.debug(f"Commit {current_commit['hash'][:8]} passed, moving right")
                left = mid + 1
            else:
                self.logger.debug(f"Commit {current_commit['hash'][:8]} failed, moving left")
                first_bad = current_commit
                right = mid - 1
                
        return first_bad

    def _bisect_range(self, good_hash: str, bad_hash: str, 
                    all_commits: List[Dict[str, Any]], 
                    test_runner: Callable[[str], bool]) -> Dict[str, Any]:
        """
        Bisect between good and bad commit to find first bad commit.
        
        Args:
            good_hash: Known good commit hash
            bad_hash: Known bad commit hash
            all_commits: List of all commits
            test_runner: Function to run tests on a commit
            
        Returns:
            First bad commit found
        """
        good_idx = next((i for i, c in enumerate(all_commits) if c['hash'] == good_hash), 0)
        bad_idx = next((i for i, c in enumerate(all_commits) if c['hash'] == bad_hash), len(all_commits) - 1)
        
        commits_range = all_commits[good_idx:bad_idx+1]
        self.logger.info(f"Bisecting between good ({good_hash[:8]}) and bad ({bad_hash[:8]}) - {len(commits_range)} commits")
        
        left = 0
        right = len(commits_range) - 1
        first_bad = commits_range[-1]  
        
        while left <= right:
            mid = (left + right) // 2
            current_commit = commits_range[mid]
            self.logger.info(f"Testing commit {current_commit['hash'][:8]}")
            
            test_passed = self._test_commit(current_commit['hash'], test_runner)
            
            if test_passed:
                left = mid + 1
            else:
                first_bad = current_commit
                right = mid - 1
                
        return first_bad

    def _verify_commit(self, commit: str, test_runner: Callable[[str], bool], 
                     expected: bool) -> bool:
        """
        Verify a commit passes/fails as expected with retries.
        
        Args:
            commit: Commit hash to verify
            test_runner: Function to run tests on a commit
            expected: Expected test result (True=pass, False=fail)
            
        Returns:
            Whether the commit behaves as expected
        """
        for attempt in range(self.max_retries):
            if attempt > 0:
                self.logger.info(f"Retry attempt {attempt + 1}")
                time.sleep(self.retry_delay)
                
            result = self._test_commit(commit, test_runner)
            if result == expected:
                return True
                
            self.logger.warning(f"Test attempt {attempt + 1} failed: "
                             f"Expected {'pass' if expected else 'fail'}, "
                             f"got {'pass' if result else 'fail'}")
        
        return False

    def _test_commit(self, commit_hash: str, test_runner: Callable[[str], bool]) -> bool:
        """
        Run tests on a commit with retries and timing.
        
        Args:
            commit_hash: Commit hash to test
            test_runner: Function to run tests on a commit
            
        Returns:
            Test result (True=pass, False=fail)
        """
        for attempt in range(self.max_retries):
            try:
                self.git_collector.checkout_commit(commit_hash)
                
                start_time = time.time()
                result = test_runner(commit_hash)
                end_time = time.time()
                
                self.history.append({
                    "commit": commit_hash,
                    "attempt": attempt + 1,
                    "result": result,
                    "timestamp": time.time(),
                    "execution_time": end_time - start_time
                })
                
                return result
                
            except Exception as e:
                self.logger.warning(f"Test attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    self.logger.info(f"Retry attempt {attempt + 2}")
                    time.sleep(self.retry_delay)
                    
        return False

    def _prepare_result(self, first_bad: Dict[str, Any], 
                      commits: List[Dict[str, Any]], 
                      test_name: str) -> Dict[str, Any]:
        """
        Prepare the final result dictionary with enhanced analysis.
        
        Args:
            first_bad: First problematic commit
            commits: List of all commits
            test_name: Name of the test
            
        Returns:
            Result dictionary with detailed information
        """
        if (first_bad):
            commit_info = self.git_collector.get_commit_info(first_bad['hash'])
            modified_files = self.git_collector.get_files_modified_in_commit(first_bad['hash'])
            
            culprits = self._identify_likely_culprits(modified_files, first_bad['hash'])
            
            result = {
                "status": "success",
                "first_bad_commit": {
                    "hash": first_bad['hash'],
                    "author": first_bad.get('author', commit_info.get('author_name', 'Unknown')),
                    "date": first_bad.get('date', commit_info.get('date', '')),
                    "message": first_bad.get('message', commit_info.get('message', '')),
                    "modified_files": modified_files,
                    "risk_score": first_bad.get('risk_score', 0.0)
                },
                "likely_culprits": culprits,
                "commits_tested": len(self.history),
                "total_commits": len(commits),
                "test_name": test_name,
                "search_history": self.history,
                "performance_impact": self._analyze_performance_impact(first_bad['hash'], self.history)
            }
            
            self.logger.info(f"Found first bad commit: {first_bad['hash'][:8]}")
            return result
            
        return {
            "status": "error",
            "message": "Could not find the problematic commit",
            "commits_tested": len(self.history)
        }

    def _identify_likely_culprits(self, modified_files: List[str], 
                                commit_hash: str) -> List[Dict[str, Any]]:
        """
        Identify files that are likely culprits for the failure.
        
        Args:
            modified_files: List of files modified in the commit
            commit_hash: Commit hash
            
        Returns:
            List of likely culprit files with risk scores
        """
        culprits = []
        
        try:
            for file_path in modified_files:
                if any(file_path.endswith(ext) for ext in ['.md', '.txt', '.png', '.jpg']):
                    continue
                    
                risk_score = 0.0
                
                for ext in self.risky_extensions:
                    if file_path.endswith(ext):
                        risk_score += 0.2
                        break
                
                for risky_path in self.risk_paths:
                    if risky_path in file_path:
                        risk_score += 0.15
                        break
                
                if any(file_path.endswith(ext) for ext in self.risky_extensions):
                    try:
                        content = self.git_collector.get_file_at_commit(file_path, commit_hash)
                        if content:
                            for pattern in self.risk_patterns:
                                if re.search(pattern, content, re.IGNORECASE):
                                    risk_score += 0.1
                                    break
                    except Exception as e:
                        self.logger.debug(f"Error getting file content: {str(e)}")
                
                if risk_score > 0.1:  
                    culprits.append({
                        "file": file_path,
                        "risk_score": min(1.0, risk_score)
                    })
            
            
            culprits.sort(key=lambda x: x['risk_score'], reverse=True)
            return culprits[:5]  
            
        except Exception as e:
            self.logger.warning(f"Error identifying culprits: {str(e)}")
            return []

    def _analyze_performance_impact(self, commit_hash: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhanced performance impact analysis with execution time metrics.
        
        Args:
            commit_hash: Commit hash to analyze
            history: Test execution history
            
        Returns:
            Dictionary with performance impact information
        """
        try:
            commit_runs = [h for h in history if h['commit'] == commit_hash]
            if not commit_runs:
                return {"has_impact": False}
                
            execution_times = [run.get('execution_time', 0) for run in commit_runs]
            avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
            
            other_runs = [h for h in history if h['commit'] != commit_hash and 'execution_time' in h]
            other_avg = sum(h['execution_time'] for h in other_runs) / len(other_runs) if other_runs else 0
            
            has_impact = False
            if other_avg > 0:
                ratio = avg_execution_time / other_avg
                has_impact = ratio > 1.2 
            
            metrics = {
                "execution_time": avg_execution_time,
                "execution_time_ratio": ratio if other_avg > 0 else None,
                "runs_analyzed": len(commit_runs)
            }
            
            recommendation = ""
            if has_impact:
                if ratio > 2.0:
                    recommendation = "Critical performance regression detected. Investigate immediately."
                elif ratio > 1.5:
                    recommendation = "Significant performance regression. Review code changes affecting performance."
                else:
                    recommendation = "Minor performance regression detected. Consider optimization."
            
            return {
                "has_impact": has_impact,
                "metrics": metrics,
                "recommendation": recommendation
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance impact: {str(e)}")
            return {"has_impact": False}

    def _pattern_in_code_context(self, pattern: str, content: str) -> bool:
        """
        Determine if a pattern appears in actual code context versus comments.
        
        Args:
            pattern: Pattern to check
            content: File content
            
        Returns:
            Whether the pattern appears in code context
        """
        lines = content.split('\n')
        pattern_lower = pattern.lower()
        
        is_in_code = False
        in_multiline_comment = False
        
        comment_starters = {
            'py': '#', 'js': '//', 'java': '//', 'c': '//', 'cpp': '//', 
            'rb': '#', 'php': '//', 'go': '//'
        }
        
        file_ext = None
        if '.' in content[:1000]:  
            for ext in comment_starters:
                if f'.{ext}' in content[:1000]:
                    file_ext = ext
                    break
        
        comment_marker = comment_starters.get(file_ext, '#') 
        
        for line in lines:
            line_lower = line.lower()
            
            if not line_lower.strip():
                continue
                
            if '/*' in line_lower and not in_multiline_comment:
                in_multiline_comment = True
                if '*/' in line_lower[line_lower.find('/*')+2:]:
                    in_multiline_comment = False
                continue
                
            if '*/' in line_lower and in_multiline_comment:
                in_multiline_comment = False
                continue
                
            if in_multiline_comment:
                continue
                
            if comment_marker in line_lower:
                code_part = line_lower.split(comment_marker)[0]
                if pattern_lower in code_part:
                    return True
            else:
                if pattern_lower in line_lower:
                    return True
        
        return False

    def _load_cache(self, filename: str, default_value=None) -> Any:
        """Load a cache file or return default value if it doesn't exist."""
        cache_path = os.path.join(self.cache_dir, filename)
        try:
            if (os.path.exists(cache_path)):
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Error loading cache file {filename}: {str(e)}")
        
        return {} if default_value is None else default_value
        
    def _save_cache(self, data: Any, filename: str) -> None:
        """Save data to cache file."""
        cache_path = os.path.join(self.cache_dir, filename)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Error saving to cache file {filename}: {str(e)}")
            
    def _cache_key(self, commit_hash: str, test_name: str) -> str:
        """Create a unique cache key for a commit-test combination."""
        return f"{commit_hash}:{test_name}"

    def _analyze_code_changes(self, commit_hash: str) -> Dict[str, Any]:
        """
        Perform static analysis on code changes in a commit.
        
        Args:
            commit_hash: Commit hash to analyze
            
        Returns:
            Dictionary with static analysis results
        """
        try:
            modified_files = self.git_collector.get_files_modified_in_commit(commit_hash)
            analysis_results = {
                'api_changes': [],
                'threading_changes': [],
                'error_handling_changes': [],
                'test_coverage': [],
                'dependency_changes': [],
                'risk_level': 'low'
            }
            
            for file_path in modified_files:
                if not any(file_path.endswith(ext) for ext in self.risky_extensions):
                    continue
                
                try:
                    parent_hash = self.git_collector.get_parent_commit(commit_hash)
                    if parent_hash:
                        old_content = self.git_collector.get_file_at_commit(file_path, parent_hash) or ""
                        new_content = self.git_collector.get_file_at_commit(file_path, commit_hash) or ""
                        
                        if self._has_api_changes(old_content, new_content):
                            analysis_results['api_changes'].append(file_path)
                        
                        if self._has_threading_changes(old_content, new_content):
                            analysis_results['threading_changes'].append(file_path)
                        
                        if self._has_error_handling_changes(old_content, new_content):
                            analysis_results['error_handling_changes'].append(file_path)
                        
                        if any(test_path in file_path for test_path in self.test_paths):
                            analysis_results['test_coverage'].append(file_path)
                        
                        if self._has_dependency_changes(old_content, new_content):
                            analysis_results['dependency_changes'].append(file_path)
                except Exception as e:
                    self.logger.debug(f"Error analyzing file {file_path}: {str(e)}")
            
            if len(analysis_results['api_changes']) > 0 or len(analysis_results['threading_changes']) > 0:
                analysis_results['risk_level'] = 'high'
            elif len(analysis_results['error_handling_changes']) > 0:
                analysis_results['risk_level'] = 'medium'
                
            return analysis_results
            
        except Exception as e:
            self.logger.warning(f"Error in static analysis: {str(e)}")
            return {'risk_level': 'unknown'}
    
    def _has_api_changes(self, old_content: str, new_content: str) -> bool:
        """Check if the changes include API modifications."""
        try:
            old_funcs = self._extract_function_signatures(old_content)
            new_funcs = self._extract_function_signatures(new_content)
            
            for func in old_funcs:
                if func not in new_funcs:
                    return True
            
            for pattern in self.api_change_patterns:
                if re.search(pattern, new_content, re.IGNORECASE):
                    return True
                    
            return False
        except Exception:
            return False
    
    def _extract_function_signatures(self, content: str) -> List[str]:
        """Extract function signatures from code content."""
        signatures = []
        try:
            function_patterns = [
                r'def\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)',  # Python
                r'function\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)',  # JavaScript
                r'([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_]+)\s*\(([^)]*)\)\s*{',  # C/C++/Java
            ]
            
            for pattern in function_patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    signatures.append(match.group(0))
                    
            return signatures
        except Exception:
            return signatures
    
    def _has_threading_changes(self, old_content: str, new_content: str) -> bool:
        """Check if the changes include threading/concurrency modifications."""
        try:
            for pattern in self.threading_patterns:
                if (re.search(pattern, new_content, re.IGNORECASE) and 
                    not re.search(pattern, old_content, re.IGNORECASE)):
                    return True
            return False
        except Exception:
            return False
    
    def _has_error_handling_changes(self, old_content: str, new_content: str) -> bool:
        """Check if the changes include error handling modifications."""
        error_patterns = [
            r'try\s*{', r'try:', r'catch\s*\(', r'except:', 
            r'throw', r'raise', r'error', r'exception'
        ]
        
        try:
            for pattern in error_patterns:
                if (re.search(pattern, new_content, re.IGNORECASE) and 
                    not re.search(pattern, old_content, re.IGNORECASE)):
                    return True
            return False
        except Exception:
            return False
            
    def _has_dependency_changes(self, old_content: str, new_content: str) -> bool:
        """Check if the changes include dependency modifications."""
        dependency_patterns = [
            r'import\s+', r'#include\s+', r'require\s+', r'using\s+'
        ]
        
        try:
            for pattern in dependency_patterns:
                old_deps = set(re.findall(f"{pattern}([^\n;]+)", old_content))
                new_deps = set(re.findall(f"{pattern}([^\n;]+)", new_content))
                if old_deps != new_deps:
                    return True
            return False
        except Exception:
            return False

    def _run_two_tier_performance_test(self, commit_hash: str, test_runner: Callable[[str, Dict[str, Any]], bool]) -> Dict[str, Any]:
        """
        Run a two-tier performance test on a commit.
        
        Args:
            commit_hash: Commit hash to test
            test_runner: Function to run tests on a commit
            
        Returns:
            Dictionary with performance test results
        """
        try:
            light_options = {'performance_test': 'light'}
            light_result = test_runner(commit_hash, light_options)
            
            perf_metrics = {
                'tier': 'light',
                'passed': light_result,
                'execution_time': self.history[-1].get('execution_time', 0) if self.history else 0
            }
            
            if not light_result or self._needs_detailed_analysis(commit_hash):
                self.logger.info(f"Running detailed performance analysis on {commit_hash[:8]}")
                full_options = {'performance_test': 'full'}
                full_result = test_runner(commit_hash, full_options)
                
                perf_metrics.update({
                    'tier': 'full',
                    'passed': full_result,
                    'execution_time': self.history[-1].get('execution_time', 0) if self.history else 0
                })
                
            return perf_metrics
            
        except Exception as e:
            self.logger.error(f"Error in two-tier performance test: {str(e)}")
            return {'tier': 'error', 'passed': False}
    
    def _needs_detailed_analysis(self, commit_hash: str) -> bool:
        """Determine if a commit needs detailed performance analysis."""
        try:
            analysis = self._analyze_code_changes(commit_hash)
            
            performance_keywords = ['performance', 'speed', 'optimization', 'slow', 'fast']
            
            commit_info = self.git_collector.get_commit_info(commit_hash)
            message = commit_info.get('message', '').lower()
            
            if any(keyword in message for keyword in performance_keywords):
                return True
                
            modified_files = self.git_collector.get_files_modified_in_commit(commit_hash)
            for file_path in modified_files:
                if any(perf_path in file_path for perf_path in ['perf', 'benchmark', 'performance']):
                    return True
            
            if analysis['risk_level'] == 'high':
                return True
                
            return False
            
        except Exception as e:
            self.logger.debug(f"Error determining need for detailed analysis: {str(e)}")
            return False

    def _initialize_classifier(self):
        """Initialize a simple decision tree classifier for commit risk assessment."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.tree import DecisionTreeClassifier
            
            model_path = os.path.join(self.cache_dir, 'risk_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                self.logger.info("Loaded existing risk classifier model")
            else:
                self.classifier = RandomForestClassifier(n_estimators=10)
                self.logger.info("Initialized new risk classifier model")
                
            self.training_data = self._load_cache('training_data.pkl', {'features': [], 'labels': []})
            
        except ImportError:
            self.logger.warning("scikit-learn not available; classifier disabled")
            self.use_classifier = False
        except Exception as e:
            self.logger.warning(f"Error initializing classifier: {str(e)}")
            self.use_classifier = False
    
    def _extract_commit_features(self, commit: Dict[str, Any]) -> List[float]:
        """
        Extract numerical features from a commit for ML classification.
        
        Args:
            commit: Commit information dictionary
            
        Returns:
            List of numerical features
        """
        features = []
        try:
            message = commit.get('message', '').lower()
            pattern_count = sum(1 for pattern in self.risk_patterns if re.search(pattern, message))
            features.append(pattern_count / len(self.risk_patterns))  # Normalize
            
            stats = commit.get('stats', {})
            file_count = stats.get('files_changed', 0)
            features.append(min(1.0, file_count / 20))  # Normalize, cap at 1.0
            
            lines_changed = stats.get('insertions', 0) + stats.get('deletions', 0)
            features.append(min(1.0, lines_changed / 500))  # Normalize, cap at 1.0
            
            if stats.get('insertions', 0) > 0:
                del_ins_ratio = stats.get('deletions', 0) / stats.get('insertions', 0)
                features.append(min(1.0, del_ins_ratio))  # Normalize, cap at 1.0
            else:
                features.append(0.0)
            
            modified_files = self.git_collector.get_files_modified_in_commit(commit['hash'])
            risky_file_count = sum(1 for f in modified_files if any(f.endswith(ext) for ext in self.risky_extensions))
            features.append(min(1.0, risky_file_count / max(1, len(modified_files))))  # Normalize
            
            author = commit.get('author', 'unknown')
            author_metrics = self.contributor_metrics[author]
            exp_feature = 1.0 - min(1.0, author_metrics['commit_count'] / 100)  # Normalize, invert, cap
            features.append(exp_feature)
            
            while len(features) < 10:  # Ensure consistent length
                features.append(0.0)
                
            return features
            
        except Exception as e:
            self.logger.debug(f"Error extracting commit features: {str(e)}")
            return [0.0] * 10  # Return zeroed features on error
    
    def _update_classifier(self, commit: Dict[str, Any], was_problematic: bool):
        """
        Update the classifier with new training data.
        
        Args:
            commit: Commit information dictionary
            was_problematic: Whether the commit was problematic
        """
        if not self.use_classifier:
            return
            
        try:
            features = self._extract_commit_features(commit)
            
            self.training_data['features'].append(features)
            self.training_data['labels'].append(1 if was_problematic else 0)
            
            max_samples = 1000
            if len(self.training_data['labels']) > max_samples:
                self.training_data['features'] = self.training_data['features'][-max_samples:]
                self.training_data['labels'] = self.training_data['labels'][-max_samples:]
            
            if len(self.training_data['labels']) >= 20:
                from sklearn.utils import shuffle
                
                X, y = shuffle(self.training_data['features'], self.training_data['labels'])
                
                self.classifier.fit(X, y)
                
                model_path = os.path.join(self.cache_dir, 'risk_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(self.classifier, f)
                    
                self._save_cache(self.training_data, 'training_data.pkl')
                
                self.logger.info(f"Updated risk classifier with {len(y)} samples")
                
        except Exception as e:
            self.logger.warning(f"Error updating classifier: {str(e)}")
    
    def _predict_commit_risk(self, commit: Dict[str, Any]) -> float:
        """
        Use the ML classifier to predict commit risk.
        
        Args:
            commit: Commit information dictionary
            
        Returns:
            Risk score between 0.0 and 1.0
        """
        if not self.use_classifier:
            return 0.0
            
        try:
            features = self._extract_commit_features(commit)
            
            if len(self.training_data['labels']) >= 20:
                prob = self.classifier.predict_proba([features])[0][1]
                return float(prob)
            
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"Error predicting commit risk: {str(e)}")
            return 0.0