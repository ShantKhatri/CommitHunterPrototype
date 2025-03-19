"""
String Matcher Analyzer Module

This module implements a rule-based analyzer that matches error messages with commits
to identify likely problematic commits causing test failures.
"""

import re
import os
import nltk
import numpy as np
import difflib
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

class StringMatcher:
    """
    Enhanced analyzer that matches error messages with commits
    and implements the SZZ algorithm to identify bug-inducing changes.
    """

    def __init__(self, git_collector):
        """
        Initialize the string matcher with a GitCollector instance.
        
        Args:
            git_collector: An instance of GitCollector for accessing repository data
        """
        self.git_collector = git_collector
        self.logger = logging.getLogger(__name__)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
            
        self.stopwords = self._load_stopwords()
        self.stemmer = PorterStemmer()
        
        self.technical_keywords = {
            'null', 'undefined', 'nan', 'exception', 'error', 'fail',
            'memory', 'leak', 'crash', 'timeout', 'deadlock', 'race',
            'synchronization', 'concurrent', 'thread', 'performance'
        }
        
        self.code_keywords = {
            'if', 'else', 'for', 'while', 'try', 'catch', 'throw',
            'return', 'class', 'function', 'method', 'synchronized',
            'volatile', 'static', 'final', 'const', 'var', 'let'
        }
        
        self.suspicious_patterns = [
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
        
        self.logger.info("String matcher initialized with {} suspicious patterns".format(len(self.suspicious_patterns)))

    def _load_stopwords(self) -> Set[str]:
        """Load and prepare stopwords for NLP processing"""
        try:
            stops = set(stopwords.words('english'))
            stops.update([
                'test', 'tests', 'testing', 'tested',
                'file', 'files', 'line', 'lines',
                'code', 'commit', 'version', 'fix', 'issue',
                'bug', 'error', 'problem', 'success', 'failure'
            ])
            return stops
        except Exception as e:
            print(f"Error loading stopwords: {e}")
            return set()

    def extract_error_keywords(self, error_message: str) -> List[str]:
        """
        Extract meaningful keywords from an error message.
        
        This improved version:
        1. Handles stack traces better
        2. Extracts class names, method names, and variable names
        3. Prioritizes technical terms
        """
        if not error_message:
            return []
            
        error_message = error_message.lower()
        
        stack_frames = []
        for line in error_message.split('\n'):
            match = re.search(r'at\s+([\w\.]+)\(([\w\.]+):(\d+)\)', line)
            if match:
                frame = match.group(1)
                stack_frames.append(frame)
                continue
                
            match = re.search(r'file\s+"[^"]+",\s+line\s+\d+,\s+in\s+([\w\.]+)', line, re.IGNORECASE)
            if match:
                frame = match.group(1)
                stack_frames.append(frame)
        
        classes_and_methods = set()
        for frame in stack_frames:
            parts = frame.split('.')
            if len(parts) > 1:
                classes_and_methods.update(parts)
        
        words = re.findall(r'\b\w+\b', error_message)
        
        identifiers = re.findall(r'\b([a-z]+[A-Z]\w+|\w+_\w+)\b', error_message)
        
        all_words = words + list(classes_and_methods) + identifiers
        
        filtered_words = [
            word for word in all_words 
            if word not in self.stopwords or word in self.technical_keywords
        ]
        
        stemmed_words = [self.stemmer.stem(word) for word in filtered_words]
        
        word_counts = Counter(stemmed_words)
        
        keywords = [word for word, count in word_counts.most_common(20)]
        
        return keywords

    def analyze_commit_diff(self, commit_hash: str, keywords: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Analyze a commit diff to find matches with error keywords.
        Returns a confidence score and list of matches.
        """
        commit_info = self.git_collector.get_commit_info(commit_hash)
        if not commit_info:
            return 0.0, []
            
        modified_files = self.git_collector.get_modified_files(commit_hash)
        
        matches = []
        total_matches = 0
        max_matches = 0
        
        message = commit_info.get('message', '').lower()
        message_matches = self._count_keyword_matches(message, keywords)
        if message_matches > 0:
            matches.append({
                'type': 'commit_message',
                'content': message,
                'matches': message_matches,
                'keywords': [kw for kw in keywords if kw.lower() in message.lower()]
            })
            total_matches += message_matches * 2  # Weight commit message matches higher
            max_matches = max(max_matches, message_matches)
        
        for file_path, file_info in modified_files.items():
            try:
                file_content = self.git_collector.repo.git.show(f"{commit_hash}:{file_path}")
                
                file_matches = self._count_keyword_matches(file_content, keywords)
                if file_matches > 0:
                    matches.append({
                        'type': 'file_content',
                        'file': file_path,
                        'matches': file_matches,
                        'keywords': [kw for kw in keywords if kw.lower() in file_content.lower()]
                    })
                    total_matches += file_matches
                    max_matches = max(max_matches, file_matches)
            except Exception as e:
                continue
                
        if not keywords:
            confidence = 0.0
        else:
            keyword_coverage = len([m for m in matches if m['matches'] > 0]) / max(len(keywords), 1)
            match_strength = min(1.0, total_matches / (len(keywords) * 3))
            confidence = 0.4 * keyword_coverage + 0.6 * match_strength
            
        return confidence, matches
    
    def _count_keyword_matches(self, text: str, keywords: List[str]) -> int:
        """Count occurrences of keywords in text"""
        if not text or not keywords:
            return 0
            
        text = text.lower()
        count = 0
        
        for keyword in keywords:
            keyword = keyword.lower()
            count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
            
        return count
        
    def identify_fix_commits(self, commit_range: List[str]) -> List[str]:
        """
        Identify commits that are likely fixing bugs based on commit messages.
        """
        fix_commits = []
        
        for commit_hash in commit_range:
            commit_info = self.git_collector.get_commit_info(commit_hash)
            if not commit_info:
                continue
                
            message = commit_info.get('message', '').lower()
            
            if any(pattern in message for pattern in [
                'fix', 'fixes', 'fixed', 'resolve', 'resolves', 'resolved',
                'close', 'closes', 'closed', 'bug', 'defect', 'issue'
            ]):
                fix_commits.append(commit_hash)
                
        return fix_commits
        
    def identify_bug_inducing_commits(self, fixing_commit: str) -> List[Dict[str, Any]]:
        """
        Implement the SZZ algorithm to identify bug-inducing commits.
        Returns a list of potential bug-inducing commits with confidence scores.
        """
        commit_info = self.git_collector.get_commit_info(fixing_commit)
        if not commit_info or not commit_info.get('parents'):
            return []
            
        modified_files = self.git_collector.get_modified_files(fixing_commit)
        
        potential_bug_inducers = []
        
        for file_path, file_info in modified_files.items():
            if not file_info.get('lines'):
                continue
                
            blame_info = self.git_collector.get_blame_info(
                fixing_commit, 
                file_path, 
                file_info.get('lines', [])
            )
            
            for bug_commit in blame_info.get('commits', []):
                if bug_commit == fixing_commit:
                    continue
                    
                bug_commit_info = self.git_collector.get_commit_info(bug_commit)
                if not bug_commit_info:
                    continue
                
                age_factor = self._calculate_age_factor(
                    bug_commit_info.get('committed_date'),
                    commit_info.get('committed_date')
                )
                
                classification = self.git_collector.classify_commit(bug_commit)
                class_factor = {
                    'feature_addition': 0.8,  # New features often introduce bugs
                    'corrective': 0.3,        # Fixes sometimes introduce new bugs
                    'merge': 0.6,             # Merges can introduce integration bugs
                    'perfective': 0.5,        # Code improvements sometimes cause issues
                    'preventive': 0.4,        # Testing changes sometimes miss cases
                    'non_functional': 0.2,    # Documentation rarely introduces bugs
                    'unknown': 0.5            # Default middle value
                }.get(classification, 0.5)
                
                churn = bug_commit_info.get('lines_added', 0) + bug_commit_info.get('lines_deleted', 0)
                churn_factor = min(1.0, churn / 500)  # Normalize with a cap
                
                exp = bug_commit_info.get('author_experience', 0)
                exp_factor = max(0.2, 1.0 - (min(exp, 50) / 50))  # More experience = lower risk
                
                confidence = (0.3 * age_factor + 
                             0.3 * class_factor + 
                             0.2 * churn_factor + 
                             0.2 * exp_factor)
                             
                potential_bug_inducers.append({
                    'commit': bug_commit,
                    'classification': classification,
                    'modified_file': file_path,
                    'confidence': confidence,
                    'author': bug_commit_info.get('author'),
                    'date': bug_commit_info.get('committed_date'),
                    'message': bug_commit_info.get('message')
                })
        
        potential_bug_inducers.sort(key=lambda x: x['confidence'], reverse=True)
        
        unique_commits = {}
        for inducer in potential_bug_inducers:
            commit = inducer['commit']
            if commit not in unique_commits or unique_commits[commit]['confidence'] < inducer['confidence']:
                unique_commits[commit] = inducer
                
        return list(unique_commits.values())
        
    def _calculate_age_factor(self, bug_date, fix_date) -> float:
        """
        Calculate age factor based on time between bug introduction and fix.
        Recent changes are more suspicious than older ones.
        """
        if not bug_date or not fix_date:
            return 0.5  # Default
            
        try:
            time_diff = (fix_date - bug_date).days
            
            if time_diff <= 0:
                return 1.0  # Same day
            elif time_diff < 7:
                return 0.9  # Within a week
            elif time_diff < 30:
                return 0.7  # Within a month
            elif time_diff < 90:
                return 0.5  # Within a quarter
            elif time_diff < 365:
                return 0.3  # Within a year
            else:
                return 0.1  # Over a year
        except Exception:
            return 0.5  # Default on error

    def find_suspicious_commits(self, good_commit: str, bad_commit: str, 
                               error_message: str, threshold: float = 0.6,
                               context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Find suspicious commits between a good and bad commit based on error message.
        
        Args:
            good_commit: The last known good commit hash
            bad_commit: The first known bad commit hash
            error_message: The error message from the failed test
            threshold: Minimum relevance score to consider a commit suspicious
            context: Additional context (like the problematic commit from binary search)
            
        Returns:
            List of suspicious commits with scores and relevant snippets
        """

        problematic_commit = None
        if context and 'problematic_commit' in context:
            problematic_commit = context['problematic_commit']
            self.logger.info(f"Using problematic commit from context: {problematic_commit[:8]}")

        self.logger.info(f"Finding suspicious commits between {good_commit} and {bad_commit}")

        if error_message:
            self.logger.info(f"Processing error message ({len(error_message)} chars): {error_message[:100]}...")
        else:
            self.logger.warning("No error message provided")
        
        if error_message:
            keywords = self.extract_error_keywords(error_message)
        else:
            try:
                keywords = self._get_default_keywords(good_commit, bad_commit)
            except Exception as e:
                self.logger.warning(f"Error generating default keywords: {str(e)}")
                keywords = ['error', 'exception', 'failure', 'crash']
        
        keywords.extend(self._get_repository_specific_keywords(good_commit, bad_commit))
        keywords = list(dict.fromkeys(keywords))  # Remove duplicates
        
        self.logger.info(f"Using {len(keywords)} keywords: {', '.join(keywords[:10])}" + 
                        ("..." if len(keywords) > 10 else ""))
        
        try:
            commits = self.git_collector.get_commits_between(good_commit, bad_commit)
            if not commits:
                self.logger.warning("No commits found between good and bad commits")
                return []
                
            self.logger.info(f"Analyzing {len(commits)} commits")
            suspicious_commits = []

            for commit in commits:
                try:
                    commit_hash = commit['hash']
                    score, modified_files = self.analyze_commit_diff(commit_hash, keywords)
                    
                    if score >= threshold:
                        suspicious_commit = {
                            'hash': commit_hash,
                            'score': score,
                            'confidence_level': 'High' if score > 0.6 else 'Medium' if score > 0.3 else 'Low',
                            'files_changed': modified_files,
                            'message': commit.get('message', '').strip(),
                            'author': commit.get('author', 'Unknown'),
                            'date': commit.get('date', ''),
                            'analysis': {
                                'keywords_matched': [k for k in keywords if k.lower() in 
                                                commit.get('message', '').lower()],
                                'patterns_matched': [p for p in self.suspicious_patterns 
                                                if re.search(p, commit.get('message', ''), 
                                                            re.IGNORECASE)]
                            }
                        }
                        suspicious_commits.append(suspicious_commit)
                        self.logger.info(f"Found suspicious commit {commit_hash[:8]} with score {score:.2f} "
                                         f"({len(modified_files)} files changed)")
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing commit {commit.get('hash', 'Unknown')}: {str(e)}")
                    continue
            
            if problematic_commit:
                for commit in suspicious_commits:
                    if commit['hash'] == problematic_commit:
                        commit['score'] = min(1.0, commit['score'] * 1.2)
                        commit['is_problematic'] = True  # Mark it
                        break
            return sorted(
                [c for c in suspicious_commits if c['score'] >= threshold], 
                key=lambda x: x['score'], 
                reverse=True
            )
            
            self.logger.info(f"Found {len(suspicious_commits)} suspicious commits above threshold {threshold}")
            return suspicious_commits
            
        except Exception as e:
            self.logger.error(f"Error in commit analysis: {str(e)}")
            return []

    def get_summary_report(self, suspicious_commits: List[Dict]) -> str:
        """Generate a human-readable summary report of suspicious commits."""
        if not suspicious_commits:
            return "No suspicious commits found."
            
        report = ["Suspicious Commits Summary:", "=" * 25, ""]
        
        for idx, commit in enumerate(suspicious_commits, 1):
            message_first_line = commit['message'].split('\n')[0] if '\n' in commit['message'] else commit['message']
            report.extend([
                f"{idx}. Commit: {commit['hash'][:8]}",
                f"   Score: {commit['score']:.2f}",
                f"   Date: {commit['date']}",
                f"   Author: {commit['author']}",
                f"   Message: {message_first_line}",
                f"   Modified Files ({len(commit['files_changed'])} total):"
            ])
            
            sorted_files = sorted(commit['files_changed'], 
                                key=lambda x: x['score'], 
                                reverse=True)
            
            for file in sorted_files:
                report.extend([
                    f"     - {file['path']}",
                    f"       Score: {file['score']:.2f}",
                    f"       Changes: +{file['additions']} -{file['deletions']}",
                    f"       Matches: {', '.join(file['matches']) if file['matches'] else 'None'}"
                ])
            
            report.append("")  # Add blank line between commits
        
        return "\n".join(report)

    def _get_default_keywords(self, good_commit: str, bad_commit: str) -> List[str]:
        """Generate default keywords based on diffs between commits."""
        try:
            keywords = set()
            
            try:
                diff_output = self.git_collector.repo.git.diff(f"{good_commit}..{bad_commit}", name_only=True)
                file_paths = diff_output.split('\n')
                
                for file_path in file_paths:
                    if not file_path.strip():
                        continue
                        
                    filename = file_path.split('/')[-1].split('\\')[-1]
                    base_name = filename.split('.')[0] if '.' in filename else filename
                    
                    for part in re.split(r'[_\-]', base_name):
                        if len(part) >= 3 and part.lower() not in self.stopwords:
                            keywords.add(part)
            except Exception as e:
                self.logger.debug(f"Error getting diff: {str(e)}")
            
            general_keywords = [
                'error', 'exception', 'failure', 'crash', 'nullptr', 
                'null', 'segfault', 'memory', 'leak', 'assertion'
            ]
            keywords.update(general_keywords)
            
            return list(keywords)
            
        except Exception as e:
            self.logger.warning(f"Error generating default keywords: {str(e)}")
            return ['error', 'exception', 'failure', 'crash', 'bug']

    def _get_repository_specific_keywords(self, good_commit: str, bad_commit: str) -> List[str]:
        """Get repository-specific keywords based on codebase patterns."""
        repo_name = self.git_collector.repo_name.lower() if hasattr(self.git_collector, 'repo_name') else ""
        
        if 'java' in repo_name or self._is_java_repo():
            return ['NullPointerException', 'RuntimeException', 'ClassNotFoundException', 
                    'OutOfMemoryError', 'ConcurrentModificationException', 'thread']
        
        elif 'openj9' in repo_name or 'c++' in repo_name or self._is_cpp_repo():
            return ['JIT', 'GC', 'memory', 'compiler', 'segmentation', 
                    'nullptr', 'stack', 'heap', 'buffer', 'overflow', 'allocation']
        
        elif 'python' in repo_name or self._is_python_repo():
            return ['TypeError', 'ValueError', 'ImportError', 'KeyError', 
                    'AttributeError', 'IndexError', 'RuntimeError']
        
        return ['regression', 'performance', 'timeout', 'failure']

    def _is_java_repo(self) -> bool:
        """Check if repository is primarily Java."""
        return self._check_file_count_by_extension('.java') > 50

    def _is_cpp_repo(self) -> bool:
        """Check if repository is primarily C++."""
        cpp_count = self._check_file_count_by_extension('.cpp') + self._check_file_count_by_extension('.h')
        return cpp_count > 50

    def _is_python_repo(self) -> bool:
        """Check if repository is primarily Python."""
        return self._check_file_count_by_extension('.py') > 50

    def _check_file_count_by_extension(self, extension: str) -> int:
        """Count number of files with given extension in repository."""
        try:
            if hasattr(self.git_collector, 'count_files_by_extension'):
                return self.git_collector.count_files_by_extension(extension)
            
            count = 0
            for obj in self.git_collector.repo.head.commit.tree.traverse():
                if obj.path.endswith(extension):
                    count += 1
                if count > 100:  # Early return if we find many
                    return count
            return count
        except Exception:
            return 0

    def _pattern_in_code_context(self, pattern: str, content: str) -> bool:
        """
        Determine if a pattern appears in actual code context versus comments.
        
        Args:
            pattern: The pattern to check
            content: The file content to analyze
            
        Returns:
            True if pattern appears in code context, False if only in comments
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
        if '.' in content[:1000]:  # Check beginning of file for common markers
            for ext in comment_starters:
                if f'.{ext}' in content[:1000]:
                    file_ext = ext
                    break
        
        comment_marker = comment_starters.get(file_ext, '#')  # Default to # if unknown
        
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