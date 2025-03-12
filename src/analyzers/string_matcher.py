"""
String Matcher Analyzer Module

This module implements a rule-based analyzer that matches error messages with commits
to identify likely problematic commits causing test failures.
"""

import re
import numpy as np
import difflib
import logging
from typing import List, Dict, Any, Optional, Tuple, Set

class StringMatcher:
    """
    A rule-based analyzer that identifies problematic commits by matching
    error messages with keywords in commit messages and code diffs.
    """


    def _load_stopwords(self) -> Set[str]:
        """Load common stopwords to filter out noise, customized for CommitHunter and Eclipse OpenJ9."""
        stopwords = {
            'test', 'assert', 'error', 'exception', 'the', 'and', 'but', 'or',
            'for', 'not', 'with', 'this', 'that', 'these', 'those', 'null',
            'true', 'false', 'java', 'org', 'com', 'net', 'public', 'private',
            'static', 'class', 'void', 'main', 'string', 'int', 'boolean',
            'expected', 'actual', 'failed', 'failure', 'method', 'trace', 'stack',
            'debug', 'log', 'warning', 'info', 'traceback', 'invoke', 'process',
            'timeout', 'halt', 'exit', 'trigger', 'event', 'execution'
        }

        # Eclipse OpenJ9 and CommitHunter-specific stopwords
        openj9_stopwords = {
            'j9', 'vm', 'jit', 'runtime', 'gc', 'javaheap', 'heap', 'garbage',
            'collector', 'compilation', 'thread', 'mutex', 'lock', 'monitor',
            'synchronization', 'segfault', 'core', 'dump', 'threading',
            'interpreter', 'optimization', 'bytecode', 'jitserver', 'signal',
            'segmentation', 'trampoline', 'frame', 'native', 'nativecode', 'sandbox',
            'oom', 'outofmemory', 'compressedrefs', 'aot', 'dll', 'so', 'library',
            'dynamic', 'classloader', 'javaclass', 'classpath', 'jni', 'jnierror',
            'gcstats', 'gcpolicy', 'heapdump', 'corefile', 'crashdump', 'oomkiller',
            'metaspace', 'verbosegc', 'xmx', 'xms', 'xmn', 'xxgc', 'compactedheap',
            'compressedoops', 'runtimeexception', 'npe', 'nullpointerexception',
            'illegalstateexception', 'illegalargumentexception', 'segv', 'sigsegv'
        }

        stopwords.update(openj9_stopwords)

        return stopwords
    
    
    def __init__(self, git_collector):
        """
        Initialize the string matcher with a GitCollector instance.
        
        Args:
            git_collector: An instance of GitCollector for accessing repository data
        """
        self.git_collector = git_collector
        self.logger = logging.getLogger(__name__)
        
        self.stopwords = self._load_stopwords()
        
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

    def extract_error_keywords(self, error_message: str) -> List[str]:
        """
        Extract key terms from test failure messages with improved pattern matching.
        
        Args:
            error_message: The error or failure message from the test
            
        Returns:
            List of extracted keywords relevant to the failure
        """
        if not error_message:
            self.logger.warning("No error message provided")
            return []
            
        self.logger.debug(f"Extracting keywords from error message: {error_message[:100]}...")
        
        keywords = set()
        
        file_line_patterns = re.findall(r'([a-zA-Z0-9_/\\.-]+\.(?:cpp|java|h|py|js|go))(?::(\d+))?', error_message)
        for file_path, line_num in file_line_patterns:
            filename = file_path.split('/')[-1].split('\\')[-1].split('.')[0]
            keywords.add(filename)
            
            path_parts = file_path.split('/')
            if len(path_parts) > 1:
                component = path_parts[0]
                if len(component) > 2 and component.lower() not in self.stopwords:
                    keywords.add(component)
        
        error_types = re.findall(r'([A-Z][a-zA-Z0-9_]*(?:Exception|Error|Failure|Fault))', error_message)
        keywords.update(error_types)
        
        if 'segmentation fault' in error_message.lower():
            keywords.add('segfault')
            keywords.add('memory')
        
        if 'null pointer' in error_message.lower() or 'nullptr' in error_message.lower():
            keywords.add('nullptr')
        
        if 'out of memory' in error_message.lower() or 'outofmemory' in error_message.lower():
            keywords.add('memory')
            keywords.add('leak')
        
        function_names = re.findall(r'(?:in|at|function)\s+([A-Za-z][A-Za-z0-9_:]+)\s*\(', error_message)
        keywords.update(function_names)
        
        class_names = re.findall(r'([A-Z][a-zA-Z0-9_]*)\.[a-zA-Z0-9_]+', error_message)
        keywords.update(class_names)
        
        identifiers = re.findall(r'\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b', error_message)  # CamelCase
        identifiers.extend(re.findall(r'\b([a-z][a-z0-9]+(?:_[a-z0-9]+)+)\b', error_message))  # snake_case
        keywords.update(identifiers)
        
        for term in re.findall(r'\b[A-Za-z][a-zA-Z0-9_]{3,}\b', error_message):
            if term.lower() not in self.stopwords and len(term) >= 4:
                keywords.add(term)
        
        noise_terms = {'error', 'exception', 'failure', 'test', 'assert', 'expected', 'actual'}
        filtered_keywords = [k for k in keywords if k.lower() not in noise_terms]
        
        if error_types:
            filtered_keywords.extend(error_types)
        
        self.logger.info(f"Extracted {len(filtered_keywords)} keywords: {', '.join(filtered_keywords)}")
        return filtered_keywords
    
    def analyze_commit_diff(self, commit_hash: str, keywords: List[str]) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze commit diff with enhanced confidence scoring and optimized normalization."""
        
        SCORING_WEIGHTS = {
            'keyword_match': {
                'weight': 0.35,  # Increased weight for direct keyword matches
                'max_occurrences': 3,
                'decay_factor': 0.7  # Diminishing returns for repeated keywords
            },
            'code_file': {
                'weight': 0.25,
                'bonus_paths': ['src/', 'main/', 'core/', 'include/']
            },
            'test_file': {
                'weight': 0.20,
                'test_paths': ['test/', 'tests/', 'unittest/', 'functional/']
            },
            'error_pattern': {
                'weight': 0.40,
                'patterns': ['error', 'exception', 'fail', 'crash', 'invalid', 'nullptr', 
                             'segfault', 'assertion', 'timeout', 'deadlock']
            },
            'change_size': {
                'weight': 0.30,
                'small_change': 30,    # Small, focused changes (higher weight)
                'medium_change': 150,  # Medium changes
                'large_change': 500    # Large changes (lower weight)
            }
        }

        try:
            modified_files = []
            total_score = 0.0
            commit = self.git_collector.repo.commit(commit_hash)
            file_count = 0
            
            commit_msg = commit.message.lower()
            msg_keywords = set()
            
            for pattern in self.suspicious_patterns:
                if re.search(pattern, commit_msg, re.IGNORECASE):
                    msg_keywords.add(pattern)
            
            commit_msg_bonus = min(0.2, len(msg_keywords) * 0.05)

            if commit.parents:
                diff_index = commit.parents[0].diff(commit)
                
                file_count = sum(1 for d in diff_index if d.b_blob)
                
                for diff in diff_index:
                    if not diff.b_blob:
                        continue
                        
                    try:
                        content = diff.b_blob.data_stream.read().decode('utf-8', errors='ignore')
                        matches = []
                        file_stats = commit.stats.files.get(diff.b_path, {})
                        confidence_metrics = {'keyword_matches': 0, 'code_relevance': 0, 
                                             'test_coverage': 0, 'error_indicators': 0, 
                                             'change_impact': 0}
                        
                        matched_keywords_count = 0
                        content_lower = content.lower()
                        
                        matched_keywords = set()
                        
                        for keyword in keywords:
                            keyword_lower = keyword.lower()
                            if len(keyword) >= 4:  # Only meaningful keywords
                                occurrences = content_lower.count(keyword_lower)
                                if occurrences > 0:
                                    if re.search(rf'\b{re.escape(keyword_lower)}\b', content_lower):
                                        matches.append(keyword)
                                        matched_keywords.add(keyword_lower)
                                        decay = SCORING_WEIGHTS['keyword_match']['decay_factor'] ** (len(matched_keywords) - 1)
                                        confidence_metrics['keyword_matches'] += min(
                                            SCORING_WEIGHTS['keyword_match']['weight'] * decay,
                                            SCORING_WEIGHTS['keyword_match']['weight']
                                        )
                        
                        if self._is_code_file(diff.b_path):
                            confidence_metrics['code_relevance'] = SCORING_WEIGHTS['code_file']['weight']
                            for bonus_path in SCORING_WEIGHTS['code_file']['bonus_paths']:
                                if bonus_path in diff.b_path:
                                    confidence_metrics['code_relevance'] += 0.1
                                    break
                        
                        for test_path in SCORING_WEIGHTS['test_file']['test_paths']:
                            if test_path in diff.b_path:
                                confidence_metrics['test_coverage'] = SCORING_WEIGHTS['test_file']['weight']
                                break
                        
                        if any(pattern in content_lower for pattern in SCORING_WEIGHTS['error_pattern']['patterns']):
                            pattern_matches = 0
                            pattern_weights = {}
                            
                            for pattern in SCORING_WEIGHTS['error_pattern']['patterns']:
                                if pattern in content_lower:
                                    occurrences = content_lower.count(pattern)
                                    
                                    if self._pattern_in_code_context(pattern, content):
                                        pattern_weight = min(0.15 * occurrences, 0.45)
                                    else:
                                        pattern_weight = min(0.05 * occurrences, 0.15)
                                        
                                    pattern_matches += 1
                                    pattern_weights[pattern] = pattern_weight
                            
                            confidence_metrics['error_indicators'] = SCORING_WEIGHTS['error_pattern']['weight'] * \
                                                                  (1 - 1/(1 + sum(pattern_weights.values())))
                        
                        lines_changed = file_stats.get('lines', 0)
                        if lines_changed <= SCORING_WEIGHTS['change_size']['small_change']:
                            confidence_metrics['change_impact'] = SCORING_WEIGHTS['change_size']['weight']
                        elif lines_changed <= SCORING_WEIGHTS['change_size']['medium_change']:
                            confidence_metrics['change_impact'] = SCORING_WEIGHTS['change_size']['weight'] * 0.7
                        else:
                            confidence_metrics['change_impact'] = SCORING_WEIGHTS['change_size']['weight'] * \
                                                              max(0.1, 2.0 / np.log10(lines_changed))
                        
                        raw_score = sum(confidence_metrics.values())
                        
                        normalized_score = 1.0 / (1.0 + np.exp(-raw_score + 0.5))
                        
                        if normalized_score >= 0.15 or matches:
                            modified_files.append({
                                'path': diff.b_path,
                                'score': normalized_score,
                                'confidence_metrics': confidence_metrics,
                                'matches': matches,
                                'type': diff.change_type,
                                'additions': file_stats.get('insertions', 0),
                                'deletions': file_stats.get('deletions', 0),
                                'lines_changed': file_stats.get('lines', 0)
                            })
                            total_score += normalized_score
                    
                    except Exception as e:
                        self.logger.warning(f"Error analyzing file {diff.b_path}: {str(e)}")
                        continue

                if modified_files:
                    file_count_factor = 1.0 / (1.0 + 0.3 * np.log1p(file_count))
                    
                    adjusted_score = (total_score * file_count_factor) + commit_msg_bonus
                    
                    final_score = 1.0 / (1.0 + np.exp(-2 * (adjusted_score - 0.5)))
                    
                    for file in modified_files:
                        file['confidence'] = {
                            'score': file['score'],
                            'relative_impact': file['score'] / total_score if total_score > 0 else 0,
                            'metrics': file['confidence_metrics']
                        }
                    
                    return final_score, modified_files

            return 0.0, []

        except Exception as e:
            self.logger.error(f"Error analyzing commit {commit_hash}: {str(e)}")
            return 0.0, []

    def _is_code_file(self, file_path: str) -> bool:
        """
        Check if a file is likely a code file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is likely a code file, False otherwise
        """
        code_extensions = [
            '.java', '.c', '.cpp', '.h', '.hpp', '.js', '.py', '.rb', '.go', 
            '.cs', '.ts', '.sh', '.xml', '.json', '.yml', '.yaml'
        ]
        
        return any(file_path.endswith(ext) for ext in code_extensions)
    
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