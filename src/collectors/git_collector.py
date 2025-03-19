"""
Git Data Collector Module

This module provides functionality to collect and process data from Git repositories.
"""

import os
import re
import git
import logging
import os
import json
import subprocess
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

class GitCollector:
    """
    Enhanced Git collector with commit metadata and SZZ algorithm support.
    """
    
    def __init__(self, repo_url: str, cache_dir: str = ".cache"):
        self.logger = logging.getLogger(__name__)
        self.repo_url = repo_url
        self.cache_dir = cache_dir
        
        # Clone or open the repository
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        self.repo_path = os.path.join(cache_dir, repo_name)
        
        if not os.path.exists(self.repo_path):
            self.repo = git.Repo.clone_from(repo_url, self.repo_path)
        else:
            self.repo = git.Repo(self.repo_path)
            self.repo.git.fetch('--all')
        
        self._commit_cache = {}
        self._classification_cache = {}
        self._modified_files_cache = {}
        self._author_experience = defaultdict(int)
        self._file_history = defaultdict(list)
        
        self._build_author_experience_index()
        
    def _build_author_experience_index(self):
        """Build an index of author experience for all authors in the repo"""
        try:
            author_log = self.repo.git.shortlog('-sne', '--all')
            for line in author_log.split('\n'):
                if not line.strip():
                    continue
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    count, author = parts
                    self._author_experience[author] = int(count)
        except Exception as e:
            print(f"Warning: Could not build author experience index: {e}")
    
    def resolve_reference(self, ref: str) -> str:
        """
        Resolve a Git reference (branch, tag, partial hash) to a full commit hash.
        """
        try:
            full_hash = self.repo.git.rev_parse(ref)
            return full_hash
        except git.GitCommandError:
            try:
                matching_tags = self.repo.git.tag('--list', f'*{ref}*').splitlines()
                if matching_tags:
                    return self.repo.git.rev_parse(matching_tags[0])
            except git.GitCommandError:
                pass
            
            raise ValueError(f"Could not resolve Git reference: {ref}")
    
    def get_commit_info(self, commit_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific commit.
        """
        if commit_hash in self._commit_cache:
            return self._commit_cache[commit_hash]
        
        try:
            commit = self.repo.commit(commit_hash)
            
            # Get basic commit information
            info = {
                'hash': commit.hexsha,
                'author': f"{commit.author.name} <{commit.author.email}>",
                'author_name': commit.author.name,
                'author_email': commit.author.email,
                'authored_date': datetime.fromtimestamp(commit.authored_date),
                'committer': f"{commit.committer.name} <{commit.committer.email}>",
                'committed_date': datetime.fromtimestamp(commit.committed_date),
                'message': commit.message,
                'parents': [p.hexsha for p in commit.parents],
                'files_changed': [],
                'lines_added': 0,
                'lines_deleted': 0,
                'lines_changed': 0,
            }
            
            file_stats = {}
            for file_path, stats in commit.stats.files.items():
                file_stats[file_path] = {
                    'lines_added': stats['insertions'],
                    'lines_deleted': stats['deletions'],
                    'lines_changed': stats['insertions'] + stats['deletions'],
                }
                info['lines_added'] += stats['insertions']
                info['lines_deleted'] += stats['deletions']
                info['lines_changed'] += stats['insertions'] + stats['deletions']
                info['files_changed'].append(file_path)
            
            info['file_stats'] = file_stats
            
            info['subsystems'] = self._extract_subsystems(info['files_changed'])
            
            info['entropy'] = self._calculate_entropy(file_stats)
            
            info['author_experience'] = self._author_experience.get(info['author'], 0)
            
            info['recent_experience'] = self._get_recent_experience(
                info['author_email'], info['committed_date']
            )
            
            self._commit_cache[commit_hash] = info
            
            return info
        
        except (git.GitCommandError, ValueError) as e:
            print(f"Error getting commit info for {commit_hash}: {e}")
            return None
    
    def _extract_subsystems(self, file_paths: List[str]) -> Set[str]:
        """Extract subsystems (top-level directories) from file paths"""
        subsystems = set()
        for file_path in file_paths:
            parts = file_path.split('/')
            if len(parts) > 1:
                subsystems.add(parts[0])
            else:
                subsystems.add('root')
        return subsystems
    
    def _calculate_entropy(self, file_stats: Dict[str, Dict[str, int]]) -> float:
        """
        Calculate Shannon's entropy for code changes across files.
        Higher entropy means changes are scattered across many files.
        """
        from math import log2
        
        total_changes = sum(stats['lines_changed'] for stats in file_stats.values())
        if total_changes == 0:
            return 0.0
        
        entropy = 0
        for file_path, stats in file_stats.items():
            if stats['lines_changed'] == 0:
                continue
                
            p = stats['lines_changed'] / total_changes
            entropy -= p * log2(p)
            
        return entropy
    
    def _get_recent_experience(self, author_email: str, commit_date: datetime, 
                              months: int = 3) -> int:
        """Get the author's number of commits in the last N months before this commit"""
        from datetime import timedelta
        
        cutoff_date = commit_date - timedelta(days=30 * months)
        
        try:
            count = len(list(self.repo.iter_commits(
                author=author_email,
                until=commit_date.strftime('%Y-%m-%d'),
                since=cutoff_date.strftime('%Y-%m-%d')
            )))
            return count
        except Exception:
            return 0
    
    def get_commit_range(self, good_commit: str, bad_commit: str) -> List[str]:
        """
        Get all commits between good_commit and bad_commit.
        """
        try:
            good_hash = self.resolve_reference(good_commit)
            bad_hash = self.resolve_reference(bad_commit)
            
            commit_range = f"{good_hash}..{bad_hash}"
            commit_list = self.repo.git.rev_list(commit_range).splitlines()
            
            return list(reversed(commit_list))
        except Exception as e:
            print(f"Error getting commit range: {e}")
            return []

    def get_modified_files(self, commit_hash: str) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about files modified in the commit.
        Returns a dict with file paths as keys and modification details as values.
        """
        if commit_hash in self._modified_files_cache:
            return self._modified_files_cache[commit_hash]
        
        try:
            commit = self.repo.commit(commit_hash)
            if not commit.parents:
                return {}  # Initial commit
                
            parent = commit.parents[0]
            
            result = {}
            diffs = parent.diff(commit)
            
            for diff in diffs:
                if diff.deleted_file:
                    continue  # Skip deleted files
                    
                file_path = diff.b_path
                
                changed_lines = []
                
                try:
                    patch = self.repo.git.diff(
                        parent.hexsha, 
                        commit.hexsha, 
                        '--', 
                        file_path,
                        unified=0
                    )
                    
                    for line in patch.splitlines():
                        if line.startswith('@@'):
                            match = re.search(r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)
                            if match:
                                start_line = int(match.group(1))
                                line_count = int(match.group(2)) if match.group(2) else 1
                                changed_lines.extend(range(start_line, start_line + line_count))
                                
                except Exception as e:
                    print(f"Error parsing diff for {file_path}: {e}")
                
                result[file_path] = {
                    'status': self._get_file_status(diff),
                    'lines': changed_lines
                }
            
            # Store in cache
            self._modified_files_cache[commit_hash] = result
            
            return result
            
        except Exception as e:
            print(f"Error getting modified files for {commit_hash}: {e}")
            return {}
    
    def _get_file_status(self, diff):
        """Determine file status from diff object"""
        if diff.new_file:
            return 'added'
        elif diff.deleted_file:
            return 'deleted'
        elif diff.renamed:
            return 'renamed'
        else:
            return 'modified'
            
    def get_blame_info(self, commit_hash: str, file_path: str, 
                      line_numbers: List[int]) -> Dict[str, Any]:
        """
        Get blame information for specific lines in a file at a specific commit.
        Used by the SZZ algorithm to trace bug-inducing changes.
        """
        result = {
            'commits': [],
            'lines': {}
        }
        
        if not line_numbers:
            return result
            
        try:
            blame_output = self.repo.git.blame(
                '-l',  # Show long commit hashes
                '-w',  # Ignore whitespace changes
                f'{commit_hash}^',  # Use parent of the commit
                '--',
                file_path
            )
            
            current_commit = None
            current_line_num = None
            
            for line in blame_output.splitlines():
                hash_match = re.match(r'^([a-f0-9]{40})', line)
                if hash_match:
                    current_commit = hash_match.group(1)
                    line_match = re.search(r'\s+(\d+)\)', line)
                    if line_match:
                        current_line_num = int(line_match.group(1))
                        
                        if current_line_num in line_numbers:
                            if current_commit not in result['commits']:
                                result['commits'].append(current_commit)
                            
                            result['lines'][current_line_num] = {
                                'commit': current_commit,
                                'content': line[line.find(')')+1:].strip()
                            }
            
            return result
            
        except Exception as e:
            print(f"Error getting blame info for {file_path} at {commit_hash}: {e}")
            return result
    
    def classify_commit(self, commit_hash: str) -> str:
        """
        Classify commit based on its message using Commit Guru's classification scheme.
        Returns: 'corrective', 'feature_addition', 'merge', 'non_functional', 'perfective', or 'preventive'
        """
        if commit_hash in self._classification_cache:
            return self._classification_cache[commit_hash]
            
        commit_info = self.get_commit_info(commit_hash)
        if not commit_info:
            return 'unknown'
            
        message = commit_info.get('message', '').lower()
        
        categories = {
            'corrective': ['bug', 'fix', 'wrong', 'error', 'fail', 'problem', 'patch'],
            'feature_addition': ['new', 'add', 'requirement', 'initial', 'create'],
            'merge': ['merge'],
            'non_functional': ['doc', 'documentation', 'comment', 'license'],
            'perfective': ['clean', 'better', 'improve', 'enhance', 'refactor'],
            'preventive': ['test', 'junit', 'coverage', 'assert']
        }
        
        for category, keywords in categories.items():
            if any(keyword in message for keyword in keywords):
                self._classification_cache[commit_hash] = category
                return category
                
        self._classification_cache[commit_hash] = 'unknown'
        return 'unknown'
    
    def get_commits_between(self, start_ref: str, end_ref: str) -> List[Dict[str, Any]]:
        """Get all commits between two references."""
        self.logger.info(f"Getting commits between {start_ref} and {end_ref}")
        
        try:
            start_hash = self.resolve_reference(start_ref)
            end_hash = self.resolve_reference(end_ref)
            
            start_commit = self.repo.commit(start_hash)
            end_commit = self.repo.commit(end_hash)
            
            if start_commit.committed_date > end_commit.committed_date:
                start_hash, end_hash = end_hash, start_hash
                
            self.logger.info(f"Analyzing commits from {start_hash} to {end_hash}")
            
            commits = []
            for commit in self.repo.iter_commits(f'{start_hash}..{end_hash}'):
                commits.append({
                    'hash': commit.hexsha,
                    'short_hash': commit.hexsha[:8],
                    'message': commit.message.strip(),
                    'author': commit.author.name,
                    'date': commit.authored_datetime.isoformat(),
                    'stats': {
                        'files_changed': len(commit.stats.files),
                        'insertions': commit.stats.total['insertions'],
                        'deletions': commit.stats.total['deletions']
                    }
                })
            
            self.logger.info(f"Found {len(commits)} commits")
            return commits
            
        except Exception as e:
            self.logger.error(f"Error getting commits: {str(e)}")
            return []


    def get_file_at_commit(self, file_path: str, commit_hash: str) -> Optional[str]:
        """
        Get the contents of a file at a specific commit.
        
        Args:
            file_path: Path to the file within the repository
            commit_hash: The commit hash to retrieve the file from
            
        Returns:
            Contents of the file as a string, or None if file doesn't exist
        """
        self.logger.debug(f"Getting file {file_path} at commit {commit_hash}")
        
        try:
            commit = self.repo.commit(commit_hash)
            blob = commit.tree / file_path
            return blob.data_stream.read().decode('utf-8')
        except KeyError:
            self.logger.warning(f"File {file_path} does not exist at commit {commit_hash}")
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving file {file_path} at commit {commit_hash}: {str(e)}")
            return None

    def get_diff_between_commits(self, old_commit: str, new_commit: str, file_path: Optional[str] = None) -> str:
        """
        Get the diff between two commits for a specific file or the entire repository.
        
        Args:
            old_commit: The older commit hash
            new_commit: The newer commit hash
            file_path: Optional path to a specific file to get diff for
            
        Returns:
            Git diff as a string
        """
        import subprocess
        
        file_path_arg = f" -- {file_path}" if file_path else ""
        diff_cmd = f"cd {self.repo_path} && git diff {old_commit}..{new_commit}{file_path_arg}"
        
        try:
            result = subprocess.run(diff_cmd, shell=True, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error getting diff: {str(e)}")
            return f"Error getting diff: {e.stderr}"
        except Exception as e:
            self.logger.error(f"Error getting diff between {old_commit} and {new_commit}: {str(e)}")
            return f"Error getting diff: {str(e)}"

    def get_files_modified_in_commit(self, commit_hash: str) -> List[str]:
        """
        Get list of files modified in a commit.
        
        Args:
            commit_hash: The commit hash to analyze
            
        Returns:
            List of modified file paths
        """
        try:
            commit = self.repo.commit(commit_hash)
            modified_files = []
            
            if commit.parents:
                diffs = commit.parents[0].diff(commit)
                
                for diff in diffs:
                    if diff.a_path:
                        modified_files.append(diff.a_path)
                    if diff.b_path and diff.b_path not in modified_files:
                        modified_files.append(diff.b_path)
            else:
                modified_files = [item.path for item in commit.tree.traverse()]
                
            self.logger.debug(f"Found {len(modified_files)} modified files in commit {commit_hash[:8]}")
            return modified_files
            
        except Exception as e:
            self.logger.error(f"Error getting modified files for commit {commit_hash}: {str(e)}")
            return []


    def checkout_commit(self, commit_hash: str) -> bool:
        """
        Checkout a specific commit.
        
        Args:
            commit_hash: The commit hash to checkout
            
        Returns:
            True if checkout was successful, False otherwise
        """
        self.logger.info(f"Checking out commit {commit_hash}")
        
        try:
            self.repo.git.clean('-fd')
            self.repo.git.reset('--hard')
            
            commit = self.repo.commit(commit_hash)
            
            self.repo.git.checkout(commit.hexsha)
            self.logger.info(f"Successfully checked out commit {commit_hash}")
            return True
            
        except git.GitCommandError as e:
            self.logger.error(f"Error checking out commit {commit_hash}: {str(e)}")
            return False

    def export_commit_data(self, output_path: str, commit_hash: str) -> bool:
        """
        Export detailed data about a commit to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
            commit_hash: The commit hash to export data for
            
        Returns:
            True if export was successful, False otherwise
        """
        self.logger.info(f"Exporting data for commit {commit_hash}")
        
        try:
            commit_data = self.get_commit_info(commit_hash)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(commit_data, f, indent=2)
                
            self.logger.info(f"Successfully exported commit data to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting commit data: {str(e)}")
            return False
        
    def generate_report(results: Dict[str, Any], format: str = 'json') -> str:
        """Generate a formatted report of the analysis results."""
        if format == 'json':
            return json.dumps(results, indent=2)
        
        elif format == 'html':
            html = ["<html><body>",
                    "<h1>CommitHunter Analysis Report</h1>"]
            
            html.extend([
                "<h2>Analysis Information</h2>",
                f"<p>Start Time: {results.get('metadata', {}).get('start_time', 'N/A')}</p>",
                f"<p>Good Commit: {results.get('metadata', {}).get('good_commit', 'N/A')}</p>",
                f"<p>Bad Commit: {results.get('metadata', {}).get('bad_commit', 'N/A')}</p>"
            ])
            
            for analyzer in results.get('analyzers', []):
                html.extend([
                    f"<h2>{analyzer.get('name', 'Unknown')} Results</h2>",
                    f"<p>Status: {analyzer.get('status', 'Unknown')}</p>"
                ])
                
                if 'duration' in analyzer:
                    html.append(f"<p>Duration: {analyzer['duration']:.2f} seconds</p>")
                
                if analyzer.get('status') == 'success':
                    results_data = analyzer.get('results', {})
                    if isinstance(results_data.get('summary'), str):
                        html.append(f"<pre>{results_data['summary']}</pre>")
                    else:
                        html.append(f"<pre>{json.dumps(results_data, indent=2)}</pre>")
                else:
                    html.append(f"<p>Error: {analyzer.get('error', 'Unknown error')}</p>")
            
            html.append("</body></html>")
            return "\n".join(html)

    def get_diff_stats(self, commit1: str, commit2: str) -> Dict[str, Any]:
        """
        Get statistics about the differences between two commits.
        
        Args:
            commit1: First commit hash
            commit2: Second commit hash
            
        Returns:
            Dictionary with diff statistics
        """
        try:
            c1 = self.repo.commit(commit1)
            c2 = self.repo.commit(commit2)
            
            diff = self.repo.git.diff(commit1, commit2, name_status=True)
            
            files_changed = []
            for line in diff.splitlines():
                parts = line.split('\t')
                if len(parts) >= 2:
                    change_type, file_path = parts[0], parts[1]
                    files_changed.append(file_path)
            
            stats = self.repo.git.diff(commit1, commit2, shortstat=True)
            
            files = int(re.search(r'(\d+) files? changed', stats).group(1)) if re.search(r'(\d+) files? changed', stats) else 0
            insertions = int(re.search(r'(\d+) insertions?', stats).group(1)) if re.search(r'(\d+) insertions?', stats) else 0
            deletions = int(re.search(r'(\d+) deletions?', stats).group(1)) if re.search(r'(\d+) deletions?', stats) else 0
            
            return {
                "files_changed": files_changed,
                "stats": {
                    "files": files,
                    "insertions": insertions,
                    "deletions": deletions,
                    "lines": insertions + deletions
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting diff stats between {commit1} and {commit2}: {str(e)}")
            return {"files_changed": [], "stats": {"files": 0, "insertions": 0, "deletions": 0, "lines": 0}}

    def get_nearby_commits(self, commit_hash: str, count: int = 20) -> Dict[str, str]:
        """
        Get commits before and after the specified commit.
        
        Args:
            commit_hash: The reference commit hash
            count: Number of commits to get in each direction
            
        Returns:
            Dictionary with 'before' and 'after' commit hashes
        """
        try:
            result_before = subprocess.run(
                f'git log --pretty=format:"%H" -n {count} {commit_hash}^',
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            result_after = subprocess.run(
                f'git log --pretty=format:"%H" -n {count+1} HEAD',
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            
            before_commits = result_before.stdout.strip().split('\n')
            after_commits = result_after.stdout.strip().split('\n')
            
            try:
                commit_idx = after_commits.index(commit_hash)
                after_commits = after_commits[:commit_idx]  # Only keep commits newer than our target
            except ValueError:
                pass
            
            before_commit = before_commits[-1] if before_commits and before_commits[-1] else None
            after_commit = after_commits[0] if after_commits else None
            
            return {
                'before': before_commit,
                'after': after_commit,
                'before_count': len(before_commits),
                'after_count': len(after_commits)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting nearby commits: {str(e)}")
            return {'before': None, 'after': None}

    def version_key(tag: str) -> tuple:
        """
        Convert version string to comparable tuple.
        Handles special cases like 'M1', 'RC1' etc.
        """
        try:
            version = tag.split('-')[-1] if '-' in tag else tag
            
            components = []
            current_num = ''
            current_str = ''
            
            for char in version:
                if char.isdigit():
                    if current_str:
                        components.append(current_str)
                        current_str = ''
                    current_num += char
                elif char == '.':
                    if current_num:
                        components.append(int(current_num))
                        current_num = ''
                    elif current_str:
                        components.append(current_str)
                        current_str = ''
                else:
                    if current_num:
                        components.append(int(current_num))
                        current_num = ''
                    current_str += char
            
            if current_num:
                components.append(int(current_num))
            if current_str:
                components.append(current_str)
            
            normalized = []
            for comp in components:
                if isinstance(comp, int):
                    normalized.extend([0, comp])
                else:
                    if comp.startswith('M'):
                        normalized.extend([-2, int(comp[1:]) if comp[1:].isdigit() else 0])
                    elif comp.startswith('RC'):
                        normalized.extend([-1, int(comp[2:]) if comp[2:].isdigit() else 0])
                    else:
                        normalized.extend([1, 0])  # Other strings sort after numbers
            
            while len(normalized) < 8:  # Support up to 4 version components
                normalized.extend([0, 0])
                
            return tuple(normalized)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error parsing version {tag}: {str(e)}")
            return (float('inf'),) * 8  # Sort problematic versions to the end