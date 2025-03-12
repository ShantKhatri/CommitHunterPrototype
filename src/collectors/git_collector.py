"""
Git Data Collector Module

This module provides functionality to collect and process data from Git repositories.
"""

import git
import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class GitCollector:
    """
    A class for collecting and processing data from Git repositories.
    
    This collector interfaces with Git repositories to extract commit information,
    diffs, and other relevant metadata for commit analysis.
    """
    
    def __init__(self, repo_url: str):
        """Initialize the Git data collector."""
        self.logger = logging.getLogger(__name__)
        self.cache_dir = os.path.abspath('.git_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        repo_name = repo_url.rstrip('/').split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        
        self.repo_path = os.path.join(self.cache_dir, repo_name)
        
        try:
            if os.path.exists(self.repo_path):
                self.logger.info(f"Using existing repository at {self.repo_path}")
                self.repo = git.Repo(self.repo_path)
                
                self.logger.info("Fetching latest changes and tags...")
                origin = self.repo.remotes.origin
                origin.fetch()
                origin.fetch(tags=True)
            else:
                self.logger.info(f"Cloning repository from {repo_url}")
                self.repo = git.Repo.clone_from(
                    repo_url, 
                    self.repo_path,
                    no_single_branch=True,
                    depth=1000
                )
                origin = self.repo.remotes.origin
                origin.fetch(tags=True)
            
            self.tags = {}
            for tag in self.repo.tags:
                tag_name = str(tag)
                self.tags[tag_name] = {
                    'hash': tag.commit.hexsha,
                    'date': tag.commit.committed_datetime
                }
            
            self.sorted_tags = sorted(
                self.tags.keys(),
                key=version_key
            )
            
            self.logger.info(f"Available tags ({len(self.tags)}): {', '.join(self.sorted_tags[:5])}...")
            self.logger.info("Successfully initialized repository")
            
        except git.GitCommandError as e:
            self.logger.error(f"Git command error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error initializing repository: {str(e)}")
            raise

    def resolve_reference(self, ref: str) -> str:
        """
        Resolve a git reference (tag, branch, or commit hash) to a commit hash.
        
        Args:
            ref: Git reference to resolve
            
        Returns:
            Resolved commit hash
        """
        try:
            if ref in self.tags:
                return self.tags[ref]['hash']
            
            commit = self.repo.commit(ref)
            return commit.hexsha
            
        except (git.GitCommandError, KeyError) as e:
            self.logger.error(f"Could not resolve reference '{ref}': {str(e)}")
            raise ValueError(f"Invalid git reference: {ref}")

    def get_commit_info(self, commit_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific commit.
        
        Args:
            commit_hash: The commit hash to analyze
            
        Returns:
            Dictionary with commit details including hash, author, message, and stats
        """
        self.logger.debug(f"Getting info for commit: {commit_hash}")
        
        try:
            commit = self.repo.commit(commit_hash)
            
            info = {
                "hash": commit.hexsha,
                "short_hash": commit.hexsha[:8],
                "author_name": commit.author.name,
                "author_email": commit.author.email,
                "committer_name": commit.committer.name,
                "committer_email": commit.committer.email,
                "date": commit.committed_datetime.isoformat(),
                "message": commit.message.strip(),
                "summary": commit.summary,
                "parent_hashes": [p.hexsha for p in commit.parents],
            }
            
            stats = {
                "files_changed": {},
                "total_insertions": 0,
                "total_deletions": 0,
                "total_lines_changed": 0
            }
            
            for file, file_stats in commit.stats.files.items():
                stats["files_changed"][file] = {
                    "insertions": file_stats["insertions"],
                    "deletions": file_stats["deletions"],
                    "lines": file_stats["insertions"] + file_stats["deletions"]
                }
                stats["total_insertions"] += file_stats["insertions"]
                stats["total_deletions"] += file_stats["deletions"]
                stats["total_lines_changed"] += file_stats["insertions"] + file_stats["deletions"]
            
            info["stats"] = stats
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting info for commit {commit_hash}: {str(e)}")
            return {"hash": commit_hash, "error": str(e)}
    
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