"""
CommitHunter Main Module

This is the main entry point for the CommitHunter application.
"""

import argparse
import sys
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

from collectors.git_collector import GitCollector
from collectors.test_collector import TestCollector
from analyzers.string_matcher import StringMatcher
from analyzers.binary_search import BinarySearchAnalyzer
from analyzers.perf_analyzer import PerformanceAnalyzer
from utils.config import Config, init_config
from utils.logging import setup_logging, get_logger, PerformanceLogger

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CommitHunter: AI-Powered Commit Debugger',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--repo', required=True, 
                       help='Path to Git repository or repository URL')
    parser.add_argument('--good', required=True, 
                       help='"Good" commit hash or tag')
    parser.add_argument('--bad', required=True, 
                       help='"Bad" commit hash or tag')
    parser.add_argument('--test-name', 
                        help='Specific test name to analyze (required for binary search)')
    
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test-results', 
                       help='Path to test results directory')
    parser.add_argument('--output', default='reports/results.json',
                       help='Path to output file')
    parser.add_argument('--analyzer', 
                       choices=['string', 'binary', 'performance', 'all'],
                       default='all', 
                       help='Analyzer to use')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--report-format',
                       choices=['json', 'html', 'text'],
                       default='json',
                       help='Output report format')
    
    return parser.parse_args()

def init_analyzers(config: Config, git_collector: GitCollector) -> List[Any]:
    """
    Initialize the appropriate analyzers based on configuration.
    
    Args:
        config: Configuration instance
        git_collector: Initialized GitCollector instance
        
    Returns:
        List of initialized analyzers
    """
    analyzers = []
    analyzer_configs = config.config['analyzers']
    
    if analyzer_configs['string_matcher']['enabled']:
        analyzers.append(StringMatcher(git_collector))
    
    if analyzer_configs['binary_search']['enabled']:
        analyzers.append(BinarySearchAnalyzer(git_collector, analyzer_configs['binary_search']))
    
    if analyzer_configs['performance']['enabled']:
        analyzers.append(PerformanceAnalyzer(git_collector, analyzer_configs['performance']))
    
    return analyzers

def analyze_commits(git_collector: GitCollector, 
                   test_collector: TestCollector,
                   analyzers: List[Any],
                   good_commit: str,
                   bad_commit: str,
                   test_name: Optional[str] = None,
                   config: Config = None) -> Dict[str, Any]:
    """
    Analyze commits to find problematic ones using a sequential approach.
    """
    logger = logging.getLogger(__name__)
    
    results = {
        "metadata": {
            "start_time": datetime.now().isoformat(),
            "repository": git_collector.repo_path,
            "good_commit": good_commit,
            "bad_commit": bad_commit,
            "test_name": test_name
        },
        "analyzers": []
    }
    
    error_message = None
    if test_collector:
        test_results = test_collector.get_test_results()
        if test_results:
            logger.info(f"Found test results: {json.dumps(test_results, indent=2)}")
            error_message = test_results.get('error_message', '')
            if not error_message:
                details = test_results.get('details', {})
                error_message = details.get('error_message', '')
        else:
            logger.warning("No test results found")
            logger.info("Attempting to collect all test results")
            all_results = test_collector.collect_test_results()
            logger.info(f"Found {len(all_results)} test result sets")
            if all_results:
                failure_info = test_collector.extract_failure_messages(all_results)
                if failure_info:
                    logger.info(f"Failure info: {json.dumps(failure_info, indent=2)}")
                    if 'error_messages' in failure_info and failure_info['error_messages']:
                        error_message = failure_info['error_messages'][0]
                        logger.info(f"Using extracted error message: {error_message}")
        
    binary_analyzer = next((a for a in analyzers if isinstance(a, BinarySearchAnalyzer)), None)
    string_matcher = next((a for a in analyzers if isinstance(a, StringMatcher)), None)
    
    first_bad_commit = None
    
    if binary_analyzer and test_name:
        logger.info("Running binary search analysis first")
        start_time = time.time()
        
        def test_runner(commit: str, options: Dict[str, Any] = None) -> bool:
            options = options or {}
            test_mode = options.get('performance_test', 'full')
            
            logger.info(f"Running test '{test_name}' on commit {commit[:8]} (mode: {test_mode})")
            
            git_collector.checkout_commit(commit)
            
            if test_collector:
                try:
                    test_result = test_collector.run_test(test_name, mode=test_mode)
                    
                    if test_result:
                        passed = test_result.get("status") == "passed"
                        logger.info(f"Test '{test_name}' {'passed' if passed else 'failed'} on commit {commit[:8]}")
                        
                        if not passed and not error_message and test_result.get("error_message"):
                            error_message = test_result.get("error_message")
                            logger.info(f"Captured error message: {error_message[:100]}...")
                            
                        return passed
                    else:
                        logger.warning(f"No result for test '{test_name}' on commit {commit[:8]}")
                        return False
                except Exception as e:
                    logger.error(f"Error running test '{test_name}' on commit {commit[:8]}: {str(e)}")
                    return False
            
            logger.warning(f"No test collector available to run test '{test_name}'")
            return False
        
        binary_results = binary_analyzer.find_problematic_commit(
            good_commit, bad_commit, test_runner, test_name
        )
        
        results["analyzers"].append({
            "name": "BinarySearchAnalyzer",
            "status": "success",
            "results": binary_results,
            "duration": time.time() - start_time
        })
        
        if binary_results.get("status") == "success" and "first_bad_commit" in binary_results:
            first_bad_commit = binary_results["first_bad_commit"]["hash"]
            logger.info(f"Binary search found problematic commit: {first_bad_commit[:8]}")
            
            if test_collector:
                test_result = test_collector.get_test_result_for_commit(first_bad_commit)
                if test_result and test_result.get("error_message"):
                    error_message = test_result.get("error_message")
                    logger.info(f"Using error message from problematic commit: {error_message[:100]}...")
    
    if string_matcher:
        if not error_message:
            logger.warning("No error message available for string matcher")
            error_message = "Unknown error"  # Fallback
        
        logger.info("Running string matcher analysis")
        start_time = time.time()
        
        if first_bad_commit:
            nearby_commits = git_collector.get_nearby_commits(first_bad_commit, 20)
            
            if nearby_commits.get('before') and nearby_commits.get('after'):
                good_commit_for_matcher = nearby_commits.get('before')
                bad_commit_for_matcher = nearby_commits.get('after')
                
                logger.info(f"Focusing string matcher on commits around {first_bad_commit[:8]}")
                logger.info(f"Narrowed range: {good_commit_for_matcher[:8]} to {bad_commit_for_matcher[:8]}")
                
                matcher_results = string_matcher.find_suspicious_commits(
                    good_commit=good_commit_for_matcher,
                    bad_commit=bad_commit_for_matcher,
                    error_message=error_message,
                    threshold=config.config.get('score_threshold', 0.95) if config else 0.6,
                    context={"problematic_commit": first_bad_commit}  # Add context
                )
            else:
                logger.info(f"Using problematic commit as focus for string matcher: {first_bad_commit[:8]}")
                
                matcher_results = string_matcher.find_suspicious_commits(
                    good_commit=good_commit,
                    bad_commit=first_bad_commit,
                    error_message=error_message,
                    threshold=config.config.get('score_threshold', 0.95) if config else 0.6,
                    context={"problematic_commit": first_bad_commit}  # Add context
                )
        else:
            logger.info("No problematic commit identified, using full commit range for string matcher")
            
            matcher_results = string_matcher.find_suspicious_commits(
                good_commit=good_commit,
                bad_commit=bad_commit,
                error_message=error_message,
                threshold=config.config.get('score_threshold', 0.95) if config else 0.6
            )
        
        results["analyzers"].append({
            "name": "StringMatcher",
            "status": "success",
            "results": {
                "suspicious_commits": matcher_results,
                "has_matches": len(matcher_results) > 0,
                "error_message": error_message[:200] + "..." if error_message and len(error_message) > 200 else error_message
            },
            "duration": time.time() - start_time
        })
    
    for analyzer in analyzers:
        if not isinstance(analyzer, BinarySearchAnalyzer) and not isinstance(analyzer, StringMatcher):
            # [existing code for other analyzers]
            pass
    
    results["metadata"]["end_time"] = datetime.now().isoformat()
    results["metadata"]["duration"] = (datetime.fromisoformat(results["metadata"]["end_time"]) - 
                                     datetime.fromisoformat(results["metadata"]["start_time"])).total_seconds()
                                     
    return results

def generate_binary_search_report(analyzer_results: Dict[str, Any]) -> List[str]:
    """Generate HTML report section for binary search results."""
    lines = ['<div class="binary-search">']
    
    if analyzer_results["status"] == "success":
        bad_commit = analyzer_results["first_bad_commit"]
        lines.extend([
            '<h3>Binary Search Results</h3>',
            '<div class="result-summary">',
            f'<p><strong>Status:</strong> Found problematic commit</p>',
            f'<p><strong>Commits Analyzed:</strong> {analyzer_results["commits_tested"]}/{analyzer_results["total_commits"]}</p>',
            '</div>',
            
            '<div class="bad-commit">',
            '<h4>Problematic Commit Details</h4>',
            f'<p><strong>Hash:</strong> {bad_commit["hash"]}</p>',
            f'<p><strong>Author:</strong> {bad_commit["author"]}</p>',
            f'<p><strong>Date:</strong> {bad_commit["date"]}</p>',
            f'<p><strong>Message:</strong></p>',
            f'<pre class="commit-message">{bad_commit["message"]}</pre>',
            
            '<h5>Modified Files:</h5>',
            '<ul class="modified-files">'
        ])
        
        for file in bad_commit["modified_files"]:
            lines.append(f'<li>{file}</li>')
        
        lines.extend(['</ul>', '</div>'])
        
        if analyzer_results.get("search_history"):
            lines.extend([
                '<div class="search-history">',
                '<h4>Search History</h4>',
                '<table>',
                '<tr><th>Commit</th><th>Attempt</th><th>Result</th></tr>'
            ])
            
            for entry in analyzer_results["search_history"]:
                result_class = "pass" if entry["result"] else "fail"
                lines.append(
                    f'<tr class="{result_class}"><td>{entry["commit"][:8]}</td>'
                    f'<td>{entry["attempt"]}</td><td>{result_class}</td></tr>'
                )
            
            lines.extend(['</table>', '</div>'])
        
        if analyzer_results.get("performance_impact", {}).get("has_impact"):
            perf = analyzer_results["performance_impact"]
            lines.extend([
                '<div class="performance-impact">',
                '<h4>Performance Impact</h4>',
                f'<p><strong>Recommendation:</strong> {perf["recommendation"]}</p>',
                '</div>'
            ])
    
    else:
        lines.extend([
            '<h3>Binary Search Results</h3>',
            '<div class="error-message">',
            f'<p>Error: {analyzer_results["message"]}</p>',
            f'<p>Commits tested: {analyzer_results["commits_tested"]}</p>',
            '</div>'
        ])
    
    lines.append('</div>')
    return lines

def get_binary_search_css() -> str:
    """Get CSS styles for binary search results."""
    return """
    .binary-search {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
    }
    
    .result-summary {
        background: #e9ecef;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    
    .bad-commit {
        border-left: 4px solid #dc3545;
        padding-left: 15px;
        margin: 15px 0;
    }
    
    .commit-message {
        background: #fff;
        padding: 10px;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        white-space: pre-wrap;
    }
    
    .modified-files {
        list-style: none;
        padding-left: 0;
    }
    
    .modified-files li {
        padding: 5px 0;
        border-bottom: 1px solid #dee2e6;
    }
    
    .search-history table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
    }
    
    .search-history th,
    .search-history td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #dee2e6;
    }
    
    .search-history .pass { color: #28a745; }
    .search-history .fail { color: #dc3545; }
    
    .performance-impact {
        background: #fff3cd;
        padding: 15px;
        border-radius: 4px;
        margin-top: 20px;
    }
    """

def generate_performance_report(analyzer_results: Dict[str, Any]) -> List[str]:
    """Generate HTML report section for performance analysis."""
    SUCCESS_SYMBOL = "&#x2705;"  # ✅
    WARNING_SYMBOL = "&#x26A0;"  # ⚠️
    UP_ARROW = "&#x25B2;"       # ▲
    DOWN_ARROW = "&#x25BC;"     # ▼
    lines = ['<div class="performance-analysis">']
    
    if analyzer_results.get("error"):
        lines.extend([
            '<div class="error-message">',
            f'<p>Error: {analyzer_results["error"]}</p>',
            '</div>'
        ])
        return lines

    status_class = "regression" if analyzer_results["regression_found"] else "success"
    status_symbol = WARNING_SYMBOL if analyzer_results["regression_found"] else SUCCESS_SYMBOL
    lines.extend([
        f'<div class="summary {status_class}">',
        f'<h3>Performance Analysis Summary</h3>',
        f'<pre>{analyzer_results["summary"].replace("✅", SUCCESS_SYMBOL).replace("⚠️", WARNING_SYMBOL)}</pre>',
        '</div>'
    ])

    lines.extend([
        '<div class="metrics-comparison">',
        '<h3>Metrics Comparison</h3>',
        '<table>',
        '<tr>',
        '<th>Metric</th>',
        '<th>Good Build</th>',
        '<th>Bad Build</th>',
        '<th>Change</th>',
        '<th>Status</th>',
        '</tr>'
    ])

    for metric, data in analyzer_results["metrics_comparison"].items():
        status_class = "regression" if data["is_regression"] else "normal"
        change = data["change_percentage"]
        change_symbol = DOWN_ARROW if change < 0 else UP_ARROW
        
        lines.append(
            f'<tr class="{status_class}">'
            f'<td>{metric}</td>'
            f'<td>{data["good_value"]:.2f}</td>'
            f'<td>{data["bad_value"]:.2f}</td>'
            f'<td>{change_symbol} {abs(change):.1f}%</td>'
            f'<td>{status_class.upper()}</td>'
            '</tr>'
        )

    lines.append('</table></div>')

    if analyzer_results.get("charts"):
        lines.extend(['<div class="performance-charts"><h3>Performance Charts</h3>'])
        for metric, chart_data in analyzer_results["charts"].items():
            if chart_data:
                lines.extend([
                    f'<div class="chart-container">',
                    f'<h4>{metric}</h4>',
                    f'<img src="data:image/png;base64,{chart_data}" alt="{metric} chart">',
                    '</div>'
                ])
        lines.append('</div>')

    lines.append('</div>')
    return lines

def generate_report(results: Dict[str, Any], format: str = 'json') -> str:
    """Generate analysis report."""
    if format == 'json':
        return json.dumps(results, indent=2)
    
    elif format == 'html':
        lines = ['<html><head>', '<title>CommitHunter Analysis Report</title>']
        
        lines.append('''
        <style>
            /* Base styles */
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                color: #333;
                line-height: 1.6;
            }
            code { font-family: Consolas, monospace; }
            pre { 
                white-space: pre-wrap; 
                background-color: #f5f5f5; 
                padding: 10px; 
                border-radius: 4px; 
                overflow: auto;
                max-height: 400px;
            }
            .error { color: #d9534f; }
            h1, h2, h3, h4, h5, h6 { 
                margin-top: 1em;
                margin-bottom: 0.5em;
                font-weight: 600;
                color: #333;
            }
            h4 { margin: 5px 0; }
            ul { margin: 5px 0; }
            
            /* Table styling */
            .commit-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            
            .commit-table th {
                background-color: #f2f2f2;
                text-align: left;
                padding: 8px;
                border: 1px solid #ddd;
            }
            
            .commit-table td {
                padding: 8px;
                border: 1px solid #ddd;
                vertical-align: top;
            }
            
            .files-container {
                background-color: #f9f9f9;
            }
            
            .file-list {
                list-style: none;
                padding: 0 15px;
                margin: 5px 0;
                max-height: 300px;
                overflow-y: auto;
                display: block !important;
            }
            
            .file-list li {
                padding: 3px 0;
                margin: 2px 0;
                border-bottom: 1px solid #eee;
                display: block !important;
            }
            
            .file-list li code {
                display: inline-block;
                word-break: break-all;
            }
            
            .score {
                color: #0066cc;
                font-weight: bold;
            }
            
            .changes {
                float: right;
                font-family: monospace;
            }
            
            .additions {
                color: green;
                margin-right: 5px;
            }
            
            .deletions {
                color: red;
            }
            
            .error-message-box {
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 10px;
                margin: 15px 0;
            }
            
            /* Binary search styles */
            ''' + get_binary_search_css() + '''
        </style>
        ''')
        
        lines.append('</head><body>')
        lines.append('<h1>CommitHunter Analysis Report</h1>')
        
        lines.extend([
            '<h2>Analysis Parameters</h2>',
            '<ul>',
            f'<li>Good Commit: {results["metadata"]["good_commit"]}</li>',
            f'<li>Bad Commit: {results["metadata"]["bad_commit"]}</li>',
            f'<li>Test Name: {results["metadata"].get("test_name", "N/A")}</li>',
            '</ul>'
        ])
        
        for analyzer in results["analyzers"]:
            lines.extend([
                f'<h2>{analyzer["name"]} Results</h2>',
                f'<p>Status: {analyzer["status"]}</p>',
                f'<p>Duration: {analyzer.get("duration", 0):.2f} seconds</p>'
            ])
            
            if analyzer["status"] == "success":
                if analyzer["name"] == "StringMatcher":
                    lines.append('<h3>Suspicious Commits</h3>')
                    
                    if isinstance(analyzer.get("results"), dict) and "error_message" in analyzer["results"]:
                        lines.append('<div class="error-message-box">')
                        lines.append('<h4>Error Message Used for Analysis:</h4>')
                        lines.append(f'<pre>{analyzer["results"]["error_message"]}</pre>')
                        lines.append('</div>')
                    
                    if isinstance(analyzer.get("results"), dict):
                        suspicious_commits = analyzer["results"].get("suspicious_commits", [])
                    elif isinstance(analyzer.get("results"), list):
                        suspicious_commits = analyzer["results"]
                    else:
                        suspicious_commits = []
                    
                    if suspicious_commits:
                        lines.append('<table class="commit-table">')
                        lines.append('<tr><th>Commit</th><th>Score</th><th>Author</th><th>Message</th></tr>')
                        
                        for commit in suspicious_commits:
                            lines.extend([
                                '<tr>',
                                f'<td><code>{commit["hash"][:8]}</code></td>',
                                f'<td>{commit["score"]:.2f}</td>',
                                f'<td>{commit.get("author", "Unknown")}</td>',
                                f'<td>{commit.get("message", "")[:100]}...</td>',
                                '</tr>',
                                '<tr>',
                                '<td colspan="4" class="files-container">',
                                '<h5>Modified Files:</h5>',
                                '<ul class="file-list">'
                            ])
                            
                            files_changed = commit.get("files_changed", [])
                            if not isinstance(files_changed, list):
                                files_changed = commit.get("modified_files", [])
                            
                            lines.append(f'<!-- Found {len(files_changed)} modified files -->')
                            
                            for file in files_changed:
                                if isinstance(file, dict):
                                    file_path = file.get("path", "Unknown")
                                    score = file.get("score", 0)
                                    additions = file.get("additions", 0)
                                    deletions = file.get("deletions", 0)
                                    
                                    lines.append(
                                        f'<li><code>{file_path}</code> '
                                        f'<span class="score">(Score: {score:.2f})</span> '
                                        f'<span class="changes">'
                                        f'<span class="additions">+{additions}</span>'
                                        f'<span class="deletions">-{deletions}</span>'
                                        f'</span></li>'
                                    )
                                else:
                                    lines.append(f'<li><code>{file}</code></li>')
                            
                            lines.extend(['</ul>', '</td>', '</tr>'])
                        
                        lines.append('</table>')
                    else:
                        lines.append('<p>No suspicious commits found.</p>')
                        
                elif analyzer["name"] == "BinarySearchAnalyzer":
                    if analyzer.get("results"):
                        lines.extend(generate_binary_search_report(analyzer["results"]))
                
                elif analyzer["name"] == "PerformanceAnalyzer":
                    if analyzer.get("results"):
                        lines.extend(generate_performance_report(analyzer["results"]))
                
                binary_results = next((a for a in results["analyzers"] if a["name"] == "BinarySearchAnalyzer"), None)
                string_results = next((a for a in results["analyzers"] if a["name"] == "StringMatcher"), None)
                
                if binary_results and string_results and binary_results.get("status") == "success" and string_results.get("status") == "success":
                    lines.append('<div class="integrated-analysis">')
                    lines.append('<h2>Integrated Analysis</h2>')
                    
                    bad_commit = binary_results["results"].get("first_bad_commit", {})
                    bad_commit_hash = bad_commit.get("hash", "")
                    
                    suspicious_commits = string_results["results"].get("suspicious_commits", [])
                    
                    lines.append('<p>This analysis combines binary search results with string matching:</p>')
                    lines.append('<ul>')
                    lines.append(f'<li><strong>Binary search</strong> identified commit {bad_commit_hash[:8]} as the first bad commit</li>')
                    
                    if suspicious_commits:
                        lines.append(f'<li><strong>String matching</strong> found {len(suspicious_commits)} commits related to the error</li>')
                        
                        problematic_in_suspicious = any(c.get("hash") == bad_commit_hash for c in suspicious_commits)
                        if problematic_in_suspicious:
                            lines.append('<li><strong>Verification:</strong> The problematic commit was also found by string matching</li>')
                        else:
                            lines.append('<li><strong>Note:</strong> The problematic commit was not in the top string matches</li>')
                    else:
                        lines.append('<li><strong>String matching</strong> did not find relevant commits</li>')
                        
                    lines.append('</ul>')
                    
                    lines.append('</div>')
            
            elif analyzer["status"] == "error":
                lines.append(f'<p class="error">Error: {analyzer.get("error", "Unknown error")}</p>')
        
        lines.extend(['</body></html>'])
        return '\n'.join(lines)
    
    else:  
        lines = ["CommitHunter Analysis Report",
                "=========================",
                f"Start Time: {results['metadata']['start_time']}",
                f"Good Commit: {results['metadata']['good_commit']}",
                f"Bad Commit: {results['metadata']['bad_commit']}",
                ""]
        
        for analyzer in results['analyzers']:
            lines.extend([
                f"{analyzer['name']} Results",
                "-" * (len(analyzer['name']) + 8),
                f"Status: {analyzer['status']}",
                f"Duration: {analyzer.get('duration', 0):.2f} seconds",
                ""
            ])
            
            if analyzer['status'] == 'success':
                if isinstance(analyzer.get('results', {}), dict) and isinstance(analyzer['results'].get('summary'), str):
                    lines.append(analyzer['results']['summary'])
                else:
                    lines.append(json.dumps(analyzer['results'], indent=2))
            else:
                lines.append(f"Error: {analyzer.get('error', 'Unknown error')}")
            
            lines.append("")
        
        return "\n".join(lines)

def create_sample_test_results():
    """Create a sample test result file that can be parsed by TestCollector."""
    import xml.etree.ElementTree as ET
    import os
    
    testsuites = ET.Element("testsuites")
    testsuite = ET.SubElement(testsuites, "testsuite", {
        "name": "SampleTestSuite",
        "tests": "3",
        "failures": "1",
        "errors": "1",
        "time": "0.5",
        "timestamp": datetime.now().isoformat()
    })
    
    testcase1 = ET.SubElement(testsuite, "testcase", {
        "name": "testSuccess",
        "classname": "com.example.TestClass",
        "time": "0.1"
    })
    
    testcase2 = ET.SubElement(testsuite, "testcase", {
        "name": "testFailure",
        "classname": "com.example.TestClass",
        "time": "0.2"
    })
    failure = ET.SubElement(testcase2, "failure", {
        "type": "AssertionError",
        "message": "Expected value was 42 but got null"
    })
    failure.text = """
    at com.example.TestClass.testFailure(TestClass.java:123)
    at java.base/java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1130)
    at java.base/java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:630)
    """
    
    testcase3 = ET.SubElement(testsuite, "testcase", {
        "name": "testError",
        "classname": "org.openj9.TestProfiler",
        "time": "0.2"
    })
    error = ET.SubElement(testcase3, "error", {
        "type": "NullPointerException",
        "message": "Cannot invoke \"org.openj9.IProfiler.getResult()\" because \"profiler\" is null"
    })
    error.text = """
    at org.openj9.TestProfiler.testMethod(TestProfiler.java:45)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
    at java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:77)
    """
    
    os.makedirs("test-results", exist_ok=True)
    
    tree = ET.ElementTree(testsuites)
    tree.write("test-results/junit-results.xml")
    
    print("Created sample test result file: test-results/junit-results.xml")
    return "test-results"

def debug_all_error_extraction_methods(test_collector):
    """Test all error message extraction methods."""
    logger.info("Testing all error message extraction methods")
    
    logger.info("Method 1: Regular test result collection")
    test_results = test_collector.get_test_results()
    if test_results and test_results.get("error_message"):
        logger.info(f"✓ Successfully extracted error message via regular method: {test_results['error_message']}")
    else:
        logger.warning("✗ Could not extract error message via regular method")
    
    logger.info("Method 2: Direct extraction from test output logs")
    error_message = test_collector.extract_error_from_test_output()
    if error_message:
        logger.info(f"✓ Successfully extracted error message from logs: {error_message}")
    else:
        logger.warning("✗ Could not extract error message from logs")
    
    logger.info("Method 3: Direct XML file parsing")
    import glob
    xml_files = glob.glob(os.path.join(test_collector.results_dir, "*.xml"))
    if xml_files:
        logger.info(f"Found {len(xml_files)} XML files to parse")
        for xml_file in xml_files:
            result = test_collector.parse_junit_xml(xml_file)
            if result.get("error_message"):
                logger.info(f"✓ Successfully extracted error message from {xml_file}: {result['error_message']}")
            else:
                logger.warning(f"✗ Could not extract error message from {xml_file}")
    else:
        logger.warning("No XML files found")
    
    logger.info("Method 4: Custom XML element extraction")
    for xml_file in xml_files:
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            errors = root.findall(".//error")
            failures = root.findall(".//failure")
            
            if errors:
                for i, error in enumerate(errors):
                    logger.info(f"Found error element {i+1}: message={error.get('message')}")
                    if error.text:
                        logger.info(f"Error text: {error.text[:100]}")
            
            if failures:
                for i, failure in enumerate(failures):
                    logger.info(f"Found failure element {i+1}: message={failure.get('message')}")
                    if failure.text:
                        logger.info(f"Failure text: {failure.text[:100]}")
                        
            if not errors and not failures:
                logger.warning(f"No error or failure elements found in {xml_file}")
                
        except Exception as e:
            logger.error(f"Error parsing {xml_file}: {e}")
    
    return test_results.get("error_message") if test_results else error_message

def main():
    """Main entry point for CommitHunter."""
    args = parse_arguments()
    
    try:
        config = init_config(args.config)
        
        setup_logging(config.config['logging'])
        global logger
        logger = get_logger("main")
        
        logger.info("Starting CommitHunter")
        logger.info(f"Repository: {args.repo}")
        logger.info(f"Good commit: {args.good}")
        logger.info(f"Bad commit: {args.bad}")
        
        if not (args.test_results):
            logger.info("No test results directory specified, creating sample test results")
            test_results_dir = create_sample_test_results()
        else:
            test_results_dir = args.test_results
        
        git_collector = GitCollector(args.repo)
        test_collector = TestCollector(test_results_dir)
        
        analyzers = init_analyzers(config, git_collector)
        
        if any(isinstance(analyzer, BinarySearchAnalyzer) for analyzer in analyzers) and not args.test_name:
            logger.warning("BinarySearchAnalyzer requires a test name. Use --test-name to specify a test.")
            
            analyzers = [a for a in analyzers if not isinstance(a, BinarySearchAnalyzer)]
            
            logger.info("Binary search will be skipped.")
        
        if not analyzers:
            logger.error("No analyzers enabled. Check your configuration.")
            sys.exit(1)
        
        logger.info("Testing test result extraction")
        test_results = test_collector.get_test_results()
        if test_results:
            logger.info(f"Found test results: {json.dumps(test_results, indent=2)}")
            error_message = test_results.get('error_message', '')
            if error_message:
                logger.info(f"Extracted error message: {error_message}")
            else:
                logger.warning("No error message found in test results")
        else:
            logger.warning("No test results found")
            
            logger.info("Attempting to collect all test results")
            all_results = test_collector.collect_test_results()
            logger.info(f"Found {len(all_results)} test result sets")
            if all_results:
                failure_info = test_collector.extract_failure_messages(all_results)
                if failure_info:
                    logger.info(f"Failure info: {json.dumps(failure_info, indent=2)}")
        results = analyze_commits(
            git_collector=git_collector,
            test_collector=test_collector,
            analyzers=analyzers,
            good_commit=args.good,
            bad_commit=args.bad,
            test_name=args.test_name,
            config=config
        )
        
        report = generate_report(results, args.report_format)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Results written to {args.output}")
        logger.info("CommitHunter completed successfully")
        
    except Exception as e:
        logger.exception("Error in CommitHunter")
        sys.exit(1)

if __name__ == "__main__":
    main()