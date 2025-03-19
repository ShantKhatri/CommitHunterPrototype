"""
Test Results Collector Module

This module provides functionality to collect and parse test results from various test frameworks.
Supports JUnit XML, JSON, log files, and other common test result formats.
"""

import os
import json
import glob
import xml.etree.ElementTree as ET
import re
import logging
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import gzip
import threading
from pathlib import Path

class TestCollector:
    """
    Advanced collector for test results with optimized parsing and failure analysis.
    
    Features:
    - Multi-format parsing (XML, JSON, log files, text files)
    - Result caching for performance
    - Thread-safe parsing for large test suites
    - Advanced error message extraction
    - Stack trace analysis
    - Integration with common CI systems
    """
    
    def __init__(self, results_dir: str, cache_dir: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the test collector with configurable options.
        
        Args:
            results_dir: Directory containing test results
            cache_dir: Directory for caching parsed results (optional)
            config: Configuration options for the collector
        """
        self.results_dir = results_dir
        self.cache_dir = cache_dir or os.path.join(results_dir, '.cache')
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.result_cache = {}
        self.cache_lock = threading.Lock()
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.java_exception_patterns = [
            r'([a-zA-Z0-9_.]+Exception)(?::|\.)',
            r'([a-zA-Z0-9_.]+Error)(?::|\.)',
            r'AssertionFailedError:?\s*(.*)',
            r'NullPointerException:?\s*(.*)',
        ]
        
        self.cpp_error_patterns = [
            r'Segmentation fault',
            r'segfault at (\w+)',
            r'SIGSEGV',
            r'core dumped',
            r'memory access violation',
            r'null pointer dereference',
            r'bus error',
            r'stack overflow',
            r'heap corruption',
        ]
        
        self.python_error_patterns = [
            r'([a-zA-Z0-9_.]+Error):?\s*(.*)',
            r'AssertionError:?\s*(.*)',
            r'ImportError:?\s*(.*)', 
        ]
        
        self.file_line_patterns = [
            r'at\s+([^(]+)\(([^:]+):(\d+)\)',
            r'([a-zA-Z0-9_/.\\-]+\.(?:cpp|java|h|py|js)):(\d+)',
        ]
        
        self.all_patterns = {
            'java': self._compile_patterns(self.java_exception_patterns),
            'cpp': self._compile_patterns(self.cpp_error_patterns),
            'python': self._compile_patterns(self.python_error_patterns),
            'file_line': self._compile_patterns(self.file_line_patterns)
        }
        
        self.stopwords = self._load_stopwords()
        
        self.logger.info(f"Test collector initialized for {results_dir}")
    
    def _compile_patterns(self, patterns: List[str]) -> List[re.Pattern]:
        """Compile regex patterns once for efficiency."""
        return [re.compile(pattern, re.MULTILINE) for pattern in patterns]
    
    def _load_stopwords(self) -> Set[str]:
        """Load common stopwords to filter out noise."""
        stopwords = {
            'test', 'assert', 'error', 'exception', 'the', 'and', 'but', 'or', 
            'for', 'not', 'with', 'this', 'that', 'these', 'those', 'null',
            'true', 'false', 'java', 'org', 'com', 'net', 'public', 'private',
            'static', 'class', 'void', 'main', 'string', 'int', 'boolean', 
            'expected', 'actual', 'failed', 'failure', 'method'
        }
        
        if self.config.get('stopwords'):
            stopwords.update(self.config.get('stopwords'))
            
        return stopwords
    
    def _get_cache_key(self, path: str) -> str:
        """Generate a stable cache key from file path and modification time."""
        mtime = os.path.getmtime(path) if os.path.exists(path) else 0
        return hashlib.md5(f"{path}:{mtime}".encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Retrieve a cached result if available."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json.gz")
        
        if os.path.exists(cache_file):
            try:
                with gzip.open(cache_file, 'rt') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Cache read error: {str(e)}")
        
        return None
    
    def _save_cached_result(self, cache_key: str, result: Dict) -> None:
        """Save parsed result to cache."""
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json.gz")
        
        try:
            with gzip.open(cache_file, 'wt') as f:
                json.dump(result, f)
        except Exception as e:
            self.logger.warning(f"Cache write error: {str(e)}")
    
    @lru_cache(maxsize=128)
    def parse_junit_xml(self, file_path: str) -> Dict[str, Any]:
        """Parse a JUnit XML file and extract error messages."""
        try:
            self.logger.info(f"Parsing JUnit XML file: {file_path}")
            
            if not os.path.exists(file_path):
                self.logger.warning(f"File not found: {file_path}")
                return {}
            
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            self.logger.debug(f"XML root element: {root.tag}")
            
            results = {
                "name": os.path.basename(file_path),
                "full_path": file_path,
                "timestamp": datetime.now().isoformat(),
                "tests": 0,
                "failures": 0,
                "errors": 0,
                "skipped": 0,
                "time": 0.0,
                "test_cases": [],
                "error_messages": []
            }
            
            errors = root.findall(".//error")
            failures = root.findall(".//failure")

            for error in errors:
                message = error.get("message", "No message")
                text = error.text.strip() if error.text else "No details"
                self.logger.info(f"Extracted error: {message}")
                results["error_messages"].append(f"{message}: {text}")

            for failure in failures:
                message = failure.get("message", "No message")
                text = failure.text.strip() if failure.text else "No details"
                self.logger.info(f"Extracted failure: {message}")
                results["error_messages"].append(f"{message}: {text}")

            return results

        except Exception as e:
            self.logger.error(f"Error parsing JUnit XML file {file_path}: {str(e)}")
            return {"error": str(e)}
    
    def _parse_testsuite(self, testsuite: ET.Element, results: Dict) -> None:
        """Parse a test suite element and update results dictionary."""
        results["tests"] += int(testsuite.get("tests", 0))
        results["failures"] += int(testsuite.get("failures", 0))
        results["errors"] += int(testsuite.get("errors", 0))
        results["skipped"] += int(testsuite.get("skipped", 0))
        results["time"] += float(testsuite.get("time", 0))
        
        if testsuite.get("timestamp") and not results.get("timestamp"):
            results["timestamp"] = testsuite.get("timestamp")
        
        suite_name = testsuite.get("name", "")
        
        for testcase in testsuite.findall("./testcase"):
            case = self._parse_testcase(testcase, suite_name)
            results["test_cases"].append(case)
    
    def _parse_testcase(self, testcase: ET.Element, suite_name: str) -> Dict:
        """Parse a test case element and return a dictionary."""
        case = {
            "name": testcase.get("name", ""),
            "classname": testcase.get("classname", ""),
            "time": float(testcase.get("time", 0)),
            "suite": suite_name,
            "status": "passed"
        }
        
        failure = testcase.find("./failure")
        if failure is not None:
            case["status"] = "failed"
            case["failure"] = {
                "message": failure.get("message", ""),
                "type": failure.get("type", ""),
                "text": failure.text.strip() if failure.text else ""
            }
            
            self.logger.debug(f"Found failure in {case['name']}: {case['failure']['message']}")
        
        error = testcase.find("./error")
        if error is not None:
            case["status"] = "error"
            case["error"] = {
                "message": error.get("message", ""),
                "type": error.get("type", ""),
                "text": error.text.strip() if error.text else ""
            }
            
            self.logger.debug(f"Found error in {case['name']}: {case['error']['message']}")
        
        skipped = testcase.find("./skipped")
        if skipped is not None:
            case["status"] = "skipped"
        
        return case
    
    def parse_json_results(self, file_path: str) -> Dict[str, Any]:
        """
        Parse JSON test results with caching.
        
        Args:
            file_path: Path to the JSON results file
            
        Returns:
            Dictionary with parsed test results
        """
        self.logger.debug(f"Parsing JSON results file: {file_path}")
        
        cache_key = self._get_cache_key(file_path)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            normalized = self._normalize_json_results(results, file_path)
            
            self._save_cached_result(cache_key, normalized)
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error parsing JSON results file {file_path}: {str(e)}")
            return {
                "name": os.path.basename(file_path),
                "full_path": file_path,
                "error": str(e),
                "test_cases": [],
                "error_messages": []
            }
    
    def _normalize_json_results(self, results: Dict, file_path: str) -> Dict:
        """Normalize JSON results to a consistent structure."""
        if "test_cases" in results and isinstance(results["test_cases"], list):
            results["name"] = results.get("name", os.path.basename(file_path))
            results["full_path"] = file_path
            
            if "error_messages" not in results:
                results["error_messages"] = self._extract_error_messages(results["test_cases"])
                
            return results
        
        normalized = {
            "name": os.path.basename(file_path),
            "full_path": file_path,
            "timestamp": results.get("timestamp", datetime.now().isoformat()),
            "tests": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "test_cases": []
        }
        
        if "reports" in results:
            for report in results["reports"]:
                self._normalize_maven_report(report, normalized)
        
        elif "testResults" in results:
            for test_result in results["testResults"]:
                self._normalize_jest_result(test_result, normalized)
        
        elif "tests" in results and isinstance(results["tests"], list):
            for test in results["tests"]:
                self._normalize_pytest_result(test, normalized)
        
        else:
            self._normalize_generic_json(results, normalized)
        
        normalized["error_messages"] = self._extract_error_messages(normalized["test_cases"])
        
        return normalized
    
    def _normalize_maven_report(self, report: Dict, normalized: Dict) -> None:
        """Convert Maven Surefire report to normalized format."""
        normalized["tests"] += report.get("tests", 0)
        normalized["failures"] += report.get("failures", 0)
        normalized["errors"] += report.get("errors", 0)
        normalized["skipped"] += report.get("skipped", 0)
        
        for test_case in report.get("testCases", []):
            case = {
                "name": test_case.get("name", ""),
                "classname": test_case.get("classname", ""),
                "time": test_case.get("time", 0),
                "status": "passed"
            }
            
            if test_case.get("failureMessage"):
                case["status"] = "failed"
                case["failure"] = {
                    "message": test_case.get("failureMessage", ""),
                    "type": test_case.get("failureType", ""),
                    "text": test_case.get("failureDetail", "")
                }
                case["error_details"] = self._extract_error_details(
                    test_case.get("failureMessage", "") + " " + test_case.get("failureDetail", "")
                )
            
            normalized["test_cases"].append(case)
    
    def _normalize_jest_result(self, test_result: Dict, normalized: Dict) -> None:
        """Convert Jest test result to normalized format."""
        for assertion_result in test_result.get("assertionResults", []):
            status = assertion_result.get("status", "")
            
            case = {
                "name": assertion_result.get("title", ""),
                "classname": test_result.get("name", ""),
                "time": assertion_result.get("duration", 0) / 1000,  # Convert ms to seconds
                "status": "passed" if status == "passed" else "failed"
            }
            
            if status != "passed":
                failure_message = ""
                for message in assertion_result.get("failureMessages", []):
                    failure_message += message + "\n"
                
                case["failure"] = {
                    "message": failure_message,
                    "type": "AssertionError",
                    "text": failure_message
                }
                case["error_details"] = self._extract_error_details(failure_message)
                
                normalized["failures"] += 1
            
            normalized["tests"] += 1
            normalized["test_cases"].append(case)
    
    def _normalize_pytest_result(self, test: Dict, normalized: Dict) -> None:
        """Convert Pytest test result to normalized format."""
        case = {
            "name": test.get("name", ""),
            "classname": test.get("nodeid", "").split("::")[0],
            "time": test.get("duration", 0),
            "status": test.get("outcome", "passed")
        }
        
        if test.get("outcome") not in ("passed", "skipped"):
            case["status"] = "failed"
            case["failure"] = {
                "message": test.get("longrepr", ""),
                "type": "AssertionError",
                "text": test.get("longrepr", "")
            }
            case["error_details"] = self._extract_error_details(test.get("longrepr", ""))
            
            normalized["failures"] += 1
        elif test.get("outcome") == "skipped":
            normalized["skipped"] += 1
        
        normalized["tests"] += 1
        normalized["test_cases"].append(case)
    
    def _normalize_generic_json(self, results: Dict, normalized: Dict) -> None:
        """Convert generic JSON test results to normalized format."""
        test_cases = results.get("testCases", results.get("tests", []))
        if isinstance(test_cases, list):
            for test_case in test_cases:
                case = {
                    "name": test_case.get("name", ""),
                    "classname": test_case.get("className", test_case.get("classname", "")),
                    "time": float(test_case.get("time", test_case.get("duration", 0))),
                    "status": test_case.get("status", test_case.get("result", "passed"))
                }
                
                if case["status"] not in ("passed", "skipped"):
                    case["failure"] = {
                        "message": test_case.get("message", test_case.get("errorMessage", "")),
                        "type": test_case.get("type", test_case.get("errorType", "")),
                        "text": test_case.get("details", test_case.get("stackTrace", ""))
                    }
                    case["error_details"] = self._extract_error_details(
                        case["failure"]["message"] + " " + case["failure"]["text"]
                    )
                    
                    normalized["failures"] += 1
                elif case["status"] == "skipped":
                    normalized["skipped"] += 1
                
                normalized["tests"] += 1
                normalized["test_cases"].append(case)
    
    def parse_log_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse test log files for error messages with optimized pattern matching.
        
        Args:
            file_path: Path to the log file
            
        Returns:
            Dictionary with parsed error information
        """
        self.logger.debug(f"Parsing log file: {file_path}")
        
        cache_key = self._get_cache_key(file_path)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            results = {
                "name": os.path.basename(file_path),
                "full_path": file_path,
                "timestamp": datetime.now().isoformat(),
                "type": "log",
                "test_cases": [],
                "error_messages": []
            }
            
            failures = self._extract_failures_from_log(content)
            
            if failures:
                for i, failure in enumerate(failures):
                    case = {
                        "name": f"log_failure_{i+1}",
                        "classname": failure.get("component", "unknown"),
                        "status": "failed",
                        "failure": {
                            "message": failure.get("message", ""),
                            "type": failure.get("type", "Error"),
                            "text": failure.get("details", "")
                        },
                        "error_details": failure.get("error_details", {})
                    }
                    results["test_cases"].append(case)
                
                results["tests"] = len(failures)
                results["failures"] = len(failures)
                results["error_messages"] = [f["message"] for f in failures]
            else:
                results["tests"] = 0
                results["failures"] = 0
            
            self._save_cached_result(cache_key, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error parsing log file {file_path}: {str(e)}")
            return {
                "name": os.path.basename(file_path),
                "full_path": file_path,
                "error": str(e),
                "test_cases": [],
                "error_messages": []
            }
    
    def _extract_failures_from_log(self, content: str) -> List[Dict]:
        """Extract test failures from log content."""
        failures = []
        
        test_blocks = re.split(r'\n\n+', content)
        
        for block in test_blocks:
            if not block.strip():
                continue
                
            error_match = None
            
            for pattern in self.all_patterns['java']:
                match = pattern.search(block)
                if match:
                    error_match = match
                    error_type = 'java'
                    break
            
            if not error_match:
                for pattern in self.all_patterns['cpp']:
                    match = pattern.search(block)
                    if match:
                        error_match = match
                        error_type = 'cpp'
                        break
            
            if not error_match:
                for pattern in self.all_patterns['python']:
                    match = pattern.search(block)
                    if match:
                        error_match = match
                        error_type = 'python'
                        break
            
            if error_match:
                file_matches = []
                for pattern in self.all_patterns['file_line']:
                    file_matches.extend(pattern.findall(block))
                
                failure = {
                    "message": error_match.group(0),
                    "type": error_match.group(1) if len(error_match.groups()) > 0 else "Error",
                    "details": block,
                    "component": self._extract_component_from_block(block),
                    "error_details": self._extract_error_details(block),
                    "files": file_matches
                }
                
                failures.append(failure)
        
        return failures
    
    def _extract_component_from_block(self, block: str) -> str:
        """Extract component name from error block."""
        component_matches = re.findall(r'(?:package|module|component)[\s:]+([a-zA-Z0-9_.]+)', block, re.IGNORECASE)
        if component_matches:
            return component_matches[0]
            
        package_matches = re.findall(r'([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+){2,})', block)
        if package_matches:
            return '.'.join(package_matches[0].split('.')[:2])
            
        dir_matches = re.findall(r'[/\\]([a-zA-Z0-9_-]+)[/\\][a-zA-Z0-9_-]+\.[a-z]+', block)
        if dir_matches:
            return dir_matches[0]
            
        return "unknown"
    
    def _extract_error_details(self, error_text: str) -> Dict[str, Any]:
        """
        Extract detailed error information from error text.
        
        Args:
            error_text: The error message and stack trace text
            
        Returns:
            Dictionary with detailed error information
        """
        if not error_text:
            return {}
            
        details = {
            "exception_type": "",
            "message": "",
            "stack_frames": [],
            "file_paths": set(),
            "components": set(),
            "keywords": set(),
        }
        
        exception_match = re.search(r'([A-Z][a-zA-Z0-9_]*(?:Exception|Error|Failure))', error_text)
        if exception_match:
            details["exception_type"] = exception_match.group(1)
        
        if details["exception_type"]:
            message_match = re.search(f'{re.escape(details["exception_type"])}:?\s*([^\n]+)', error_text)
            if message_match:
                details["message"] = message_match.group(1).strip()
        
        stack_frames = []
        
        java_frames = re.findall(r'at\s+([^(]+)\(([^:]+):(\d+)\)', error_text)
        for method, file, line in java_frames:
            stack_frames.append({
                "language": "java",
                "method": method.strip(),
                "file": file.strip(),
                "line": int(line)
            })
            details["file_paths"].add(file.strip())
            
            package_parts = method.strip().split('.')
            if len(package_parts) > 1:
                details["components"].add(package_parts[0])
        
        cpp_frames = re.findall(r'([a-zA-Z0-9_:]+)\s+\[0x[0-9a-f]+\]', error_text)
        for method in cpp_frames:
            stack_frames.append({
                "language": "cpp",
                "method": method.strip(),
                "file": "",
                "line": 0
            })
        
        python_frames = re.findall(r'File "([^"]+)", line (\d+), in (.+)', error_text)
        for file, line, method in python_frames:
            stack_frames.append({
                "language": "python",
                "method": method.strip(),
                "file": file.strip(),
                "line": int(line)
            })
            details["file_paths"].add(file.strip())
        
        details["stack_frames"] = stack_frames
        
        if details["message"]:
            words = re.findall(r'\b([a-zA-Z][a-zA-Z0-9_]{3,})\b', details["message"])
            for word in words:
                if word.lower() not in self.stopwords:
                    details["keywords"].add(word)
        
        details["file_paths"] = list(details["file_paths"])
        details["components"] = list(details["components"])
        details["keywords"] = list(details["keywords"])
        
        return details
    
    def _extract_error_messages(self, test_cases: List[Dict]) -> List[str]:
        """Extract error messages from test cases."""
        error_messages = []
        
        for case in test_cases:
            if case["status"] in ["failed", "error"]:
                message = ""
                
                if "failure" in case:
                    message = case["failure"].get("message", "")
                    if not message:
                        message = case["failure"].get("text", "")
                
                elif "error" in case:
                    message = case["error"].get("message", "")
                    if not message:
                        message = case["error"].get("text", "")
                
                if message:
                    error_messages.append(message)
        
        return error_messages
    
    def collect_test_results(self, patterns: List[str] = None) -> List[Dict[str, Any]]:
        """Collect and parse all test result files."""
        if patterns is None:
            patterns = ["**/*.xml", "**/*.json", "**/*.txt"]
        
        start_time = time.time()
        results = []
        
        try:
            self.logger.info(f"Looking for test result files in {self.results_dir}")
            
            if not self.results_dir or not os.path.exists(self.results_dir):
                self.logger.warning(f"Test results directory not found: {self.results_dir}")
                return []
            
            all_files = []
            for pattern in patterns:
                matching_files = list(Path(self.results_dir).glob(pattern))
                self.logger.info(f"Found {len(matching_files)} files matching pattern {pattern}")
                all_files.extend([str(f) for f in matching_files])
            
            if not all_files:
                self.logger.warning(f"No test result files found in {self.results_dir}")
                return []
            
            for file_path in all_files:
                self.logger.info(f"Found test result file: {file_path}")
                
                if file_path.endswith(".xml"):
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        self.logger.debug(f"Content of {file_path}: {content[:200]}...")
                    except Exception as e:
                        self.logger.warning(f"Could not read {file_path}: {e}")
            
            futures = []
            
            with ThreadPoolExecutor(max_workers=min(10, os.cpu_count() or 4)) as executor:
                for file_path in all_files:
                    if file_path.endswith(".xml"):
                        futures.append(executor.submit(self.parse_junit_xml, file_path))
                    elif file_path.endswith(".json"):
                        futures.append(executor.submit(self._parse_json_results, file_path))
                    else:
                        futures.append(executor.submit(self._parse_text_results, file_path))
                
                for future in futures:
                    try:
                        result = future.result()
                        if result:
                            self.logger.info(f"Parsed test result: {result.get('name')} with {result.get('tests', 0)} tests")
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"Error collecting test results: {str(e)}")
            
            self.logger.info(f"Collected {len(results)} test result sets in {time.time() - start_time:.2f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in collect_test_results: {str(e)}")
            return []

    def extract_failure_messages(self, results: List[Dict]) -> Dict[str, List[str]]:
        """
        Extract organized failure messages from test results.
        
        Args:
            results: List of parsed test results
            
        Returns:
            Dictionary mapping error types to failure messages
        """
        failure_info = {
            "exceptions": [],
            "assertion_failures": [],
            "error_messages": [],
            "stack_traces": [],
            "file_references": set(),
            "components": set(),
            "most_common_errors": {}
        }
        
        error_counts = {}
        
        for result in results:
            for message in result.get("error_messages", []):
                if message:
                    failure_info["error_messages"].append(message)
            
            for case in result.get("test_cases", []):
                if case["status"] in ["failed", "error"]:
                    if "failure" in case:
                        failure = case["failure"]
                        message = failure.get("message", "")
                        failure_type = failure.get("type", "")
                        details = failure.get("text", "")
                        
                        if failure_type:
                            error_counts[failure_type] = error_counts.get(failure_type, 0) + 1
                        
                        if "assert" in failure_type.lower() or "assert" in message.lower():
                            failure_info["assertion_failures"].append(message)
                        else:
                            failure_info["exceptions"].append(f"{failure_type}: {message}")
                        
                        if details and len(details) > 10:
                            failure_info["stack_traces"].append(details)
                    
                    if "error_details" in case:
                        details = case["error_details"]
                        if details.get("exception_type"):
                            failure_info["exceptions"].append(details["exception_type"])
                        
                        for file_path in details.get("file_paths", []):
                            failure_info["file_references"].add(file_path)
                        
                        for component in details.get("components", []):
                            failure_info["components"].add(component)
        
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        for error_type, count in sorted_errors[:5]:  # Get top 5
            failure_info["most_common_errors"][error_type] = count
        
        failure_info["file_references"] = list(failure_info["file_references"])
        failure_info["components"] = list(failure_info["components"])
        
        return failure_info

    def get_test_results(self, test_name: Optional[str] = None) -> Optional[Dict]:
        """
        Get test results with enhanced error message extraction.
        
        Args:
            test_name: Name of the test to find (optional)
            
        Returns:
            Dictionary with test results and error information, or None if not found
        """
        self.logger.info(f"Getting test results for {'test: ' + test_name if test_name else 'all tests'}")
        
        all_results = self.collect_test_results()
        if not all_results:
            self.logger.warning("No test results found through standard collection methods")
            
            error_message = self.extract_error_from_test_output()
            if error_message:
                self.logger.info(f"Found error message from logs: {error_message[:100]}")
                return {
                    "test_name": test_name or "unknown",
                    "status": "failed",
                    "error_message": error_message,
                    "source": "log_extraction",
                    "timestamp": datetime.now().isoformat()
                }
            return None
        
        if test_name:
            for result in all_results:
                for test_case in result.get("test_cases", []):
                    if (test_case["name"] == test_name or 
                        test_name in test_case["name"] or
                        (test_case.get("classname") and test_name in test_case["classname"])):
                        
                        if test_case["status"] in ["failed", "error"]:
                            failure_info = {}
                            
                            if "failure" in test_case:
                                failure = test_case["failure"]
                                failure_info = {
                                    "message": failure.get("message", ""),
                                    "type": failure.get("type", ""),
                                    "details": failure.get("text", ""),
                                    "error_details": test_case.get("error_details", {})
                                }
                            elif "error" in test_case:
                                error = test_case["error"]
                                failure_info = {
                                    "message": error.get("message", ""),
                                    "type": error.get("type", ""),
                                    "details": error.get("text", ""),
                                    "error_details": test_case.get("error_details", {})
                                }
                            
                            return {
                                "test_name": test_case["name"],
                                "test_class": test_case.get("classname", ""),
                                "status": test_case["status"],
                                "error_message": failure_info.get("message", ""),
                                "error_type": failure_info.get("type", ""),
                                "stack_trace": failure_info.get("details", ""),
                                "error_details": failure_info.get("error_details", {}),
                                "file_path": result["full_path"],
                                "timestamp": result.get("timestamp", "")
                            }
        
        failures = self.extract_failure_messages(all_results)
        
        if failures["error_messages"]:
            error_message = failures["error_messages"][0]
            error_type = ""
            
            if failures["exceptions"]:
                error_types = [e.split(':', 1)[0] for e in failures["exceptions"]]
                error_type = error_types[0]
            
            stack_trace = failures["stack_traces"][0] if failures["stack_traces"] else ""
            
            return {
                "test_name": test_name or "unknown",
                "status": "failed",
                "error_message": error_message,
                "error_type": error_type,
                "stack_trace": stack_trace,
                "failure_summary": failures,
                "file_references": failures["file_references"][:5],  # Limit to 5
                "components": failures["components"][:3],  # Limit to 3
                "timestamp": datetime.now().isoformat()
            }
        
        self.logger.warning("No test failures found in standard results, trying logs...")
        error_message = self.extract_error_from_test_output()
        if error_message:
            self.logger.info(f"Found error message from logs: {error_message[:100]}")
            return {
                "test_name": test_name or "unknown",
                "status": "failed",
                "error_message": error_message,
                "error_type": "Unknown",  # Can't determine from logs
                "source": "log_extraction",
                "timestamp": datetime.now().isoformat()
            }
        
        return None

    def find_relevant_test_failures(self, commit_hash: str, keywords: List[str] = None) -> List[Dict]:
        """
        Find test failures relevant to a specific commit based on keywords.
        
        Args:
            commit_hash: The commit hash to find relevant failures for
            keywords: Optional keywords to match against failures
            
        Returns:
            List of relevant test failures
        """
        self.logger.info(f"Finding test failures relevant to commit {commit_hash[:8]}")
        
        all_results = self.collect_test_results()
        if not all_results:
            return []
        
        failures = []
        for result in all_results:
            for test_case in result.get("test_cases", []):
                if test_case["status"] in ["failed", "error"]:
                    failure_info = {}
                    
                    if "failure" in test_case:
                        failure = test_case["failure"]
                        failure_info = {
                            "message": failure.get("message", ""),
                            "type": failure.get("type", ""),
                            "details": failure.get("text", ""),
                            "error_details": test_case.get("error_details", {})
                        }
                    elif "error" in test_case:
                        error = test_case["error"]
                        failure_info = {
                            "message": error.get("message", ""),
                            "type": error.get("type", ""),
                            "details": error.get("text", ""),
                            "error_details": test_case.get("error_details", {})
                        }
                    
                    failures.append({
                        "test_name": test_case["name"],
                        "test_class": test_case.get("classname", ""),
                        "error_message": failure_info.get("message", ""),
                        "error_type": failure_info.get("type", ""),
                        "stack_trace": failure_info.get("details", ""),
                        "error_details": failure_info.get("error_details", {}),
                        "file_path": result["full_path"],
                        "timestamp": result.get("timestamp", ""),
                        "relevance_score": 0  # Will be calculated below
                    })
        
        if not keywords:
            keywords = []
            keywords = ["error", "failure", "crash"]
        
        for failure in failures:
            score = 0
            error_text = (
                failure["error_message"] + " " + 
                failure["error_type"] + " " + 
                failure["stack_trace"]
            ).lower()
            
            for keyword in keywords:
                if keyword.lower() in error_text:
                    score += 1
                    
                    if keyword.lower() in failure["error_message"].lower():
                        score += 2
            
            if failure.get("error_details", {}).get("file_paths"):
                score += 1
                
            failure["relevance_score"] = score
        
        relevant_failures = sorted(failures, key=lambda x: x["relevance_score"], reverse=True)
        
        return relevant_failures[:10]

    def get_failure_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about test failures.
        
        Returns:
            Dictionary with failure statistics
        """
        all_results = self.collect_test_results()
        if not all_results:
            return {"total_tests": 0, "failures": 0, "error_types": {}}
        
        stats = {
            "total_tests": 0,
            "passed": 0,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "error_types": {},
            "failing_components": {},
            "failure_trends": {}
        }
        
        for result in all_results:
            stats["total_tests"] += result.get("tests", 0)
            stats["failures"] += result.get("failures", 0)
            stats["errors"] += result.get("errors", 0)
            stats["skipped"] += result.get("skipped", 0)
            
            for test_case in result.get("test_cases", []):
                if test_case["status"] == "passed":
                    stats["passed"] += 1
                
                elif test_case["status"] in ["failed", "error"]:
                    error_type = "Unknown"
                    if "failure" in test_case:
                        error_type = test_case["failure"].get("type", "Unknown")
                    elif "error" in test_case:
                        error_type = test_case["error"].get("type", "Unknown")
                    
                    stats["error_types"][error_type] = stats["error_types"].get(error_type, 0) + 1
                    
                    classname = test_case.get("classname", "")
                    if classname:
                        component = classname.split('.')[0] if '.' in classname else classname
                        stats["failing_components"][component] = stats["failing_components"].get(component, 0) + 1
        
        if stats["total_tests"] > 0:
            stats["pass_rate"] = stats["passed"] / stats["total_tests"]
        else:
            stats["pass_rate"] = 0
        
        return stats

    def clear_cache(self) -> None:
        """Clear the results cache."""
        try:
            cache_files = glob.glob(os.path.join(self.cache_dir, "*.json.gz"))
            for file in cache_files:
                try:
                    os.remove(file)
                except:
                    pass
            
            self.result_cache = {}
            self.logger.info(f"Cache cleared: {len(cache_files)} files removed")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")

    def generate_test_failure_report(self, output_format: str = "json") -> Any:
        """
        Generate a comprehensive test failure report.
        
        Args:
            output_format: Format of the report ('json', 'html', or 'markdown')
            
        Returns:
            Report in the requested format
        """
        all_results = self.collect_test_results()
        if not all_results:
            return {"error": "No test results found"}
        
        failure_info = self.extract_failure_messages(all_results)
        stats = self.get_failure_statistics()
        
        report = {
            "summary": {
                "total_tests": stats["total_tests"],
                "pass_rate": stats["pass_rate"],
                "failures": stats["failures"],
                "errors": stats["errors"],
                "skipped": stats["skipped"]
            },
            "error_types": stats["error_types"],
            "failing_components": stats["failing_components"],
            "failure_details": failure_info,
            "timestamp": datetime.now().isoformat()
        }
        
        if output_format == "json":
            return report
        
        elif output_format == "html":
            html = ["<html><head><style>",
                    "body { font-family: Arial, sans-serif; margin: 20px; }",
                    "table { border-collapse: collapse; width: 100%; }",
                    "th, td { border: 1px solid #ddd; padding: 8px; }",
                    "th { background-color: #f2f2f2; }",
                    "tr:nth-child(even) { background-color: #f9f9f9; }",
                    ".failure { color: red; }",
                    ".pass { color: green; }",
                    "</style></head><body>",
                    f"<h1>Test Failure Report</h1>",
                    f"<h2>Summary</h2>",
                    f"<p>Total Tests: {stats['total_tests']}</p>",
                    f"<p>Pass Rate: {stats['pass_rate']:.2%}</p>",
                    f"<p>Failures: <span class='failure'>{stats['failures']}</span></p>",
                    f"<h2>Error Types</h2>",
                    "<table><tr><th>Error Type</th><th>Count</th></tr>"]
            
            for error_type, count in sorted(stats["error_types"].items(), key=lambda x: x[1], reverse=True):
                html.append(f"<tr><td>{error_type}</td><td>{count}</td></tr>")
            
            html.append("</table>")
            html.append("<h2>Failure Messages</h2><ul>")
            
            for message in failure_info["error_messages"][:10]:  # Show top 10
                html.append(f"<li>{message}</li>")
            
            html.append("</ul></body></html>")
            return "\n".join(html)
        
        elif output_format == "markdown":
            md = [
                "# Test Failure Report",
                "",
                "## Summary",
                f"- Total Tests: {stats['total_tests']}",
                f"- Pass Rate: {stats['pass_rate']:.2%}",
                f"- Failures: **{stats['failures']}**",
                "",
                "## Error Types",
                "",
                "| Error Type | Count |",
                "| --- | --- |"
            ]
            
            for error_type, count in sorted(stats["error_types"].items(), key=lambda x: x[1], reverse=True):
                md.append(f"| {error_type} | {count} |")
            
            md.append("")
            md.append("## Failure Messages")
            md.append("")
            
            for i, message in enumerate(failure_info["error_messages"][:10], 1):
                md.append(f"{i}. {message}")
            
            return "\n".join(md)
        
        else:
            return {"error": f"Unsupported output format: {output_format}"}  
    
    def _extract_direct_error_messages(self, results: Dict[str, Any]) -> None:
        """Extract error messages directly from test cases with detailed logging."""
        error_messages = []
        
        self.logger.debug(f"Extracting error messages from {len(results.get('test_cases', []))} test cases")
        
        for test_case in results.get("test_cases", []):
            if test_case["status"] in ["failed", "error"]:
                if "failure" in test_case:
                    failure = test_case["failure"]
                    message = failure.get("message", "")
                    if message:
                        self.logger.debug(f"Found failure message: {message[:100]}")
                        error_messages.append(message)
                        
                    elif failure.get("text"):
                        text = failure["text"].strip()
                        if text:
                            first_line = next((line.strip() for line in text.split('\n') if line.strip()), "")
                            if first_line:
                                self.logger.debug(f"Using first line of failure text: {first_line[:100]}")
                                error_messages.append(first_line)
                
                if "error" in test_case and not error_messages:
                    error = test_case["error"]
                    message = error.get("message", "")
                    if message:
                        self.logger.debug(f"Found error message: {message[:100]}")
                        error_messages.append(message)
                        
                    elif error.get("text"):
                        text = error["text"].strip()
                        if text:
                            first_line = next((line.strip() for line in text.split('\n') if line.strip()), "")
                            if first_line:
                                self.logger.debug(f"Using first line of error text: {first_line[:100]}")
                                error_messages.append(first_line)
        
        if error_messages:
            self.logger.info(f"Found {len(error_messages)} error messages")
            results["error_messages"] = error_messages
            results["error_message"] = error_messages[0]
            self.logger.info(f"Primary error message: {results['error_message']}")
        else:
            self.logger.warning("No error messages found in test cases")
            results["error_messages"] = []

    def extract_error_from_test_output(self) -> Optional[str]:
        """
        Extract error messages from test logs when XML parsing fails.
        """
        self.logger.info("Extracting error messages from test logs.")

        log_files = [f for f in os.listdir(self.results_dir) if f.endswith(".log") or f.endswith(".txt")]

        for log_file in log_files:
            log_path = os.path.join(self.results_dir, log_file)
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in reversed(lines):
                        if "ERROR" in line or "FAILURE" in line:
                            self.logger.info(f"Extracted error from logs: {line.strip()}")
                            return line.strip()
            except Exception as e:
                self.logger.warning(f"Failed to read log file {log_file}: {e}")

        return None

    def run_test(self, test_name: str, mode: str = 'full') -> Dict[str, Any]:
        """
        Run a specific test and return results.
        
        Args:
            test_name: Name of the test to run
            mode: Test execution mode ('full', 'lightweight')
            
        Returns:
            Dictionary with test results or None if test couldn't be run
        """
        self.logger.info(f"Running test '{test_name}' in {mode} mode")
        
        test_command = self._build_test_command(test_name, mode)
        if not test_command:
            self.logger.warning(f"Could not determine how to run test '{test_name}'")
            return None
        
        try:
            self.logger.info(f"Executing command: {test_command}")
            
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as stdout_file, \
                 tempfile.NamedTemporaryFile(mode='w+', delete=False) as stderr_file:
                
                start_time = time.time()
                process = subprocess.Popen(
                    test_command,
                    shell=True,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    universal_newlines=True
                )
                
                timeout = 300 if mode == 'full' else 60  # Shorter timeout for lightweight mode
                try:
                    exit_code = process.wait(timeout=timeout)
                    duration = time.time() - start_time
                    
                    stdout_file.flush()
                    stderr_file.flush()
                    
                    with open(stdout_file.name, 'r') as f:
                        stdout_content = f.read()
                    
                    with open(stderr_file.name, 'r') as f:
                        stderr_content = f.read()
                    
                    os.unlink(stdout_file.name)
                    os.unlink(stderr_file.name)
                    
                    test_passed = exit_code == 0
                    
                    result = {
                        "test_name": test_name,
                        "status": "passed" if test_passed else "failed",
                        "execution_time": duration,
                        "exit_code": exit_code,
                        "mode": mode,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    if not test_passed:
                        error_msg = self._extract_error_message(stdout_content, stderr_content)
                        result["error_message"] = error_msg
                        
                        if 'performance' in test_name.lower() or 'bench' in test_name.lower():
                            result["performance_metrics"] = self._extract_performance_metrics(stdout_content, stderr_content)
                    
                    elif mode == 'full' and ('performance' in test_name.lower() or 'bench' in test_name.lower()):
                        result["performance_metrics"] = self._extract_performance_metrics(stdout_content, stderr_content)
                    
                    self._save_test_results(result, test_name, mode)
                    
                    return result
                    
                except subprocess.TimeoutExpired:
                    process.kill()
                    self.logger.warning(f"Test '{test_name}' timed out after {timeout} seconds")
                    
                    os.unlink(stdout_file.name)
                    os.unlink(stderr_file.name)
                    
                    return {
                        "test_name": test_name,
                        "status": "failed",
                        "error_message": f"Test timed out after {timeout} seconds",
                        "execution_time": timeout,
                        "exit_code": -1,
                        "mode": mode,
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            self.logger.error(f"Error running test '{test_name}': {str(e)}")
            return {
                "test_name": test_name,
                "status": "error",
                "error_message": str(e),
                "exit_code": -1,
                "mode": mode,
                "timestamp": datetime.now().isoformat()
            }

    def _build_test_command(self, test_name: str, mode: str) -> str:
        """Build command to run the specified test."""
        
        if os.path.exists('pom.xml'):
            if mode == 'lightweight':
                return f"mvn test -Dtest={test_name} -DfailIfNoTests=false"
            else:
                return f"mvn test -Dtest={test_name} -DfailIfNoTests=false"
        
        elif os.path.exists('build.gradle') or os.path.exists('build.gradle.kts'):
            if mode == 'lightweight':
                return f"./gradlew test --tests {test_name} --info"
            else:
                return f"./gradlew test --tests {test_name} --info"
        
        elif os.path.exists('setup.py') or os.path.exists('pyproject.toml'):
            if mode == 'lightweight':
                return f"python -m pytest {test_name} -v --no-header"
            else:
                return f"python -m pytest {test_name} -v"
        
        elif os.path.exists('package.json'):
            if mode == 'lightweight':
                return f"npm test -- -t '{test_name}'"
            else:
                return f"npm test -- -t '{test_name}'"
        
        if 'performance' in test_name.lower() or 'benchmark' in test_name.lower():
            if mode == 'lightweight':
                return f"python -m {test_name} --quick"
            else:
                return f"python -m {test_name} --full"
        
        if self.config.get('test_command_pattern'):
            custom_command = self.config['test_command_pattern'].replace('{test_name}', test_name)
            custom_command = custom_command.replace('{mode}', mode)
            return custom_command
            
        self.logger.warning(f"Could not determine test command for '{test_name}'")
        return ""

    def _extract_error_message(self, stdout: str, stderr: str) -> str:
        """Extract error message from command output."""
        if stderr.strip():
            exception_lines = re.findall(r'^.*Exception.*', stderr, re.MULTILINE)
            if exception_lines:
                return exception_lines[0].strip()
                
            error_lines = re.findall(r'^.*Error.*', stderr, re.MULTILINE)
            if error_lines:
                return error_lines[0].strip()
                
            for line in stderr.split('\n'):
                if line.strip():
                    return line.strip()
        
        if stdout.strip():
            exception_lines = re.findall(r'^.*Exception.*', stdout, re.MULTILINE)
            if exception_lines:
                return exception_lines[0].strip()
                
            failure_lines = re.findall(r'^.*FAILED.*', stdout, re.MULTILINE)
            if failure_lines:
                return failure_lines[0].strip()
        
        return "Test failed, but no specific error message found"

    def _extract_performance_metrics(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Extract performance metrics from test output."""
        metrics = {
            "execution_time_ms": None,
            "throughput": None,
            "latency_ms": None,
            "memory_mb": None
        }
        
        
        time_match = re.search(r'(?:Time|Duration|Elapsed)[\s:]+(\d+(?:\.\d+)?)\s*(?:ms|s)', 
                             stdout + stderr, re.IGNORECASE)
        if time_match:
            time_value = float(time_match.group(1))
            if 's' in time_match.group(0) and 'ms' not in time_match.group(0):
                time_value *= 1000
            metrics["execution_time_ms"] = time_value
        
        throughput_match = re.search(r'(?:Throughput|Ops/sec)[\s:]+(\d+(?:\.\d+)?)', 
                                   stdout + stderr, re.IGNORECASE)
        if throughput_match:
            metrics["throughput"] = float(throughput_match.group(1))
        
        latency_match = re.search(r'(?:Latency|Response time)[\s:]+(\d+(?:\.\d+)?)\s*(?:ms|s)', 
                                stdout + stderr, re.IGNORECASE)
        if latency_match:
            latency_value = float(latency_match.group(1))
            if 's' in latency_match.group(0) and 'ms' not in latency_match.group(0):
                latency_value *= 1000
            metrics["latency_ms"] = latency_value
        
        memory_match = re.search(r'(?:Memory|Heap|RAM)[\s:]+(\d+(?:\.\d+)?)\s*(?:MB|KB)', 
                               stdout + stderr, re.IGNORECASE)
        if memory_match:
            memory_value = float(memory_match.group(1))
            if 'KB' in memory_match.group(0):
                memory_value /= 1024
            metrics["memory_mb"] = memory_value
        
        return metrics

    def _save_test_results(self, result: Dict[str, Any], test_name: str, mode: str) -> None:
        """Save test results to output directory."""
        try:
            sanitized_name = re.sub(r'[^\w.-]', '_', test_name)
            filename = f"{sanitized_name}_{mode}_{int(time.time())}.json"
            file_path = os.path.join(self.results_dir, filename)
            
            with open(file_path, 'w') as f:
                json.dump(result, f, indent=2)
                
            self.logger.info(f"Test results saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving test results: {str(e)}")

    def get_test_result_for_commit(self, commit_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get test result for a specific commit.
        
        Args:
            commit_hash: Commit hash to get result for
            
        Returns:
            Test result dictionary or None if not found
        """
        try:
            cache_key = f"result:{commit_hash}"
            with self.cache_lock:
                if cache_key in self.result_cache:
                    return self.result_cache[cache_key]
            
            all_results = self.collect_test_results()
            for result in all_results:
                if result.get('commit') == commit_hash:
                    with self.cache_lock:
                        self.result_cache[cache_key] = result
                    return result
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error getting test result for commit {commit_hash}: {str(e)}")
            return None

    def analyze_openj9_test_failure(self, test_name: str, error_message: str, good_sha: str, bad_sha: str) -> Dict[str, Any]:
        """
        Analyze OpenJ9 test failures to extract relevant information for commit classification.
        
        Args:
            test_name: Name of the failing test
            error_message: Error message from the test failure
            good_sha: SHA of the commit where tests pass
            bad_sha: SHA of the commit where tests fail
            
        Returns:
            Dictionary with analysis results and keywords for matching commits
        """
        self.logger.info(f"Analyzing OpenJ9 test failure: {test_name}")
        
        analysis = {
            "test_name": test_name,
            "error_message": error_message,
            "good_sha": good_sha,
            "bad_sha": bad_sha,
            "keywords": set(),
            "components": set(),
            "affected_files": set(),
            "jvm_specific": False,
            "classification_features": {}
        }
        
        if re.search(r'(JVMJ9|OpenJ9|JIT|GC|JVM)', error_message):
            analysis["jvm_specific"] = True
            
        gc_match = re.search(r'GC(\d+)', error_message)
        if gc_match:
            analysis["components"].add("gc")
            analysis["keywords"].add(f"GC{gc_match.group(1)}")
            
        jit_match = re.search(r'JIT\s*(compiler|optimization|codegen)', error_message, re.IGNORECASE)
        if jit_match:
            analysis["components"].add("jit")
            analysis["keywords"].add("jit")
            analysis["keywords"].add(jit_match.group(1).lower())
            
        thread_match = re.search(r'(thread|deadlock|race\s*condition|synchronization)', error_message, re.IGNORECASE)
        if thread_match:
            analysis["components"].add("thread")
            analysis["keywords"].add(thread_match.group(1).lower())
            
        class_matches = re.findall(r'([A-Z][a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)', error_message)
        for class_name, method_name in class_matches:
            analysis["keywords"].add(class_name)
            if len(method_name) > 3 and not method_name.lower() in self.stopwords:
                analysis["keywords"].add(method_name)
        
        file_matches = re.findall(r'(?:at|in|from)\s+([a-zA-Z0-9_/.\\-]+\.(?:c|cpp|java|h|hpp))', error_message)
        for file_path in file_matches:
            file_name = os.path.basename(file_path)
            analysis["affected_files"].add(file_name)
            if '/' in file_path:
                component = file_path.split('/')[0]
                analysis["components"].add(component)
        
        analysis["classification_features"] = {
            "has_assertion_failure": 1 if re.search(r'assert(ion)?\s+failed', error_message, re.IGNORECASE) else 0,
            "has_segmentation_fault": 1 if re.search(r'segmentation\s+fault|sigsegv', error_message, re.IGNORECASE) else 0,
            "has_null_pointer": 1 if re.search(r'null\s+pointer', error_message, re.IGNORECASE) else 0,
            "has_class_loading_issue": 1 if re.search(r'class\s+loading|classloader|noclassdef', error_message, re.IGNORECASE) else 0,
            "has_memory_issue": 1 if re.search(r'out\s+of\s+memory|memor(y|ies)\s+leak', error_message, re.IGNORECASE) else 0,
            "is_jit_related": 1 if analysis["components"] and "jit" in analysis["components"] else 0,
            "is_gc_related": 1 if analysis["components"] and "gc" in analysis["components"] else 0,
            "has_precise_file_reference": 1 if analysis["affected_files"] else 0
        }
        
        analysis["keywords"] = list(analysis["keywords"])
        analysis["components"] = list(analysis["components"])
        analysis["affected_files"] = list(analysis["affected_files"])
        
        return analysis

    def classify_commit_for_test_failure(self, commit_info: Dict[str, Any], 
                                        test_failure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a commit as "Likely Problematic" or "Safe" based on test failure analysis.
        
        Args:
            commit_info: Dictionary with commit information
            test_failure: Dictionary with test failure analysis from analyze_openj9_test_failure
            
        Returns:
            Dictionary with classification results
        """
        self.logger.info(f"Classifying commit {commit_info.get('hash', '')[:8]} for test failure")
        
        score = 0.0
        reasons = []
        
        # Get commit data
        commit_message = commit_info.get('message', '').lower()
        commit_files = [os.path.basename(f) for f in commit_info.get('files_changed', [])]
        author = commit_info.get('author_name', commit_info.get('author', ''))
        
        keyword_matches = 0
        for keyword in test_failure.get('keywords', []):
            if keyword.lower() in commit_message:
                keyword_matches += 1
                reasons.append(f"Keyword '{keyword}' found in commit message")
        
        if keyword_matches > 0:
            score += min(0.3, keyword_matches * 0.1)  # Up to 0.3 for keywords
        
        file_matches = 0
        for file_name in test_failure.get('affected_files', []):
            if file_name in commit_files:
                file_matches += 1
                reasons.append(f"Modified file '{file_name}' appears in test failure")
        
        if file_matches > 0:
            score += min(0.4, file_matches * 0.2)  # Up to 0.4 for file matches
        
        component_matches = 0
        commit_components = set()
        for file_path in commit_info.get('files_changed', []):
            if '/' in file_path:
                commit_components.add(file_path.split('/')[0])
        
        for component in test_failure.get('components', []):
            if component in commit_components:
                component_matches += 1
                reasons.append(f"Modified component '{component}' matches test failure")
        
        if component_matches > 0:
            score += min(0.3, component_matches * 0.15)  # Up to 0.3 for component matches
        
        if re.search(r'fix(?:ed|ing)?\s+(?:build|compile|warning)', commit_message):
            score -= 0.1
            reasons.append("Commit appears to be fixing build or compile issues")
        
        classification = {
            "commit_hash": commit_info.get('hash', ''),
            "score": score,
            "reasons": reasons,
            "classification": "Likely Problematic" if score >= 0.4 else "Safe",
            "confidence": score if score >= 0.4 else 1.0 - score
        }
        
        return classification

    def process_openj9_issue(self, issue_number: str, good_sha: str, bad_sha: str, 
                            git_collector) -> Dict[str, Any]:
        """
        Process an OpenJ9 issue to classify commits between good and bad SHAs.
        
        Args:
            issue_number: OpenJ9 issue number
            good_sha: SHA of the commit where tests pass
            bad_sha: SHA of the commit where tests fail
            git_collector: GitCollector instance to get commit information
            
        Returns:
            Dictionary with analysis results and classified commits
        """
        self.logger.info(f"Processing OpenJ9 issue #{issue_number} between {good_sha[:8]} and {bad_sha[:8]}")
        
        test_result = self.get_test_result_for_commit(bad_sha)
        if not test_result:
            test_result = {
                "error_message": f"Unknown failure in issue #{issue_number}",
                "test_name": "unknown"
            }
        
        failure_analysis = self.analyze_openj9_test_failure(
            test_result.get("test_name", "unknown"),
            test_result.get("error_message", "Unknown error"),
            good_sha,
            bad_sha
        )
        
        commits = git_collector.get_commits_between(good_sha, bad_sha)
        
        classified_commits = []
        for commit in commits:
            commit_info = git_collector.get_commit_info(commit['hash'])
            classification = self.classify_commit_for_test_failure(commit_info, failure_analysis)
            classified_commits.append(classification)
        
        classified_commits.sort(key=lambda x: x['score'], reverse=True)
        
        likely_problematic = [c for c in classified_commits if c['classification'] == "Likely Problematic"]
        safe_commits = [c for c in classified_commits if c['classification'] == "Safe"]
        
        return {
            "issue": issue_number,
            "good_sha": good_sha,
            "bad_sha": bad_sha,
            "failure_analysis": failure_analysis,
            "classified_commits": classified_commits,
            "likely_problematic_count": len(likely_problematic),
            "safe_count": len(safe_commits),
            "top_suspect": likely_problematic[0] if likely_problematic else None
        }

    def extract_openj9_specific_errors(self, error_message: str) -> Dict[str, Any]:
        """
        Extract OpenJ9-specific error patterns from error messages.
        
        Args:
            error_message: Error message from test failure
            
        Returns:
            Dictionary with extracted OpenJ9-specific error information
        """
        openj9_patterns = {
            "jit_failures": [
                r'JIT\s+compilation\s+failed',
                r'AOT\s+compilation\s+failed',
                r'Failed\s+to\s+JIT\s+compile\s+method',
                r'JIT\s+optimization\s+level'
            ],
            
            "gc_failures": [
                r'GC\s+cycle\s+started',
                r'concurrent\s+mark',
                r'out\s+of\s+memory',
                r'heap\s+exhausted',
                r'gc\s+allocation\s+failure'
            ],
            
            "class_loading_failures": [
                r'ClassLoader',
                r'NoClassDefFoundError',
                r'ClassNotFoundException',
                r'failed\s+to\s+load\s+class'
            ],
            
            "threading_failures": [
                r'deadlock\s+detected',
                r'thread\s+dump',
                r'timed\s+out\s+waiting',
                r'InterruptedException'
            ],
            
            "native_failures": [
                r'JNI\s+error',
                r'native\s+method',
                r'native\s+crash',
                r'SIGSEGV',
                r'segmentation\s+fault'
            ]
        }
        
        results = {
            "detected_patterns": {},
            "main_category": None,
            "error_type": "unknown"
        }
        
        for category, patterns in openj9_patterns.items():
            matches = []
            for pattern in patterns:
                if re.search(pattern, error_message, re.IGNORECASE):
                    matches.append(pattern)
            
            if matches:
                results["detected_patterns"][category] = matches
        
        if results["detected_patterns"]:
            main_category = max(results["detected_patterns"].items(), key=lambda x: len(x[1]))
            results["main_category"] = main_category[0]
            
            category_to_error = {
                "jit_failures": "JIT Compilation Error",
                "gc_failures": "Garbage Collection Error",
                "class_loading_failures": "Class Loading Error",
                "threading_failures": "Threading Issue",
                "native_failures": "Native Code Failure"
            }
            results["error_type"] = category_to_error.get(main_category[0], "Unknown Error")
        
        return results