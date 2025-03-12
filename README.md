# CommitHunter: AI-Powered Commit Debugger

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

CommitHunter is an intelligent tool designed to identify problematic Git commits causing test failures and performance regressions. Using a combination of string matching, binary search, and performance analysis, it helps developers quickly pinpoint the exact commits that introduced issues.

## 🚀 Features

- **Smart Commit Analysis**: Uses AI-powered pattern matching to identify suspicious commits
- **Binary Search**: Efficiently narrows down problematic commits
- **Performance Analysis**: Detects performance regressions and bottlenecks
- **Multiple Report Formats**: Supports JSON, HTML, and text output
- **Flexible Integration**: Works with various test frameworks and Git repositories

## 📋 Requirements

- Python 3.8 or higher
- Git
- Dependencies listed in `requirements.txt`

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/ShantKhatri/CommitHunter
cd CommitHunter
```

2. Set up the environment:
```bash
setup.bat
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Basic Command (Sample commits and versions provided)

#### For light run and between good and bad commits
```bash
python src\main.py --repo https://github.com/eclipse/openj9 --good "96ef5c5b4026552ec5d6f0413d034cf09ba7103f" --bad "eac681f0cee21af67657f575a366590b937a2a13" --output ./reports/openj9_results.html --report-format html --verbos
```

#### For Normal/High run and between two versions (for all commits in the last version)
```bash
python src/main.py --repo https://github.com/eclipse/openj9 --good openj9-0.38.0 --bad openj9-0.39.0 --output ./reports/analysis.html --report-format html --verbose
```

### Command Line Options

| Option | Description | Required |
|--------|-------------|----------|
| `--repo` | Repository URL or path | Yes |
| `--good` | Known good commit/tag | Yes |
| `--bad` | Known bad commit/tag | Yes |
| `--output` | Output file path | No |
| `--report-format` | Output format (json/html/text) | No |
| `--test-results` | Test results directory | No |
| `--verbose` | Enable verbose logging | No |
| `--test-name` | Specific test to analyze | No |

## 📁 Project Structure

```
CommitHunter/
├── src/
│   ├── collectors/              # Data collection modules
│   │   ├── git_collector.py     # Collects data from Git repositories
│   │   └── test_collector.py    # Collects and parses test results
│   │
│   ├── analyzers/               # Analysis modules
│   │   ├── string_matcher.py    # String matching analyzer
│   │   ├── binary_search.py     # Binary search analyzer
│   │   └── perf_analyzer.py     # Performance test analyzer
│   │
│   ├── utils/                   # Utility functions
│   │   ├── config.py            # Configuration handling
│   │   └── logging.py           # Logging setup
│   │
│   └── main.py                  # Main entry point
│
├── tests/                       # Unit and integration tests
├── docs/                        # Documentation
├── reports/                     # Analysis reports
├── test_results/                # Test result files
├── logs/                        # Log files
├── config/                      # Configuration files
│   └── config.yaml              # Default configuration
│
└── requirements.txt             # Dependencies
```

## 🛠 Processing Flow

Below is a visual representation of the sequential processing flow:

![Processing Flow](docs/Sequential_Processing_Flow_Diagram.png)


## ⚙️ Configuration

Edit `config/config.yaml` to customize:
- Analysis thresholds
- Logging settings
- Repository settings
- Test framework configurations

Example configuration:
```yaml
analyzers:
  string_matcher:
    enabled: true
    min_score: 0.5
  binary_search:
    enabled: true
    test_retry_count: 3
  performance:
    enabled: true
    regression_threshold: 0.05
```

## 📊 Sample Output

### HTML Report
```html
<h2>Analysis Results</h2>
<p>Found 3 suspicious commits:</p>
<ul>
    <li>commit abc123: Performance regression in memory usage</li>
    <li>commit def456: Test failure in TestClass.java</li>
    <li>commit ghi789: Suspicious code pattern detected</li>
</ul>
```

## 🔍 Advanced Usage

### Custom Test Integration

```python
def custom_test_runner(commit: str) -> bool:
    # Implement custom test logic
    return test_passed

analyzer.find_problematic_commit(
    good_commit="abc123",
    bad_commit="def456",
    test_runner=custom_test_runner
)
```

### Performance Analysis

```python
analyzer = PerformanceAnalyzer(git_collector, config)
results = analyzer.analyze_performance_regression(
    good_metrics=[1.2, 1.3, 1.1],
    bad_metrics=[1.8, 1.9, 1.7]
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 📮 Contact

Your Name - [Prashantkumar Khatri](https://www.linkedin.com/in/prashantkumar-khatri/)
Project Link: [https://github.com/ShantKhatri/CommitHunter](https://github.com/ShantKhatri/CommitHunter)

## 🙏 Acknowledgments

- Eclipse OpenJ9 project for test cases
- GitPython library
- All contributors and testers