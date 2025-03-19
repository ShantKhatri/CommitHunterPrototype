"""
Configuration Management Module

This module handles loading, validation, and access to configuration settings
for the CommitHunter application.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class AnalyzerConfig:
    """Configuration settings for analyzers"""
    string_matcher: Dict[str, Any]
    binary_search: Dict[str, Any]
    performance: Dict[str, Any]

@dataclass
class CollectorConfig:
    """Configuration settings for collectors"""
    git: Dict[str, Any]
    test: Dict[str, Any]

class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass

class Config:
    """
    Central configuration management class that handles loading and validating
    configuration settings from YAML files.
    """
    
    DEFAULT_CONFIG = {
        "analyzers": {
            "string_matcher": {
                "enabled": True,
                "min_score": 0.5,
                "max_results": 10,
                "keyword_weights": {
                    "message": 1.0,
                    "stacktrace": 0.8,
                    "code": 0.6
                }
            },
            "binary_search": {
                "enabled": True,
                "timeout": 600,
                "max_iterations": 20,
                "test_retry_count": 3,
                "test_retry_delay": 5
            },
            "performance": {
                "enabled": True,
                "regression_threshold": 0.05,
                "significance_level": 0.05,
                "min_samples": 3,
                "metric_fields": [
                    "execution_time",
                    "memory_usage"
                ]
            }
        },
        "collectors": {
            "git": {
                "clone_depth": 50,
                "timeout": 300,
                "cache_dir": ".git_cache"
            },
            "test": {
                "frameworks": ["junit", "pytest", "mocha"],
                "result_dir": "test_results",
                "pattern": "*.xml"
            }
        },
        "logging": {
            "level": "INFO",
            "file": "commit_hunter.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "max_size": 10485760,  # 10MB
            "backup_count": 5
        },
        "openj9": {
            "issue_url_pattern": "https://github.com/eclipse-openj9/openj9/issues/{issue_number}",
            "risk_thresholds": {
                "high": 0.7,
                "medium": 0.4
            },
            "test_patterns": {
                "system_tests": ["systemtest", "system-test"],
                "functional_tests": ["functional", "fvtest"],
                "perf_tests": ["performance", "throughput", "footprint"]
            },
            "component_mapping": {
                "jit": ["jit_", "compiler", "codegen"],
                "gc": ["gc_", "mm_"],
                "thread": ["thread_", "j9thread"],
                "classloader": ["classloader", "bootstrap"],
                "interpreter": ["interp_"],
                "runtime": ["runtime", "vm_"]
            }
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Raises:
            ConfigurationError: If configuration file is invalid or cannot be loaded
        """
        try:
            if not os.path.exists(config_path):
                raise ConfigurationError(f"Configuration file not found: {config_path}")
                
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                
            if not isinstance(yaml_config, dict):
                raise ConfigurationError("Invalid configuration format")
                
            # Merge with default config
            self._merge_configs(self.config, yaml_config)
            
            # Validate the merged configuration
            self._validate_config(self.config)
            
            self.logger.info(f"Successfully loaded configuration from {config_path}")
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {str(e)}")
    
    def _merge_configs(self, base: Dict, override: Dict) -> None:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
        """
        for key, value in override.items():
            if (key in base and isinstance(base[key], dict) 
                and isinstance(value, dict)):
                self._merge_configs(base[key], value)
            else:
                base[key] = value
    
    def _validate_config(self, config: Dict) -> None:
        """
        Validate configuration structure and values.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        required_sections = ['analyzers', 'collectors', 'logging']
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                raise ConfigurationError(f"Missing required configuration section: {section}")
        
        # Validate analyzer configurations
        analyzers = config['analyzers']
        for analyzer in ['string_matcher', 'binary_search', 'performance']:
            if analyzer not in analyzers:
                raise ConfigurationError(f"Missing analyzer configuration: {analyzer}")
            if 'enabled' not in analyzers[analyzer]:
                raise ConfigurationError(f"Missing 'enabled' flag for analyzer: {analyzer}")
        
        # Validate collector configurations
        collectors = config['collectors']
        for collector in ['git', 'test']:
            if collector not in collectors:
                raise ConfigurationError(f"Missing collector configuration: {collector}")
        
        # Validate logging configuration
        logging_config = config['logging']
        required_logging_fields = ['level', 'file', 'format']
        for field in required_logging_fields:
            if field not in logging_config:
                raise ConfigurationError(f"Missing logging configuration field: {field}")
    
    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Configuration key path (e.g., 'analyzers.string_matcher.enabled')
            default: Default value to return if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        try:
            value = self.config
            for key in key_path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_analyzer_config(self, analyzer_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific analyzer.
        
        Args:
            analyzer_name: Name of the analyzer
            
        Returns:
            Analyzer configuration dictionary
        
        Raises:
            ConfigurationError: If analyzer configuration doesn't exist
        """
        try:
            return self.config['analyzers'][analyzer_name]
        except KeyError:
            raise ConfigurationError(f"Configuration not found for analyzer: {analyzer_name}")
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to a YAML file.
        
        Args:
            output_path: Path to save the configuration file
            
        Raises:
            ConfigurationError: If configuration cannot be saved
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                yaml.safe_dump(self.config, f, default_flow_style=False)
                
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {str(e)}")

# Global configuration instance
_config_instance = None

def init_config(config_path: Optional[str] = None) -> Config:
    """
    Initialize global configuration instance.
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Config instance
    """
    global _config_instance
    _config_instance = Config(config_path)
    return _config_instance

def get_config() -> Config:
    """
    Get global configuration instance.
    
    Returns:
        Config instance
        
    Raises:
        RuntimeError: If configuration hasn't been initialized
    """
    if _config_instance is None:
        raise RuntimeError("Configuration not initialized. Call init_config first.")
    return _config_instance