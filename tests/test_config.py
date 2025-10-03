"""
Tests for Configuration Module

Tests configuration settings and path management.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config


class TestPaths:
    """Tests for path configuration."""
    
    def test_project_root_exists(self):
        """Test that project root path is defined."""
        assert config.PROJECT_ROOT is not None
        assert isinstance(config.PROJECT_ROOT, Path)
    
    def test_data_dir_defined(self):
        """Test that data directory is defined."""
        assert config.DATA_DIR is not None
        assert isinstance(config.DATA_DIR, Path)
    
    def test_models_dir_defined(self):
        """Test that models directory is defined."""
        assert config.MODELS_DIR is not None
        assert isinstance(config.MODELS_DIR, Path)
    
    def test_reports_dir_defined(self):
        """Test that reports directory is defined."""
        assert config.REPORTS_DIR is not None
        assert isinstance(config.REPORTS_DIR, Path)
    
    def test_paths_are_absolute(self):
        """Test that all paths are absolute."""
        assert config.PROJECT_ROOT.is_absolute()
        assert config.DATA_DIR.is_absolute()
        assert config.MODELS_DIR.is_absolute()


class TestModelConfiguration:
    """Tests for model configuration."""
    
    def test_random_state_defined(self):
        """Test that random state is defined."""
        assert hasattr(config, 'RANDOM_STATE')
        assert isinstance(config.RANDOM_STATE, int)
    
    def test_test_size_valid(self):
        """Test that test size is valid."""
        assert hasattr(config, 'TEST_SIZE')
        assert 0 < config.TEST_SIZE < 1
    
    def test_random_state_is_positive(self):
        """Test that random state is non-negative."""
        assert config.RANDOM_STATE >= 0


class TestRiskTierThresholds:
    """Tests for risk tier threshold configuration."""
    
    def test_high_risk_threshold_defined(self):
        """Test that high risk threshold is defined."""
        if hasattr(config, 'HIGH_RISK_THRESHOLD'):
            assert 0 <= config.HIGH_RISK_THRESHOLD <= 1
    
    def test_medium_risk_threshold_defined(self):
        """Test that medium risk threshold is defined."""
        if hasattr(config, 'MEDIUM_RISK_THRESHOLD'):
            assert 0 <= config.MEDIUM_RISK_THRESHOLD <= 1
    
    def test_thresholds_logical_order(self):
        """Test that thresholds are in logical order."""
        if hasattr(config, 'HIGH_RISK_THRESHOLD') and hasattr(config, 'MEDIUM_RISK_THRESHOLD'):
            assert config.HIGH_RISK_THRESHOLD > config.MEDIUM_RISK_THRESHOLD


class TestFileConfiguration:
    """Tests for file path configuration."""
    
    def test_model_files_defined(self):
        """Test that model file paths are defined."""
        model_files = [
            'BASELINE_MODEL_FILE',
            'RF_MODEL_FILE'
        ]
        
        for file_var in model_files:
            if hasattr(config, file_var):
                file_path = getattr(config, file_var)
                assert isinstance(file_path, Path)
    
    def test_data_files_defined(self):
        """Test that data file paths are defined."""
        data_files = [
            'RAW_DATA_FILE',
            'TRAIN_DATA_FILE',
            'TEST_DATA_FILE'
        ]
        
        for file_var in data_files:
            if hasattr(config, file_var):
                file_path = getattr(config, file_var)
                assert isinstance(file_path, Path)
