# Test Suite for Customer Churn Prediction

This directory contains comprehensive tests for the customer churn prediction system.

## 📁 Test Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Pytest configuration and shared fixtures
├── test_data_loader.py           # Tests for data loading and preprocessing
├── test_feature_engineering.py   # Tests for feature engineering pipeline
├── test_models.py                # Tests for model training and evaluation
├── test_retention_strategy.py    # Tests for retention recommendations
├── test_api.py                   # Tests for Flask API endpoints
├── test_config.py                # Tests for configuration module
└── README.md                     # This file
```

## 🚀 Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test File
```bash
pytest tests/test_data_loader.py
pytest tests/test_models.py
```

### Run Tests with Coverage
```bash
pytest tests/ --cov=src --cov=app --cov-report=html
```

### Run Tests with Verbose Output
```bash
pytest tests/ -v
```

### Run Specific Test Class or Function
```bash
pytest tests/test_models.py::TestTrainBaselineModel
pytest tests/test_api.py::TestPredictEndpoint::test_predict_returns_200_with_valid_data
```

## 📊 Test Coverage

The test suite covers:

### Data Loading (`test_data_loader.py`)
- ✅ Loading raw data from CSV
- ✅ Data validation and quality checks
- ✅ Data cleaning and preprocessing
- ✅ Train/validation/test splitting
- ✅ Stratification preservation
- ✅ Data persistence

### Feature Engineering (`test_feature_engineering.py`)
- ✅ Tenure feature creation
- ✅ Spending pattern features
- ✅ Service usage features
- ✅ Contract and payment features
- ✅ Categorical encoding
- ✅ Numerical scaling
- ✅ Transform consistency
- ✅ Model persistence (save/load)

### Model Training (`test_models.py`)
- ✅ Baseline logistic regression training
- ✅ Random forest training
- ✅ Model evaluation metrics (ROC-AUC, precision, recall, F1)
- ✅ Top-decile precision calculation
- ✅ Confusion matrix generation
- ✅ Model persistence (save/load)
- ✅ Prediction consistency

### Retention Strategy (`test_retention_strategy.py`)
- ✅ Risk tier classification (HIGH/MEDIUM/LOW)
- ✅ Action recommendations by risk tier
- ✅ Personalized recommendations based on customer profile
- ✅ Retention value calculation
- ✅ Batch report generation
- ✅ Priority sorting

### API Endpoints (`test_api.py`)
- ✅ Home endpoint
- ✅ Health check endpoint
- ✅ Model info endpoint
- ✅ Single customer prediction
- ✅ Batch prediction
- ✅ Error handling
- ✅ Request/response validation
- ✅ Risk tier mapping

### Configuration (`test_config.py`)
- ✅ Path configuration
- ✅ Model parameters
- ✅ Risk tier thresholds
- ✅ File paths

## 🔧 Test Fixtures

Common fixtures are defined in `conftest.py`:

- `project_root`: Project root directory path
- `data_dir`: Data directory path
- `raw_data_path`: Path to raw data file
- `models_dir`: Models directory path
- `sample_customer_data`: Single customer record
- `sample_dataframe`: DataFrame with 100 sample customers
- `sample_features_dataframe`: Features-only DataFrame
- `temp_dir`: Temporary directory for test files
- `mock_model`: Simple trained model for testing

## 📝 Writing New Tests

### Test Naming Convention
- Test files: `test_<module_name>.py`
- Test classes: `Test<FeatureName>`
- Test functions: `test_<what_is_being_tested>`

### Example Test Structure
```python
class TestFeatureName:
    """Tests for specific feature."""
    
    def test_basic_functionality(self, fixture_name):
        """Test basic functionality."""
        # Arrange
        input_data = create_test_data()
        
        # Act
        result = function_to_test(input_data)
        
        # Assert
        assert result == expected_output
```

### Best Practices
1. **Arrange-Act-Assert**: Structure tests clearly
2. **One assertion per test**: Keep tests focused
3. **Use descriptive names**: Test names should explain what they test
4. **Use fixtures**: Reuse common test data and setup
5. **Test edge cases**: Include boundary conditions and error cases
6. **Keep tests independent**: Tests should not depend on each other
7. **Mock external dependencies**: Use mocks for APIs, databases, etc.

## 🐛 Debugging Tests

### Run with Print Statements
```bash
pytest tests/ -s
```

### Run Failed Tests Only
```bash
pytest tests/ --lf
```

### Run with Detailed Traceback
```bash
pytest tests/ --tb=long
```

### Run Tests in Parallel (faster)
```bash
pytest tests/ -n auto
```

## 📈 Continuous Integration

Tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest tests/ --cov=src --cov=app --cov-report=xml
```

## 🎯 Test Metrics

Target coverage: **>80%** for all modules

Current test count by module:
- Data Loader: 25+ tests
- Feature Engineering: 30+ tests
- Models: 25+ tests
- Retention Strategy: 30+ tests
- API: 35+ tests
- Config: 10+ tests

**Total: 155+ comprehensive tests**

## 🔍 Troubleshooting

### Common Issues

**Issue: Tests can't find modules**
```bash
# Solution: Ensure pytest is run from project root
cd customer-churn-prediction
pytest tests/
```

**Issue: Tests fail due to missing data**
```bash
# Solution: Some tests require raw data file
# Check if data/raw/Telco-Customer-Churn.csv exists
# Tests will skip if data is not available
```

**Issue: Model loading tests fail**
```bash
# Solution: Ensure models are trained
python -m src.models  # Train models first
pytest tests/test_models.py
```

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Python Testing with pytest](https://pragprog.com/titles/bopytest/python-testing-with-pytest/)

## 🤝 Contributing Tests

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain >80% coverage
4. Update this README if adding new test files
