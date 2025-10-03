"""
Tests for Retention Strategy Module

Tests risk tier classification and retention action recommendations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retention_strategy import (
    classify_risk_tier,
    recommend_action,
    calculate_retention_value,
    generate_retention_report
)


class TestClassifyRiskTier:
    """Tests for risk tier classification."""
    
    def test_classify_high_risk(self):
        """Test high risk classification."""
        assert classify_risk_tier(0.70) == "HIGH"
        assert classify_risk_tier(0.85) == "HIGH"
        assert classify_risk_tier(0.99) == "HIGH"
        assert classify_risk_tier(1.0) == "HIGH"
    
    def test_classify_medium_risk(self):
        """Test medium risk classification."""
        assert classify_risk_tier(0.40) == "MEDIUM"
        assert classify_risk_tier(0.55) == "MEDIUM"
        assert classify_risk_tier(0.69) == "MEDIUM"
    
    def test_classify_low_risk(self):
        """Test low risk classification."""
        assert classify_risk_tier(0.39) == "LOW"
        assert classify_risk_tier(0.20) == "LOW"
        assert classify_risk_tier(0.05) == "LOW"
        assert classify_risk_tier(0.0) == "LOW"
    
    def test_classify_boundary_values(self):
        """Test boundary values."""
        assert classify_risk_tier(0.7) == "HIGH"
        assert classify_risk_tier(0.699) == "MEDIUM"
        assert classify_risk_tier(0.4) == "MEDIUM"
        assert classify_risk_tier(0.399) == "LOW"
    
    def test_classify_risk_returns_string(self):
        """Test that function returns a string."""
        result = classify_risk_tier(0.5)
        assert isinstance(result, str)


class TestRecommendAction:
    """Tests for action recommendations."""
    
    @pytest.fixture
    def high_value_customer(self):
        """Create a high-value customer profile."""
        return {
            'MonthlyCharges': 100.0,
            'tenure': 36,
            'Contract': 'One year'
        }
    
    @pytest.fixture
    def low_value_customer(self):
        """Create a low-value customer profile."""
        return {
            'MonthlyCharges': 30.0,
            'tenure': 48,
            'Contract': 'Two year'
        }
    
    @pytest.fixture
    def new_customer(self):
        """Create a new customer profile."""
        return {
            'MonthlyCharges': 50.0,
            'tenure': 6,
            'Contract': 'Month-to-month'
        }
    
    def test_recommend_action_returns_dict(self, high_value_customer):
        """Test that recommendation returns a dictionary."""
        recommendation = recommend_action("HIGH", high_value_customer)
        
        assert isinstance(recommendation, dict)
    
    def test_recommend_action_contains_required_keys(self, high_value_customer):
        """Test that recommendation contains all required keys."""
        recommendation = recommend_action("MEDIUM", high_value_customer)
        
        required_keys = [
            'risk_tier', 'priority', 'action', 'channel',
            'discount_percentage', 'estimated_cost'
        ]
        
        for key in required_keys:
            assert key in recommendation, f"Missing key: {key}"
    
    def test_high_risk_gets_priority_1(self, high_value_customer):
        """Test that high risk customers get priority 1."""
        recommendation = recommend_action("HIGH", high_value_customer)
        
        assert recommendation['priority'] == 1
        assert recommendation['channel'] == 'Phone Call'
    
    def test_medium_risk_gets_priority_2(self, high_value_customer):
        """Test that medium risk customers get priority 2."""
        recommendation = recommend_action("MEDIUM", high_value_customer)
        
        assert recommendation['priority'] == 2
        assert recommendation['channel'] == 'Email'
    
    def test_low_risk_gets_priority_3(self, low_value_customer):
        """Test that low risk customers get priority 3."""
        recommendation = recommend_action("LOW", low_value_customer)
        
        assert recommendation['priority'] == 3
        assert recommendation['channel'] == 'Newsletter'
    
    def test_high_risk_high_value_gets_higher_discount(self, high_value_customer):
        """Test that high-value customers get better discounts."""
        recommendation = recommend_action("HIGH", high_value_customer)
        
        assert recommendation['discount_percentage'] >= 15
        assert recommendation['estimated_cost'] > 0
    
    def test_medium_risk_no_discount(self, high_value_customer):
        """Test that medium risk customers don't get discounts."""
        recommendation = recommend_action("MEDIUM", high_value_customer)
        
        assert recommendation['discount_percentage'] == 0
    
    def test_low_risk_low_cost(self, low_value_customer):
        """Test that low risk actions are low cost."""
        recommendation = recommend_action("LOW", low_value_customer)
        
        assert recommendation['estimated_cost'] <= 5
        assert recommendation['discount_percentage'] == 0
    
    def test_month_to_month_contract_recommendation(self):
        """Test specific recommendations for month-to-month customers."""
        customer = {
            'MonthlyCharges': 60.0,
            'tenure': 24,
            'Contract': 'Month-to-month'
        }
        
        recommendation = recommend_action("HIGH", customer)
        
        # Should encourage contract upgrade
        assert 'contract' in recommendation['action'].lower() or 'upgrade' in recommendation['action'].lower()
    
    def test_new_customer_recommendation(self, new_customer):
        """Test specific recommendations for new customers."""
        recommendation = recommend_action("MEDIUM", new_customer)
        
        # Action should be appropriate for new customers
        assert recommendation['action'] is not None
        assert len(recommendation['action']) > 0


class TestCalculateRetentionValue:
    """Tests for retention value calculation."""
    
    def test_calculate_retention_value_basic(self):
        """Test basic retention value calculation."""
        customer = {
            'MonthlyCharges': 100.0,
            'tenure': 24,
            'Contract': 'One year'
        }
        
        value = calculate_retention_value(customer, months_retained=12)
        
        # Base value: 100 * 12 = 1200
        assert value > 0
        assert isinstance(value, (int, float))
    
    def test_calculate_retention_value_month_to_month_lower(self):
        """Test that month-to-month contracts have lower retention value."""
        customer_mtm = {
            'MonthlyCharges': 100.0,
            'tenure': 24,
            'Contract': 'Month-to-month'
        }
        
        customer_long = {
            'MonthlyCharges': 100.0,
            'tenure': 24,
            'Contract': 'Two year'
        }
        
        value_mtm = calculate_retention_value(customer_mtm)
        value_long = calculate_retention_value(customer_long)
        
        # Two year contract should have higher retention value
        assert value_long > value_mtm
    
    def test_calculate_retention_value_tenure_effect(self):
        """Test that longer tenure increases retention value."""
        customer_new = {
            'MonthlyCharges': 100.0,
            'tenure': 3,
            'Contract': 'One year'
        }
        
        customer_established = {
            'MonthlyCharges': 100.0,
            'tenure': 36,
            'Contract': 'One year'
        }
        
        value_new = calculate_retention_value(customer_new)
        value_established = calculate_retention_value(customer_established)
        
        # Established customer should have higher value
        assert value_established > value_new
    
    def test_calculate_retention_value_with_different_months(self):
        """Test retention value calculation with different retention periods."""
        customer = {
            'MonthlyCharges': 100.0,
            'tenure': 24,
            'Contract': 'One year'
        }
        
        value_6 = calculate_retention_value(customer, months_retained=6)
        value_12 = calculate_retention_value(customer, months_retained=12)
        value_24 = calculate_retention_value(customer, months_retained=24)
        
        # Longer retention should have higher value
        assert value_6 < value_12 < value_24
    
    def test_calculate_retention_value_proportional_to_charges(self):
        """Test that retention value is proportional to monthly charges."""
        customer_low = {
            'MonthlyCharges': 50.0,
            'tenure': 24,
            'Contract': 'One year'
        }
        
        customer_high = {
            'MonthlyCharges': 150.0,
            'tenure': 24,
            'Contract': 'One year'
        }
        
        value_low = calculate_retention_value(customer_low)
        value_high = calculate_retention_value(customer_high)
        
        # Higher charges should yield higher value (approximately 3x)
        assert value_high > value_low * 2


class TestGenerateRetentionReport:
    """Tests for retention report generation."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample customer predictions."""
        np.random.seed(42)
        
        predictions = []
        for i in range(50):
            pred = {
                'customerID': f'CUST-{i:04d}',
                'churn_probability': np.random.random(),
                'MonthlyCharges': np.random.uniform(20, 120),
                'tenure': np.random.randint(1, 72),
                'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year']),
                'gender': np.random.choice(['Male', 'Female']),
                'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'])
            }
            predictions.append(pred)
        
        return pd.DataFrame(predictions)
    
    def test_generate_retention_report_returns_dataframe(self, sample_predictions):
        """Test that report generation returns a DataFrame."""
        report = generate_retention_report(sample_predictions)
        
        assert isinstance(report, pd.DataFrame)
    
    def test_generate_retention_report_contains_required_columns(self, sample_predictions):
        """Test that report contains required columns."""
        report = generate_retention_report(sample_predictions)
        
        required_columns = [
            'customerID', 'churn_probability', 'risk_tier',
            'priority', 'recommended_action', 'retention_value'
        ]
        
        for col in required_columns:
            assert col in report.columns, f"Missing column: {col}"
    
    def test_generate_retention_report_sorted_by_priority(self, sample_predictions):
        """Test that report is sorted by priority and probability."""
        report = generate_retention_report(sample_predictions)
        
        # High priority (1) customers should be first
        first_priorities = report['priority'].head(10).values
        assert all(p <= 2 for p in first_priorities)
    
    def test_generate_retention_report_high_risk_count(self, sample_predictions):
        """Test that high risk customers are correctly identified."""
        # Add some high-risk customers
        sample_predictions.loc[0:5, 'churn_probability'] = 0.85
        
        report = generate_retention_report(sample_predictions)
        
        high_risk_count = (report['risk_tier'] == 'HIGH').sum()
        assert high_risk_count >= 6
    
    def test_generate_retention_report_retention_value_calculated(self, sample_predictions):
        """Test that retention value is calculated for all customers."""
        report = generate_retention_report(sample_predictions)
        
        # All customers should have a retention value
        assert report['retention_value'].notna().all()
        assert (report['retention_value'] > 0).all()
    
    def test_generate_retention_report_top_n_limit(self, sample_predictions):
        """Test that report respects top_n limit."""
        # Add many customers
        report = generate_retention_report(sample_predictions, top_n=10)
        
        # Should return at most 10 customers
        assert len(report) <= 10


class TestIntegration:
    """Integration tests for retention strategy."""
    
    def test_complete_retention_workflow(self):
        """Test complete retention strategy workflow."""
        # Create sample customer
        customer = {
            'customerID': 'CUST-001',
            'MonthlyCharges': 85.0,
            'tenure': 18,
            'Contract': 'Month-to-month',
            'gender': 'Female',
            'InternetService': 'Fiber optic'
        }
        
        # Simulate churn probability
        churn_prob = 0.75
        
        # Classify risk
        risk_tier = classify_risk_tier(churn_prob)
        assert risk_tier == "HIGH"
        
        # Get recommendation
        recommendation = recommend_action(risk_tier, customer)
        assert recommendation['priority'] == 1
        assert recommendation['discount_percentage'] > 0
        
        # Calculate retention value
        retention_value = calculate_retention_value(customer)
        assert retention_value > 0
        
        # Verify recommendation is cost-effective
        # Discount cost should be less than retention value
        assert recommendation['estimated_cost'] < retention_value
    
    def test_batch_retention_analysis(self):
        """Test retention analysis for multiple customers."""
        np.random.seed(42)
        
        # Create batch of customers
        customers = []
        for i in range(100):
            customer = {
                'customerID': f'CUST-{i:04d}',
                'churn_probability': np.random.random(),
                'MonthlyCharges': np.random.uniform(20, 120),
                'tenure': np.random.randint(1, 72),
                'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'])
            }
            customers.append(customer)
        
        df = pd.DataFrame(customers)
        
        # Generate report
        report = generate_retention_report(df)
        
        # Verify report
        assert len(report) > 0
        assert 'risk_tier' in report.columns
        assert 'recommended_action' in report.columns
        
        # High priority customers should be at the top
        assert report.iloc[0]['priority'] <= report.iloc[-1]['priority']
