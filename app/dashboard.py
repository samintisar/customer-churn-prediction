"""
Customer Churn Prediction Dashboard

Streamlit application for interactive churn risk analysis and retention planning.

Features:
- Upload or load customer data
- View churn risk scores
- Filter by risk tier, contract type, tenure
- Display top N at-risk customers
- Show recommended retention actions
- Interactive charts (risk distribution, feature importance)
- Download at-risk customer list

Pages:
1. Overview: Summary metrics and distribution
2. At-Risk Customers: Prioritized list with actions
3. Model Performance: Evaluation metrics and explainability
4. Individual Prediction: Single customer risk assessment

Usage:
    streamlit run app/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import joblib
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models import load_model, evaluate_model
from src.retention_strategy import classify_risk_tier, recommend_action, generate_retention_report
from src.feature_engineering import FeatureEngineer
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix


# Global session state for data and model caching
@st.cache_resource
def load_trained_model():
    """Load the trained churn prediction model."""
    model_path = Path(__file__).parent.parent / "models" / "churn_predictor.pkl"
    if not model_path.exists():
        st.error(f"❌ Model not found at {model_path}")
        return None
    return load_model(str(model_path))


@st.cache_resource
def load_feature_engineer():
    """Load the fitted feature engineer."""
    fe_path = Path(__file__).parent.parent / "models" / "feature_engineer.pkl"
    if fe_path.exists():
        return FeatureEngineer.load(str(fe_path))
    return None


@st.cache_data
def load_test_data():
    """Load test data for predictions."""
    test_path = Path(__file__).parent.parent / "data" / "processed" / "test.csv"
    if not test_path.exists():
        st.error(f"❌ Test data not found at {test_path}")
        return None
    return pd.read_csv(test_path)


@st.cache_data
def load_original_data():
    """Load original customer data (before feature engineering)."""
    clean_path = Path(__file__).parent.parent / "data" / "processed" / "cleaned_data.csv"
    if clean_path.exists():
        return pd.read_csv(clean_path)
    
    # Fallback to raw data
    raw_path = Path(__file__).parent.parent / "data" / "raw" / "Telco-Customer-Churn.csv"
    if raw_path.exists():
        return pd.read_csv(raw_path)
    
    return None


def generate_predictions(model, X_data):
    """Generate churn predictions and risk tiers."""
    # Get predictions
    churn_probabilities = model.predict_proba(X_data)[:, 1]
    
    # Classify risk tiers
    risk_tiers = [classify_risk_tier(prob) for prob in churn_probabilities]
    
    return churn_probabilities, risk_tiers


def main():
    """
    Main dashboard application.
    """
    st.set_page_config(
        page_title="Customer Churn Prediction",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Customer Churn Prediction & Retention Dashboard")
    st.markdown("---")
    
    # Load model and data
    model = load_trained_model()
    test_data = load_test_data()
    original_data = load_original_data()
    
    if model is None or test_data is None:
        st.error("⚠️ Cannot load model or data. Please ensure files exist.")
        st.stop()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Dashboard Settings")
    
    # Data source selection
    st.sidebar.subheader("Data Source")
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Test Data", "Upload New Data"]
    )
    
    # Handle data upload
    if data_source == "Upload New Data":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload customer data for prediction"
        )
        
        if uploaded_file is not None:
            try:
                custom_data = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Loaded {len(custom_data)} customers")
                # Use custom data (would need feature engineering)
                st.sidebar.warning("⚠️ Custom data upload requires feature engineering implementation")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {str(e)}")
    
    # Risk tier threshold customization
    st.sidebar.subheader("Risk Tier Thresholds")
    high_threshold = st.sidebar.slider(
        "High Risk Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.70,
        step=0.05,
        help="Churn probability threshold for high risk"
    )
    medium_threshold = st.sidebar.slider(
        "Medium Risk Threshold",
        min_value=0.0,
        max_value=high_threshold,
        value=0.40,
        step=0.05,
        help="Churn probability threshold for medium risk"
    )
    
    # Generate predictions for test data (already preprocessed with 46 features)
    X_test = test_data.drop('Churn', axis=1, errors='ignore')
    y_test = test_data.get('Churn', None)
    
    churn_probabilities, risk_tiers = generate_predictions(model, X_test)
    
    # Apply custom thresholds
    risk_tiers_custom = []
    for prob in churn_probabilities:
        if prob >= high_threshold:
            risk_tiers_custom.append("HIGH")
        elif prob >= medium_threshold:
            risk_tiers_custom.append("MEDIUM")
        else:
            risk_tiers_custom.append("LOW")
    
    # Create results dataframe using original data for display (raw features are more readable)
    # Need to recreate the split to align with test data
    if original_data is not None:
        from sklearn.model_selection import train_test_split
        
        # Drop customerID if it exists
        original_for_split = original_data.drop('customerID', axis=1, errors='ignore')
        
        # Separate features and target
        X_orig = original_for_split.drop('Churn', axis=1, errors='ignore')
        y_orig = original_for_split['Churn'] if 'Churn' in original_for_split.columns else None
        
        if y_orig is not None:
            # Recreate the same split as in the pipeline (random_state=42, test_size=0.2)
            _, X_test_raw, _, y_test_raw = train_test_split(
                X_orig, y_orig, test_size=0.2, random_state=42, stratify=y_orig
            )
            results_df = X_test_raw.reset_index(drop=True).copy()
        else:
            # Fallback to using preprocessed features
            results_df = X_test.copy()
    else:
        # Fallback to using preprocessed features if original data not available
        results_df = X_test.copy()
    
    # Add predictions
    results_df['churn_probability'] = churn_probabilities
    results_df['risk_tier'] = risk_tiers_custom
    if y_test is not None:
        results_df['actual_churn'] = y_test.values
    
    # Load feature engineer for individual predictions
    feature_engineer = load_feature_engineer()
    
    # Store in session state
    st.session_state['results_df'] = results_df
    st.session_state['model'] = model
    st.session_state['X_test'] = X_test  # Preprocessed features for model
    st.session_state['y_test'] = y_test
    st.session_state['feature_engineer'] = feature_engineer
    st.session_state['original_data'] = original_data
    
    # Create tabs for different pages
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Overview",
        "🚨 At-Risk Customers",
        "📊 Model Performance",
        "🔍 Individual Prediction"
    ])
    
    with tab1:
        render_overview_page(results_df, model, X_test)
    
    with tab2:
        render_atrisk_customers_page(results_df)
    
    with tab3:
        render_model_performance_page(model, X_test, y_test, churn_probabilities)
    
    with tab4:
        render_individual_prediction_page(model, original_data)


def render_overview_page(results_df, model, X_test):
    """
    Render the overview page with summary metrics.
    
    Displays:
    - Total customers analyzed
    - High/medium/low risk counts
    - Churn probability distribution
    - Top risk factors
    """
    st.header("📈 Overview Dashboard")
    
    # Summary Metrics
    st.subheader("Summary Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_customers = len(results_df)
    high_risk_count = (results_df['risk_tier'] == 'HIGH').sum()
    medium_risk_count = (results_df['risk_tier'] == 'MEDIUM').sum()
    low_risk_count = (results_df['risk_tier'] == 'LOW').sum()
    avg_churn_prob = results_df['churn_probability'].mean()
    
    with col1:
        st.metric(
            label="Total Customers",
            value=f"{total_customers:,}",
            help="Total number of customers analyzed"
        )
    
    with col2:
        st.metric(
            label="🔴 High Risk",
            value=f"{high_risk_count:,}",
            delta=f"{high_risk_count/total_customers:.1%}",
            delta_color="inverse",
            help="Customers with high churn probability"
        )
    
    with col3:
        st.metric(
            label="🟠 Medium Risk",
            value=f"{medium_risk_count:,}",
            delta=f"{medium_risk_count/total_customers:.1%}",
            delta_color="off",
            help="Customers with medium churn probability"
        )
    
    with col4:
        st.metric(
            label="🟢 Low Risk",
            value=f"{low_risk_count:,}",
            delta=f"{low_risk_count/total_customers:.1%}",
            delta_color="normal",
            help="Customers with low churn probability"
        )
    
    with col5:
        st.metric(
            label="Avg Churn Probability",
            value=f"{avg_churn_prob:.1%}",
            help="Average predicted churn probability"
        )
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution")
        
        # Pie chart for risk tiers
        risk_counts = results_df['risk_tier'].value_counts()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.3,
            marker=dict(colors=['#ff4444', '#ff9944', '#44ff44']),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig_pie.update_layout(
            title="Customer Risk Tier Distribution",
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Churn Probability Distribution")
        
        # Histogram with risk tier thresholds
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=results_df['churn_probability'],
            nbinsx=50,
            name='Churn Probability',
            marker_color='steelblue',
            opacity=0.7
        ))
        
        # Add threshold lines
        fig_hist.add_vline(
            x=0.70, line_dash="dash", line_color="red",
            annotation_text="High Risk", annotation_position="top"
        )
        fig_hist.add_vline(
            x=0.40, line_dash="dash", line_color="orange",
            annotation_text="Medium Risk", annotation_position="top"
        )
        
        fig_hist.update_layout(
            title="Distribution of Churn Probabilities",
            xaxis_title="Churn Probability",
            yaxis_title="Number of Customers",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Feature Importance
    st.subheader("Top Risk Factors (Feature Importance)")
    
    # Extract feature importance from model
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        fig_importance = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Features Influencing Churn Risk',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='reds'
        )
        
        fig_importance.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
    elif hasattr(model, 'coef_'):
        # For logistic regression
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False).head(10)
        
        fig_importance = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Features Influencing Churn Risk (Coefficient Magnitude)',
            labels={'importance': 'Coefficient Magnitude', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='reds'
        )
        
        fig_importance.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type")


def render_atrisk_customers_page(results_df):
    """
    Render the at-risk customers page with advanced filtering and customer details.
    
    Features:
    - Advanced filters (risk tier, probability, contract, tenure)
    - Sortable table with color gradients
    - Customer detail expanders with SHAP explanations
    - CSV export functionality
    """
    st.header("🚨 At-Risk Customers")
    
    # Sidebar Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Customer Filters")
    
    # Risk Tier Filter
    risk_filter = st.sidebar.multiselect(
        "Risk Tier",
        options=['HIGH', 'MEDIUM', 'LOW'],
        default=['HIGH', 'MEDIUM'],
        help="Select one or more risk tiers to display"
    )
    
    # Minimum Churn Probability Slider
    min_churn_prob = st.sidebar.slider(
        "Minimum Churn Probability",
        min_value=0,
        max_value=100,
        value=40,
        step=5,
        help="Show customers with churn probability above this threshold"
    ) / 100.0
    
    # Contract Type Filter
    if 'Contract' in results_df.columns:
        contract_options = results_df['Contract'].unique().tolist()
        contract_filter = st.sidebar.multiselect(
            "Contract Type",
            options=contract_options,
            default=contract_options,
            help="Filter by contract type"
        )
    else:
        contract_filter = None
    
    # Tenure Range Filter
    if 'tenure' in results_df.columns:
        min_tenure = int(results_df['tenure'].min())
        max_tenure = int(results_df['tenure'].max())
        tenure_range = st.sidebar.slider(
            "Tenure Range (months)",
            min_value=min_tenure,
            max_value=max_tenure,
            value=(min_tenure, max_tenure),
            help="Filter customers by tenure range"
        )
    else:
        tenure_range = None
    
    # Sort Options
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Sort Options")
    
    sort_by = st.sidebar.selectbox(
        "Sort By",
        options=['Churn Probability', 'Retention Value', 'Tenure', 'Monthly Charges'],
        index=0,
        help="Choose how to sort the customer list"
    )
    
    sort_order = st.sidebar.radio(
        "Sort Order",
        options=['Descending', 'Ascending'],
        index=0,
        help="Sort in ascending or descending order"
    )
    
    # Display Options
    st.sidebar.markdown("---")
    st.sidebar.subheader("👁️ Display Options")
    
    display_limit = st.sidebar.selectbox(
        "Show Top N Customers",
        options=[10, 25, 50, 100, 'All'],
        index=2,
        help="Number of customers to display"
    )
    
    # Apply Filters
    filtered_df = results_df.copy()
    
    # Risk tier filter
    if risk_filter:
        filtered_df = filtered_df[filtered_df['risk_tier'].isin(risk_filter)]
    
    # Minimum probability filter
    filtered_df = filtered_df[filtered_df['churn_probability'] >= min_churn_prob]
    
    # Contract filter
    if contract_filter and 'Contract' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Contract'].isin(contract_filter)]
    
    # Tenure filter
    if tenure_range and 'tenure' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['tenure'] >= tenure_range[0]) & 
            (filtered_df['tenure'] <= tenure_range[1])
        ]
    
    # Calculate retention values
    from src.retention_strategy import calculate_retention_value
    
    # Check if we have any customers after filtering
    if len(filtered_df) == 0:
        st.warning("⚠️ No customers match the current filters. Try adjusting your filter criteria.")
        return
    
    retention_values = []
    recommendations = []
    
    for idx, row in filtered_df.iterrows():
        customer_profile = {
            'MonthlyCharges': row.get('MonthlyCharges', 0),
            'tenure': row.get('tenure', 0),
            'Contract': row.get('Contract', 'Month-to-month'),
            'TotalCharges': row.get('TotalCharges', 0)
        }
        
        # Calculate retention value
        ret_value = calculate_retention_value(customer_profile)
        retention_values.append(ret_value)
        
        # Get recommendation
        rec = recommend_action(row['risk_tier'], customer_profile)
        recommendations.append(rec)
    
    filtered_df['Retention_Value'] = retention_values
    rec_df = pd.DataFrame(recommendations)
    filtered_df['Recommended_Action'] = rec_df['action'].values
    filtered_df['Contact_Channel'] = rec_df['channel'].values
    filtered_df['Priority'] = rec_df['priority'].values
    filtered_df['Discount_Pct'] = rec_df['discount_percentage'].values
    
    # Apply Sorting
    sort_column_map = {
        'Churn Probability': 'churn_probability',
        'Retention Value': 'Retention_Value',
        'Tenure': 'tenure',
        'Monthly Charges': 'MonthlyCharges'
    }
    
    sort_col = sort_column_map[sort_by]
    sort_ascending = (sort_order == 'Ascending')
    
    if sort_col in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_col, ascending=sort_ascending)
    
    # Apply Display Limit
    if display_limit != 'All':
        filtered_df = filtered_df.head(display_limit)
    
    # Summary Metrics
    st.subheader("📊 Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Customers Found",
            f"{len(filtered_df):,}",
            help="Number of customers matching filters"
        )
    
    with col2:
        if len(filtered_df) > 0:
            avg_churn = filtered_df['churn_probability'].mean()
            st.metric(
                "Avg Churn Probability",
                f"{avg_churn:.1%}",
                help="Average churn probability of filtered customers"
            )
    
    with col3:
        if len(filtered_df) > 0:
            total_retention_value = filtered_df['Retention_Value'].sum()
            st.metric(
                "Total Retention Value",
                f"${total_retention_value:,.0f}",
                help="Total estimated value of retaining these customers"
            )
    
    with col4:
        if len(filtered_df) > 0:
            high_risk_count = (filtered_df['risk_tier'] == 'HIGH').sum()
            st.metric(
                "High Risk Count",
                f"{high_risk_count:,}",
                help="Number of high-risk customers in filtered list"
            )
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No customers match the current filter criteria. Try adjusting the filters.")
        return
    
    st.markdown("---")
    
    # Customer Table with Color Coding
    st.subheader("📋 Customer List")
    
    # Prepare display dataframe
    display_cols = []
    
    # Build column list
    if 'customerID' in filtered_df.columns:
        display_cols.append('customerID')
    
    display_cols.extend(['churn_probability', 'risk_tier'])
    
    if 'Contract' in filtered_df.columns:
        display_cols.append('Contract')
    if 'tenure' in filtered_df.columns:
        display_cols.append('tenure')
    if 'MonthlyCharges' in filtered_df.columns:
        display_cols.append('MonthlyCharges')
    
    display_cols.extend(['Recommended_Action', 'Retention_Value'])
    
    # Create formatted display dataframe
    display_df = filtered_df[display_cols].copy()
    
    # Color gradient function for churn probability
    def color_gradient_churn(val):
        """Apply red-to-green gradient based on churn probability."""
        if isinstance(val, str):
            return ''
        
        # Convert to 0-100 scale
        normalized = val * 100
        
        # Red (high churn) to Green (low churn)
        if normalized >= 70:
            return 'background-color: #ff4444; color: white; font-weight: bold'
        elif normalized >= 60:
            return 'background-color: #ff6666; color: white'
        elif normalized >= 50:
            return 'background-color: #ff9944; color: white'
        elif normalized >= 40:
            return 'background-color: #ffbb44; color: black'
        elif normalized >= 30:
            return 'background-color: #ffdd77; color: black'
        else:
            return 'background-color: #88ff88; color: black'
    
    # Risk tier badge styling
    def style_risk_tier(val):
        """Apply badge-style formatting to risk tiers."""
        if val == 'HIGH':
            return 'background-color: #ff4444; color: white; font-weight: bold; padding: 5px; border-radius: 5px'
        elif val == 'MEDIUM':
            return 'background-color: #ff9944; color: white; font-weight: bold; padding: 5px; border-radius: 5px'
        else:
            return 'background-color: #44ff44; color: black; font-weight: bold; padding: 5px; border-radius: 5px'
    
    # Display table with basic styling (Streamlit limitation on custom styling)
    st.dataframe(
        display_df.style.applymap(
            color_gradient_churn, 
            subset=['churn_probability']
        ).applymap(
            style_risk_tier,
            subset=['risk_tier']
        ).format({
            'churn_probability': '{:.1%}',
            'MonthlyCharges': '${:.2f}' if 'MonthlyCharges' in display_df.columns else '{}',
            'Retention_Value': '${:,.0f}',
            'tenure': '{:.0f}'
        }),
        use_container_width=True,
        height=400
    )
    
    st.markdown("---")
    
    # Customer Detail Expanders
    st.subheader("🔍 Customer Details")
    st.info("💡 Click on a customer below to view detailed information and SHAP explanation")
    
    # Show top 10 for detail view
    detail_df = filtered_df.head(10)
    
    for idx, row in detail_df.iterrows():
        customer_id = row.get('customerID', f'Customer #{idx}')
        churn_prob = row['churn_probability']
        risk_tier = row['risk_tier']
        
        # Create expander with colored header
        risk_emoji = "🔴" if risk_tier == "HIGH" else "🟠" if risk_tier == "MEDIUM" else "🟢"
        
        with st.expander(
            f"{risk_emoji} {customer_id} - {churn_prob:.1%} Churn Risk - {risk_tier} Priority",
            expanded=False
        ):
            # Customer overview
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📊 Risk Profile**")
                st.write(f"**Churn Probability:** {churn_prob:.1%}")
                st.write(f"**Risk Tier:** {risk_tier}")
                st.write(f"**Priority:** {row['Priority']}")
            
            with col2:
                st.markdown("**💼 Account Information**")
                if 'tenure' in row:
                    st.write(f"**Tenure:** {row['tenure']:.0f} months")
                if 'Contract' in row:
                    st.write(f"**Contract:** {row['Contract']}")
                if 'MonthlyCharges' in row:
                    st.write(f"**Monthly Charges:** ${row['MonthlyCharges']:.2f}")
            
            with col3:
                st.markdown("**💰 Retention Value**")
                st.write(f"**Est. Value:** ${row['Retention_Value']:,.0f}")
                st.write(f"**Discount:** {row['Discount_Pct']}%")
                st.write(f"**Channel:** {row['Contact_Channel']}")
            
            # Recommended Action
            st.markdown("**🎯 Recommended Action**")
            st.success(row['Recommended_Action'])
            
            # Feature values table
            st.markdown("**📋 All Features**")
            
            # Get all feature columns (exclude our computed columns)
            exclude_cols = ['churn_probability', 'risk_tier', 'Retention_Value', 'Recommended_Action', 
                          'Contact_Channel', 'Priority', 'Discount_Pct', 'actual_churn']
            
            feature_cols = [col for col in row.index if col not in exclude_cols]
            
            # Create feature value dataframe
            feature_data = []
            for col in feature_cols[:20]:  # Limit to first 20 features
                feature_data.append({
                    'Feature': col,
                    'Value': row[col]
                })
            
            feature_df = pd.DataFrame(feature_data)
            st.dataframe(feature_df, use_container_width=True, height=200)
            
            # SHAP Explanation (if available)
            st.markdown("**🔬 SHAP Explanation**")
            
            # Try to generate SHAP values
            try:
                model = st.session_state.get('model')
                X_test = st.session_state.get('X_test')
                
                if model is not None and X_test is not None:
                    import shap
                    
                    # Get the row index in X_test
                    test_idx = detail_df.index.get_loc(idx)
                    
                    # Create SHAP explainer (cached)
                    @st.cache_resource
                    def get_shap_explainer(_model, _X_sample):
                        """Create and cache SHAP explainer."""
                        # Use a sample for tree explainer to speed up
                        sample_size = min(100, len(_X_sample))
                        return shap.TreeExplainer(_model, _X_sample.iloc[:sample_size])
                    
                    if hasattr(model, 'estimators_'):  # Random Forest
                        explainer = get_shap_explainer(model, X_test)
                        
                        # Calculate SHAP values for this customer
                        customer_features = X_test.iloc[[test_idx]]
                        shap_values = explainer.shap_values(customer_features)
                        
                        # For binary classification, take the positive class
                        if isinstance(shap_values, list):
                            shap_values = shap_values[1]
                        
                        # Get top contributing features
                        feature_names = X_test.columns.tolist()
                        shap_importance = pd.DataFrame({
                            'Feature': feature_names,
                            'SHAP Value': shap_values[0],
                            'Impact': ['Increases Risk' if v > 0 else 'Decreases Risk' for v in shap_values[0]]
                        })
                        
                        # Sort by absolute value
                        shap_importance['Abs_SHAP'] = shap_importance['SHAP Value'].abs()
                        shap_importance = shap_importance.sort_values('Abs_SHAP', ascending=False).head(10)
                        
                        # Display top factors
                        st.write("**Top 10 Factors Contributing to This Prediction:**")
                        
                        for _, shap_row in shap_importance.iterrows():
                            feature = shap_row['Feature']
                            shap_val = shap_row['SHAP Value']
                            impact = shap_row['Impact']
                            
                            color = "🔴" if shap_val > 0 else "🟢"
                            st.write(f"{color} **{feature}**: {shap_val:+.3f} ({impact})")
                    else:
                        st.info("SHAP explanation is only available for tree-based models (Random Forest)")
                
                else:
                    st.info("Model or test data not available in session state")
                    
            except Exception as e:
                st.warning(f"⚠️ Could not generate SHAP explanation: {str(e)}")
                st.info("SHAP explanations require the `shap` library and a tree-based model")
    
    st.markdown("---")
    
    # Download Section
    st.subheader("📥 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare export dataframe
        export_df = filtered_df.copy()
        
        # Format for export
        if 'churn_probability' in export_df.columns:
            export_df['Churn_Probability_Pct'] = (export_df['churn_probability'] * 100).round(1)
        
        csv = export_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Filtered Customer List (CSV)",
            data=csv,
            file_name=f"at_risk_customers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download the complete filtered customer list with all columns"
        )
    
    with col2:
        # Create action plan export (simplified for campaigns)
        action_plan_df = export_df[[
            col for col in ['customerID', 'churn_probability', 'risk_tier', 
                          'Contact_Channel', 'Recommended_Action', 'Discount_Pct', 
                          'Retention_Value']
            if col in export_df.columns
        ]].copy()
        
        action_csv = action_plan_df.to_csv(index=False)
        
        st.download_button(
            label="📋 Download Action Plan (CSV)",
            data=action_csv,
            file_name=f"retention_action_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download simplified action plan for retention campaigns"
        )
    
    # Visualizations
    st.markdown("---")
    st.subheader("📊 Analysis Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Contract type distribution
        if 'Contract' in filtered_df.columns:
            fig_contract = px.histogram(
                filtered_df,
                x='Contract',
                color='risk_tier',
                title='Customers by Contract Type',
                color_discrete_map={'HIGH': '#ff4444', 'MEDIUM': '#ff9944', 'LOW': '#44ff44'},
                barmode='group'
            )
            fig_contract.update_layout(height=350)
            st.plotly_chart(fig_contract, use_container_width=True)
    
    with col2:
        # Churn probability vs Retention value scatter
        if 'Retention_Value' in filtered_df.columns:
            fig_scatter = px.scatter(
                filtered_df,
                x='churn_probability',
                y='Retention_Value',
                color='risk_tier',
                size='MonthlyCharges' if 'MonthlyCharges' in filtered_df.columns else None,
                title='Churn Risk vs Retention Value',
                color_discrete_map={'HIGH': '#ff4444', 'MEDIUM': '#ff9944', 'LOW': '#44ff44'},
                hover_data=['customerID'] if 'customerID' in filtered_df.columns else None
            )
            fig_scatter.update_layout(
                xaxis_title='Churn Probability',
                yaxis_title='Retention Value ($)',
                height=350
            )
            st.plotly_chart(fig_scatter, use_container_width=True)


def render_model_performance_page(model, X_test, y_test, churn_probabilities):
    """
    Render model performance metrics page.
    
    Displays:
    - ROC-AUC, precision, recall
    - Confusion matrix
    - Feature importance chart
    - Top-decile precision
    """
    st.header("📊 Model Performance")
    
    if y_test is None:
        st.warning("⚠️ Ground truth labels not available. Cannot compute performance metrics.")
        return
    
    # Convert target to binary
    y_binary = (y_test == 'Yes').astype(int)
    y_pred = (churn_probabilities >= 0.5).astype(int)
    
    # Calculate metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    roc_auc = roc_auc_score(y_binary, churn_probabilities)
    precision = precision_score(y_binary, y_pred)
    recall = recall_score(y_binary, y_pred)
    
    # Calculate top-decile precision
    threshold = np.percentile(churn_probabilities, 90)
    top_decile_mask = churn_probabilities >= threshold
    top_decile_precision = y_binary[top_decile_mask].mean() if top_decile_mask.sum() > 0 else 0
    
    with col1:
        st.metric(
            "ROC-AUC Score",
            f"{roc_auc:.3f}",
            help="Area Under the ROC Curve - measures model's ability to discriminate"
        )
    
    with col2:
        st.metric(
            "Precision",
            f"{precision:.3f}",
            help="Proportion of predicted churners who actually churned"
        )
    
    with col3:
        st.metric(
            "Recall",
            f"{recall:.3f}",
            help="Proportion of actual churners correctly identified"
        )
    
    with col4:
        st.metric(
            "Top-Decile Precision",
            f"{top_decile_precision:.3f}",
            help="Precision for the top 10% highest risk customers"
        )
    
    st.markdown("---")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        
        cm = confusion_matrix(y_binary, y_pred)
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No', 'Predicted Yes'],
            y=['Actual No', 'Actual Yes'],
            text=cm,
            texttemplate='%{text}',
            colorscale='Blues',
            showscale=False
        ))
        
        fig_cm.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Display metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        st.write(f"**True Negatives:** {tn}")
        st.write(f"**False Positives:** {fp}")
        st.write(f"**False Negatives:** {fn}")
        st.write(f"**True Positives:** {tp}")
    
    with col2:
        st.subheader("ROC Curve")
        
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_binary, churn_probabilities)
        
        fig_roc = go.Figure()
        
        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='steelblue', width=2)
        ))
        
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig_roc.update_layout(
            title="Receiver Operating Characteristic (ROC) Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_roc, use_container_width=True)
    
    # Model information
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Type:**", type(model).__name__)
        
        if hasattr(model, 'n_estimators'):
            st.write("**Number of Estimators:**", model.n_estimators)
        
        if hasattr(model, 'max_depth'):
            st.write("**Max Depth:**", model.max_depth)
        
        st.write("**Number of Features:**", X_test.shape[1])
    
    with col2:
        st.write("**Test Set Size:**", len(X_test))
        st.write("**Churn Rate (Actual):**", f"{y_binary.mean():.1%}")
        st.write("**Churn Rate (Predicted):**", f"{y_pred.mean():.1%}")


def render_individual_prediction_page(model, original_data):
    """
    Render individual customer prediction page.
    
    Features:
    - Interactive input form with all customer attributes
    - Real-time churn probability calculation
    - Risk tier classification with color-coded metrics
    - SHAP explanation for prediction
    - Top factors increasing/decreasing churn risk
    - Recommended retention action with priority and cost
    - What-if analysis for exploring interventions
    """
    st.header("🔍 Individual Customer Prediction")
    
    st.markdown("""
    Enter customer details below to predict churn risk in real-time. 
    This tool is designed for customer service representatives to quickly assess risk 
    and identify the best retention strategies.
    """)
    
    # Initialize session state for what-if analysis
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'original_inputs' not in st.session_state:
        st.session_state.original_inputs = {}
    
    # Create input form
    st.subheader("📝 Customer Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        senior_citizen = st.selectbox("👴 Senior Citizen", ["No", "Yes"], key="senior")
        partner = st.selectbox("👥 Has Partner", ["No", "Yes"], key="partner")
        dependents = st.selectbox("👶 Has Dependents", ["No", "Yes"], key="dependents")
        
        st.markdown("---")
        st.markdown("**Account Details**")
        tenure = st.slider("📅 Tenure (months)", min_value=0, max_value=72, value=12, key="tenure")
        monthly_charges = st.number_input(
            "💰 Monthly Charges ($)", 
            min_value=0.0, 
            max_value=200.0, 
            value=70.0, 
            step=5.0,
            key="monthly"
        )
    
    with col2:
        st.markdown("**Contract & Billing**")
        contract = st.selectbox(
            "📄 Contract Type", 
            ["Month-to-month", "One year", "Two year"],
            key="contract"
        )
        payment_method = st.selectbox(
            "💳 Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            key="payment"
        )
        paperless_billing = st.selectbox("📧 Paperless Billing", ["No", "Yes"], key="paperless")
        
        st.markdown("---")
        st.markdown("**Phone Services**")
        phone_service = st.selectbox("📞 Phone Service", ["No", "Yes"], key="phone")
        multiple_lines = st.selectbox("📱 Multiple Lines", ["No", "Yes", "No phone service"], key="lines")
    
    with col3:
        st.markdown("**Internet Services**")
        internet_service = st.selectbox(
            "🌐 Internet Service", 
            ["No", "DSL", "Fiber optic"],
            key="internet"
        )
        online_security = st.selectbox(
            "🔒 Online Security", 
            ["No", "Yes", "No internet service"],
            key="security"
        )
        tech_support = st.selectbox(
            "🛠️ Tech Support", 
            ["No", "Yes", "No internet service"],
            key="support"
        )
        online_backup = st.selectbox(
            "💾 Online Backup", 
            ["No", "Yes", "No internet service"],
            key="backup"
        )
        device_protection = st.selectbox(
            "🛡️ Device Protection", 
            ["No", "Yes", "No internet service"],
            key="protection"
        )
    
    st.markdown("**Streaming Services**")
    col1, col2, col3 = st.columns(3)
    with col1:
        streaming_tv = st.selectbox("📺 Streaming TV", ["No", "Yes", "No internet service"], key="tv")
    with col2:
        streaming_movies = st.selectbox("🎬 Streaming Movies", ["No", "Yes", "No internet service"], key="movies")
    with col3:
        gender = st.selectbox("⚧ Gender", ["Male", "Female"], key="gender")
    
    st.markdown("---")
    
    # Predict button
    predict_clicked = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)
    
    if predict_clicked:
        # Calculate total charges based on tenure and monthly charges
        total_charges = tenure * monthly_charges
        
        # Create customer dataframe
        customer_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],  # Keep as "Yes" or "No" string
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })
        
        # Try to apply feature engineering
        fe = load_feature_engineer()
        
        if fe is None:
            st.error("⚠️ Feature engineer not found. Cannot process custom input.")
            st.info("To enable individual predictions, ensure `models/feature_engineer.pkl` exists.")
            st.stop()
        
        try:
            # Transform customer data
            customer_features = fe.transform(customer_data)
            
            # Get prediction
            churn_prob = model.predict_proba(customer_features)[0, 1]
            risk_tier = classify_risk_tier(churn_prob)
            
            # Store in session state for what-if analysis
            st.session_state.prediction_made = True
            st.session_state.original_inputs = {
                'customer_data': customer_data,
                'customer_features': customer_features,
                'churn_prob': churn_prob,
                'risk_tier': risk_tier,
                'tenure': tenure,
                'monthly_charges': monthly_charges,
                'total_charges': total_charges,
                'contract': contract,
                'internet_service': internet_service,
                'online_security': online_security,
                'tech_support': tech_support,
                'payment_method': payment_method
            }
            
            # ============================================
            # 3. RESULTS DISPLAY
            # ============================================
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            # Large metrics with color-coding
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Color-code based on risk level
                if churn_prob >= 0.70:
                    prob_color = "🔴"
                    delta_color = "inverse"
                elif churn_prob >= 0.40:
                    prob_color = "🟠"
                    delta_color = "off"
                else:
                    prob_color = "🟢"
                    delta_color = "normal"
                
                st.metric(
                    "Churn Probability",
                    f"{prob_color} {churn_prob:.1%}",
                    delta=f"{(churn_prob - 0.5):.1%} from baseline",
                    delta_color=delta_color,
                    help="Predicted likelihood that customer will churn"
                )
            
            with col2:
                # Risk tier badge
                tier_emoji = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}
                st.metric(
                    "Risk Tier",
                    f"{tier_emoji[risk_tier]} {risk_tier}",
                    help="Risk classification based on probability threshold"
                )
            
            with col3:
                # Calculate retention value
                from src.retention_strategy import calculate_retention_value
                retention_value = calculate_retention_value({
                    'MonthlyCharges': monthly_charges,
                    'tenure': tenure,
                    'Contract': contract
                })
                st.metric(
                    "Est. Retention Value",
                    f"${retention_value:,.0f}",
                    help="Estimated value of retaining this customer over 12 months"
                )
            
            with col4:
                # Customer lifetime value
                clv = monthly_charges * 36  # 3-year estimate
                st.metric(
                    "3-Year CLV",
                    f"${clv:,.0f}",
                    help="Estimated customer lifetime value over 3 years"
                )
            
            st.markdown("---")
            
            # ============================================
            # Recommended Action
            # ============================================
            st.header("💡 Recommended Retention Action")
            
            customer_profile = {
                'MonthlyCharges': monthly_charges,
                'tenure': tenure,
                'Contract': contract,
                'TotalCharges': total_charges
            }
            
            rec = recommend_action(risk_tier, customer_profile)
            
            # Display recommendation in a nice card
            if risk_tier == "HIGH":
                st.error(f"**⚠️ HIGH PRIORITY ACTION REQUIRED**")
            elif risk_tier == "MEDIUM":
                st.warning(f"**⚡ PROACTIVE ENGAGEMENT RECOMMENDED**")
            else:
                st.success(f"**✅ STANDARD CUSTOMER CARE**")
            
            st.markdown(f"### {rec['action']}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Priority Level", rec['priority'], help="1=Urgent, 2=Important, 3=Standard")
            with col2:
                st.metric("Contact Channel", rec['channel'], help="Recommended communication channel")
            with col3:
                st.metric("Discount Offer", f"{rec['discount_percentage']}%", help="Suggested discount percentage")
            with col4:
                st.metric("Est. Cost", f"${rec['estimated_cost']:,.0f}", help="Estimated cost of intervention")
            
            # ROI Calculation
            roi = ((retention_value - rec['estimated_cost']) / rec['estimated_cost'] * 100) if rec['estimated_cost'] > 0 else float('inf')
            if roi != float('inf'):
                st.info(f"📈 **Expected ROI:** {roi:.0f}% (${retention_value:,.0f} value - ${rec['estimated_cost']:,.0f} cost)")
            
            st.markdown("---")
            
            # ============================================
            # 4. EXPLANATION SECTION
            # ============================================
            st.header("🔍 Prediction Explanation")
            
            try:
                import shap
                import matplotlib.pyplot as plt
                from io import BytesIO
                
                # Create SHAP explainer
                try:
                    explainer = shap.TreeExplainer(model)
                except:
                    # Use a sample of training data for KernelExplainer
                    X_test = st.session_state.get('X_test', None)
                    if X_test is not None:
                        sample_data = X_test.sample(min(100, len(X_test)), random_state=42)
                        explainer = shap.Explainer(model.predict, sample_data)
                    else:
                        raise Exception("Cannot create SHAP explainer without training data")
                
                # Calculate SHAP values for this prediction
                shap_values = explainer.shap_values(customer_features)
                
                # Handle different SHAP value formats
                if isinstance(shap_values, list):
                    shap_values_single = shap_values[1][0]  # Binary classification, positive class
                    expected_value = explainer.expected_value[1]
                else:
                    shap_values_single = shap_values[0]
                    expected_value = explainer.expected_value
                
                # Get feature names
                feature_names = customer_features.columns.tolist()
                
                # Create two columns for explanation
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("📊 SHAP Waterfall Plot")
                    st.caption("Shows how each feature pushes the prediction from the base value")
                    
                    # Create waterfall plot
                    fig, ax = plt.subplots(figsize=(10, 8))
                    shap.waterfall_plot(
                        shap.Explanation(
                            values=shap_values_single,
                            base_values=expected_value,
                            data=customer_features.iloc[0].values,
                            feature_names=feature_names
                        ),
                        show=False
                    )
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.subheader("📈 Feature Impact Summary")
                    
                    # Get top positive and negative SHAP values
                    shap_df = pd.DataFrame({
                        'feature': feature_names,
                        'shap_value': shap_values_single,
                        'feature_value': customer_features.iloc[0].values
                    })
                    shap_df = shap_df.sort_values('shap_value', ascending=False)
                    
                    # Top factors increasing risk
                    st.markdown("**🔺 Top 5 Factors Increasing Risk:**")
                    top_increasing = shap_df.head(5)
                    for idx, row in top_increasing.iterrows():
                        if row['shap_value'] > 0:
                            st.markdown(f"- **{row['feature']}**: +{row['shap_value']:.3f}")
                            st.caption(f"  Value: {row['feature_value']:.2f}")
                    
                    st.markdown("---")
                    
                    # Top factors decreasing risk
                    st.markdown("**🔻 Top 5 Factors Decreasing Risk:**")
                    top_decreasing = shap_df.tail(5).iloc[::-1]
                    for idx, row in top_decreasing.iterrows():
                        if row['shap_value'] < 0:
                            st.markdown(f"- **{row['feature']}**: {row['shap_value']:.3f}")
                            st.caption(f"  Value: {row['feature_value']:.2f}")
                
            except Exception as e:
                st.warning(f"⚠️ Could not generate SHAP explanation: {str(e)}")
                st.info("SHAP explanations require the model to be compatible with TreeExplainer or sufficient training data.")
            
            st.markdown("---")
            
            # ============================================
            # 5. WHAT-IF ANALYSIS
            # ============================================
            st.header("🔄 What-If Analysis")
            st.markdown("Explore how changes to customer attributes would impact churn risk")
            
            # Create tabs for different what-if scenarios
            scenario_tab1, scenario_tab2, scenario_tab3 = st.tabs([
                "📄 Contract Upgrade",
                "🌐 Service Changes",
                "💳 Payment Method"
            ])
            
            with scenario_tab1:
                st.subheader("Impact of Contract Changes")
                
                # Show current contract
                st.write(f"**Current Contract:** {contract}")
                st.write(f"**Current Risk:** {churn_prob:.1%}")
                
                # Simulate different contracts
                contracts_to_test = ["Month-to-month", "One year", "Two year"]
                contract_results = []
                
                for test_contract in contracts_to_test:
                    if test_contract != contract:
                        # Create modified customer data
                        modified_data = customer_data.copy()
                        modified_data['Contract'] = [test_contract]
                        
                        # Transform and predict
                        modified_features = fe.transform(modified_data)
                        new_prob = model.predict_proba(modified_features)[0, 1]
                        prob_change = new_prob - churn_prob
                        
                        contract_results.append({
                            'Contract': test_contract,
                            'New Probability': f"{new_prob:.1%}",
                            'Change': f"{prob_change:+.1%}",
                            'Risk Reduction': f"{abs(prob_change):.1%}" if prob_change < 0 else "N/A"
                        })
                
                if contract_results:
                    results_df = pd.DataFrame(contract_results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Best recommendation
                    if len(contract_results) > 0:
                        best_option = min(contract_results, key=lambda x: float(x['New Probability'].strip('%')))
                        st.success(f"💡 **Recommendation:** Switching to **{best_option['Contract']}** would reduce risk to {best_option['New Probability']}")
            
            with scenario_tab2:
                st.subheader("Impact of Additional Services")
                
                services_to_test = [
                    ('OnlineSecurity', 'Yes', 'Adding Online Security'),
                    ('TechSupport', 'Yes', 'Adding Tech Support'),
                    ('OnlineBackup', 'Yes', 'Adding Online Backup'),
                    ('StreamingTV', 'Yes', 'Adding Streaming TV'),
                    ('StreamingMovies', 'Yes', 'Adding Streaming Movies')
                ]
                
                service_results = []
                
                for service_col, service_val, description in services_to_test:
                    # Only test if customer doesn't already have the service
                    current_val = customer_data[service_col].iloc[0]
                    if current_val != service_val and current_val != "No internet service":
                        # Create modified customer data
                        modified_data = customer_data.copy()
                        modified_data[service_col] = [service_val]
                        
                        # Transform and predict
                        modified_features = fe.transform(modified_data)
                        new_prob = model.predict_proba(modified_features)[0, 1]
                        prob_change = new_prob - churn_prob
                        
                        service_results.append({
                            'Service Change': description,
                            'New Probability': f"{new_prob:.1%}",
                            'Change': f"{prob_change:+.1%}",
                            'Impact': '✅ Reduces Risk' if prob_change < -0.05 else '⚠️ Minor Impact' if abs(prob_change) < 0.05 else '❌ Increases Risk'
                        })
                
                if service_results:
                    results_df = pd.DataFrame(service_results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Best services to add
                    positive_impact = [r for r in service_results if 'Reduces' in r['Impact']]
                    if positive_impact:
                        st.success(f"💡 **Top Recommendation:** {positive_impact[0]['Service Change']} (Risk: {positive_impact[0]['New Probability']})")
                else:
                    st.info("Customer already has all testable services enabled.")
            
            with scenario_tab3:
                st.subheader("Impact of Payment Method Changes")
                
                st.write(f"**Current Payment Method:** {payment_method}")
                st.write(f"**Current Risk:** {churn_prob:.1%}")
                
                # Test different payment methods
                payment_methods = [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
                
                payment_results = []
                
                for test_payment in payment_methods:
                    if test_payment != payment_method:
                        # Create modified customer data
                        modified_data = customer_data.copy()
                        modified_data['PaymentMethod'] = [test_payment]
                        
                        # Transform and predict
                        modified_features = fe.transform(modified_data)
                        new_prob = model.predict_proba(modified_features)[0, 1]
                        prob_change = new_prob - churn_prob
                        
                        payment_results.append({
                            'Payment Method': test_payment,
                            'New Probability': f"{new_prob:.1%}",
                            'Change': f"{prob_change:+.1%}",
                            'Recommendation': '✅ Switch' if prob_change < -0.05 else '➡️ Consider' if prob_change < 0 else '❌ No Change'
                        })
                
                if payment_results:
                    results_df = pd.DataFrame(payment_results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Best payment method
                    best_payment = min(payment_results, key=lambda x: float(x['New Probability'].strip('%')))
                    if float(best_payment['New Probability'].strip('%')) < churn_prob * 100:
                        st.success(f"💡 **Recommendation:** Switching to **{best_payment['Payment Method']}** would reduce risk to {best_payment['New Probability']}")
            
            # Combined intervention simulation
            st.markdown("---")
            st.subheader("🎯 Combined Intervention Simulator")
            st.markdown("See the impact of multiple changes at once")
            
            col1, col2 = st.columns(2)
            
            with col1:
                combo_contract = st.selectbox("Contract", contracts_to_test, key="combo_contract")
                combo_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], key="combo_security")
                combo_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], key="combo_support")
            
            with col2:
                combo_payment = st.selectbox("Payment Method", payment_methods, key="combo_payment")
                combo_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], key="combo_backup")
            
            if st.button("🔮 Simulate Combined Changes"):
                # Create modified customer data
                combined_data = customer_data.copy()
                combined_data['Contract'] = [combo_contract]
                combined_data['OnlineSecurity'] = [combo_security]
                combined_data['TechSupport'] = [combo_support]
                combined_data['PaymentMethod'] = [combo_payment]
                combined_data['OnlineBackup'] = [combo_backup]
                
                # Transform and predict
                combined_features = fe.transform(combined_data)
                combined_prob = model.predict_proba(combined_features)[0, 1]
                combined_change = combined_prob - churn_prob
                combined_risk = classify_risk_tier(combined_prob)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Risk", f"{churn_prob:.1%}", help="Current churn probability")
                with col2:
                    st.metric("New Risk", f"{combined_prob:.1%}", delta=f"{combined_change:+.1%}", delta_color="inverse")
                with col3:
                    tier_emoji = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}
                    st.metric("New Risk Tier", f"{tier_emoji[combined_risk]} {combined_risk}")
                
                if combined_change < -0.10:
                    st.success(f"🎉 **Excellent!** This combination reduces churn risk by {abs(combined_change):.1%}")
                elif combined_change < 0:
                    st.info(f"✅ **Good.** This combination reduces churn risk by {abs(combined_change):.1%}")
                else:
                    st.warning(f"⚠️ This combination increases churn risk by {combined_change:.1%}")
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("The feature engineer may not be compatible with the input data format.")
            import traceback
            with st.expander("Show detailed error"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

