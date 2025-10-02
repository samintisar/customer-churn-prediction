# Dashboard Structure & Navigation

## Visual Layout

```
╔════════════════════════════════════════════════════════════════════════════════╗
║  📊 Customer Churn Prediction & Retention Dashboard                           ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  ┌────────────────┬────────────────┬────────────────┬────────────────────┐   ║
║  │  📈 Overview   │ 🚨 At-Risk    │ 📊 Model       │ 🔍 Individual     │   ║
║  │                │   Customers    │   Performance  │   Prediction      │   ║
║  └────────────────┴────────────────┴────────────────┴────────────────────┘   ║
║                                                                                ║
║  ┌─ ACTIVE TAB: Overview ─────────────────────────────────────────────────┐  ║
║  │                                                                          │  ║
║  │  Summary Metrics                                                         │  ║
║  │  ┌──────────┬──────────┬──────────┬──────────┬──────────────────┐      │  ║
║  │  │  Total   │ 🔴 High  │ 🟠 Medium│ 🟢 Low   │ Avg Churn Prob   │      │  ║
║  │  │  1,409   │   143    │   312    │   954    │     26.8%        │      │  ║
║  │  └──────────┴──────────┴──────────┴──────────┴──────────────────┘      │  ║
║  │                                                                          │  ║
║  │  ┌────────────────────────────┬────────────────────────────────────┐    │  ║
║  │  │  Risk Distribution         │  Churn Probability Distribution    │    │  ║
║  │  │                            │                                    │    │  ║
║  │  │    [PIE CHART]             │      [HISTOGRAM]                   │    │  ║
║  │  │   High/Med/Low             │      │                             │    │  ║
║  │  │   with %                   │      │     │                       │    │  ║
║  │  │                            │      │     │ ││                    │    │  ║
║  │  │                            │    ──┼─────┼─┼┼──── 0.40 (Med)     │    │  ║
║  │  │                            │      │     │ ││││                  │    │  ║
║  │  │                            │    ──┼─────┼─┼┼┼┼──── 0.70 (High)  │    │  ║
║  │  └────────────────────────────┴────────────────────────────────────┘    │  ║
║  │                                                                          │  ║
║  │  Top Risk Factors (Feature Importance)                                  │  ║
║  │  ┌──────────────────────────────────────────────────────────────────┐   │  ║
║  │  │  tenure                     ████████████████████ 0.145            │   │  ║
║  │  │  MonthlyCharges             ███████████████ 0.118                │   │  ║
║  │  │  TotalCharges               ████████████ 0.095                   │   │  ║
║  │  │  Contract_Two year          ██████████ 0.082                     │   │  ║
║  │  │  ...                                                              │   │  ║
║  │  └──────────────────────────────────────────────────────────────────┘   │  ║
║  └──────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  SIDEBAR                                                                       ║
║  ┌──────────────┐                                                             ║
║  │ ⚙️ Settings  │                                                             ║
║  ├──────────────┤                                                             ║
║  │ Data Source  │                                                             ║
║  │ ○ Test Data  │                                                             ║
║  │ ○ Upload     │                                                             ║
║  ├──────────────┤                                                             ║
║  │ Risk Tiers   │                                                             ║
║  │ High: [─●──] │                                                             ║
║  │       0.70   │                                                             ║
║  │ Med:  [──●─] │                                                             ║
║  │       0.40   │                                                             ║
║  └──────────────┘                                                             ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

## Page-by-Page Breakdown

### 📈 Overview Page

**Purpose**: High-level dashboard view for executives and managers

**Components**:
1. **Metrics Row** (5 cards)
   - Total customers
   - High risk count + %
   - Medium risk count + %
   - Low risk count + %
   - Average churn probability

2. **Visualizations** (2 columns)
   - Left: Pie chart of risk distribution
   - Right: Histogram of churn probabilities with threshold lines

3. **Feature Importance** (full width)
   - Bar chart of top 10 most important features

**User Actions**:
- View at-a-glance metrics
- Understand risk distribution
- Identify key churn drivers

---

### 🚨 At-Risk Customers Page

**Purpose**: Operational page for retention teams

**Components**:
1. **Filters** (3 columns)
   - Risk tier multiselect
   - Contract type multiselect
   - Number to display

2. **Customer Table** (full width)
   - Churn probability
   - Risk tier (color-coded)
   - Customer ID
   - Tenure, Contract, Charges
   - Recommended action
   - Contact channel
   - Priority

3. **Export Section**
   - Download CSV button with timestamp

4. **Analysis Charts** (2 columns)
   - Left: Distribution by contract type
   - Right: Distribution by tenure

**User Actions**:
- Filter high-risk customers
- Review recommended actions
- Export campaign lists
- Analyze patterns

---

### 📊 Model Performance Page

**Purpose**: Technical page for data scientists and analysts

**Components**:
1. **Metrics Row** (4 cards)
   - ROC-AUC Score
   - Precision
   - Recall
   - Top-Decile Precision

2. **Visualizations** (2 columns)
   - Left: Confusion Matrix heatmap
   - Right: ROC Curve

3. **Model Information** (2 columns)
   - Left: Model type, hyperparameters
   - Right: Dataset statistics

**User Actions**:
- Evaluate model performance
- Understand prediction quality
- Assess targeting effectiveness
- Review model details

---

### 🔍 Individual Prediction Page

**Purpose**: Interactive page for customer service and sales teams

**Components**:
1. **Input Form** (collapsible sections)
   - Customer Information (3 columns)
     - Tenure, Charges, Total
   - Demographics (3 columns)
     - Gender, Senior, Partner, Dependents
   - Services (3 columns)
     - Phone, Internet, Contract, Billing
   - Additional Services (4 columns)
     - Multiple lines, Security, Backup, etc.

2. **Submit Button**
   - "🔮 Predict Churn Risk"

3. **Results Section** (appears after submit)
   - Metrics (3 columns)
     - Churn Probability
     - Risk Tier (with emoji)
     - Retention Value
   - Recommended Action (full width)
     - Action description
     - Priority, Channel, Discount

**User Actions**:
- Score individual customers
- Get real-time predictions
- Review retention recommendations
- Estimate customer value

---

## Sidebar Components

### Data Source Section
```
Data Source
┌─────────────────┐
│ ○ Test Data     │ ← Default
│ ○ Upload New    │
└─────────────────┘
    │
    └─► [File Uploader] (if Upload selected)
```

### Risk Tier Thresholds
```
Risk Tier Thresholds
┌─────────────────────┐
│ High Risk Threshold │
│ [────────●──]  0.70 │ ← Slider (0.50-1.00)
│                     │
│ Med Risk Threshold  │
│ [──●────────]  0.40 │ ← Slider (0.00-High)
└─────────────────────┘
```

## Data Flow

```
┌──────────────────┐
│  Load Model      │ ← models/churn_predictor.pkl
│  Load Test Data  │ ← data/processed/test.csv
│  Load Original   │ ← data/processed/cleaned_data.csv
└────────┬─────────┘
         │
         ↓
┌────────────────────┐
│ Generate           │
│ Predictions        │ ← model.predict_proba()
└────────┬───────────┘
         │
         ↓
┌────────────────────┐
│ Apply Thresholds   │ ← Custom or default (0.70, 0.40)
│ Classify Risk      │ ← HIGH / MEDIUM / LOW
└────────┬───────────┘
         │
         ↓
┌────────────────────┐
│ Store in           │
│ Session State      │ ← st.session_state['results_df']
└────────┬───────────┘
         │
         ↓
┌────────────────────┐
│ Render Active Tab  │
│ - Overview         │
│ - At-Risk          │
│ - Performance      │
│ - Individual       │
└────────────────────┘
```

## Interaction Flow

### Overview Page
```
User Action          →  System Response
─────────────────────────────────────────────
Load page            →  Display metrics & charts
Adjust threshold     →  Recalculate risk tiers
                     →  Update all visualizations
Change tab           →  Navigate to new page
```

### At-Risk Customers Page
```
User Action          →  System Response
─────────────────────────────────────────────
Select risk tier     →  Filter table
Select contract      →  Further filter table
Change top N         →  Limit displayed rows
Click download       →  Generate CSV export
```

### Model Performance Page
```
User Action          →  System Response
─────────────────────────────────────────────
Load page            →  Calculate metrics
                     →  Generate confusion matrix
                     →  Plot ROC curve
View model info      →  Display hyperparameters
```

### Individual Prediction Page
```
User Action          →  System Response
─────────────────────────────────────────────
Fill form fields     →  Enable submit button
Click submit         →  Apply feature engineering
                     →  Generate prediction
                     →  Display results
                     →  Show recommendation
```

## Caching Strategy

```
@st.cache_resource (Persistent across sessions)
├── load_trained_model()
│   └── Loaded once, never recomputed
└── load_feature_engineer()
    └── Loaded once, never recomputed

@st.cache_data (Refreshes when data changes)
├── load_test_data()
│   └── Reloaded if CSV file modified
└── load_original_data()
    └── Reloaded if CSV file modified
```

## Error Handling

```
┌─────────────────┐
│ File Not Found  │ → Display error message
│                 │ → Show instructions to create file
│                 │ → Stop execution (st.stop())
└─────────────────┘

┌─────────────────┐
│ Feature Eng     │ → Disable individual predictions
│ Missing         │ → Show warning message
│                 │ → Continue with other features
└─────────────────┘

┌─────────────────┐
│ Prediction      │ → Display error message
│ Error           │ → Log traceback
│                 │ → Provide troubleshooting hints
└─────────────────┘
```

## Performance Optimization

1. **Initial Load**: ~2-3 seconds
   - Model loading: ~1s
   - Data loading: ~1s
   - Prediction generation: ~0.5s

2. **Tab Switching**: Instant
   - Data cached in session state
   - No recomputation needed

3. **Filter Updates**: <100ms
   - In-memory DataFrame operations
   - Efficient pandas indexing

4. **Export Operations**: ~500ms
   - CSV generation from DataFrame
   - Download handled by browser

## Browser Compatibility

✅ Chrome 90+  
✅ Firefox 88+  
✅ Safari 14+  
✅ Edge 90+  

Recommended screen resolution: 1920x1080 or higher

---

**Summary**: The dashboard provides a complete, production-ready interface for customer churn prediction and retention planning with intuitive navigation, professional visualizations, and comprehensive functionality.

