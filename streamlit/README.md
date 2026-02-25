# Streamlit App Deployment

## Deploy to Snowflake

To deploy this Streamlit app in Snowflake:

1. **Create the Streamlit app via SQL:**

```sql
CREATE OR REPLACE STREAMLIT BANKING_ML_DEMO.FRAUD_DETECTION.ML_DEEPDIVE__FRAUD_DETECTION_APP
  ROOT_LOCATION = '@BANKING_ML_DEMO.FRAUD_DETECTION.NOTEBOOK_STAGE/ml-deepdive/streamlit'
  MAIN_FILE = 'app.py'
  QUERY_WAREHOUSE = 'ML_DEMO_WH';
```

2. **The app will automatically use packages from `packages.txt`:**
   - `snowflake-ml-python>=1.27.0`
   - `pandas>=2.0.0`
   - `numpy>=1.24.0`

3. **Access the app in Snowsight:**
   - Navigate to **Projects → Streamlit**
   - Click on `ML_DEEPDIVE__FRAUD_DETECTION_APP`

## Required Snowflake Objects

The app expects these objects to exist (created by running notebooks 00-04):

- `BANKING_ML_DEMO.FRAUD_DETECTION.RAW_TRANSACTIONS` table
- `BANKING_ML_DEMO.FRAUD_DETECTION.TXN_FEATURES` table
- Feature Store entities and feature views
- Model Registry with `FRAUD_DETECTION_MODEL`
- Experiments in `FRAUD_DETECTION_EXPERIMENT`

## App Pages

1. **Data Overview**: Transaction statistics and visualizations
2. **Feature Store**: Entities, Feature Views, and Dynamic Tables
3. **Experiments**: Model comparison and metrics
4. **Model Registry**: Registered models with versions and metrics
5. **Live Fraud Scoring**: Real-time transaction risk assessment
