"""
Credit Card Fraud Detection - Streamlit in Snowflake Demo App

Multi-page app showcasing end-to-end Snowflake ML capabilities:
- Data Overview
- Feature Store Explorer
- Experiment Dashboard
- Model Registry
- Live Fraud Scoring
"""

import streamlit as st
import pandas as pd
import numpy as np

from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.functions import col, count, avg, sum as sum_, when, lit, stddev, max as max_, min as min_

session = get_active_session()

DB = "BANKING_ML_DEMO"
SCHEMA = "FRAUD_DETECTION"

st.set_page_config(page_title="Banking Fraud Detection ML Demo", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("Banking ML Demo")
st.sidebar.caption("Credit Card Fraud Detection")
st.sidebar.divider()

page = st.sidebar.radio("Navigate", [
    "Data Overview",
    "Feature Store",
    "Experiments",
    "Model Registry",
    "Live Fraud Scoring"
])

st.sidebar.divider()
st.sidebar.markdown(
    "**Snowflake ML Capabilities:**\n"
    "- Feature Store\n"
    "- Experiment Tracking\n"
    "- Model Registry\n"
    "- Model Serving (SPCS)\n"
    "- Streamlit in Snowflake"
)


# ============================================================
# PAGE 1: Data Overview
# ============================================================
if page == "Data Overview":
    st.title("Transaction Data Overview")
    st.caption("Exploring the synthetic credit card transaction dataset")

    try:
        raw = session.table(f"{DB}.{SCHEMA}.RAW_TRANSACTIONS")

        # Key metrics
        stats = raw.agg(
            count("*").alias("TOTAL"),
            avg("IS_FRAUD").alias("FRAUD_RATE"),
            count(when(col("IS_FRAUD") == 1, 1)).alias("FRAUD_COUNT"),
            avg("AMOUNT").alias("AVG_AMOUNT")
        ).to_pandas().iloc[0]

        n_customers = raw.select("CUSTOMER_ID").distinct().count()
        n_merchants = raw.select("MERCHANT_ID").distinct().count()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", f"{int(stats['TOTAL']):,}")
        col2.metric("Fraud Rate", f"{stats['FRAUD_RATE']:.2%}")
        col3.metric("Unique Customers", f"{n_customers:,}")
        col4.metric("Unique Merchants", f"{n_merchants:,}")

        st.divider()

        # Charts
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Transactions by Merchant Category")
            cat_df = raw.group_by("MERCHANT_CATEGORY").agg(
                count("*").alias("COUNT")
            ).to_pandas().sort_values("COUNT", ascending=False)
            st.bar_chart(cat_df.set_index("MERCHANT_CATEGORY")["COUNT"])

        with c2:
            st.subheader("Fraud vs Legitimate")
            fraud_df = raw.group_by("IS_FRAUD").agg(
                count("*").alias("COUNT"),
                avg("AMOUNT").alias("AVG_AMOUNT")
            ).to_pandas()
            fraud_df["LABEL"] = fraud_df["IS_FRAUD"].map({0: "Legitimate", 1: "Fraud"})
            st.bar_chart(fraud_df.set_index("LABEL")["COUNT"])

        st.divider()

        # Amount comparison
        st.subheader("Amount Distribution: Fraud vs Legitimate")
        amount_stats = raw.group_by("IS_FRAUD").agg(
            avg("AMOUNT").alias("AVG"),
            min_("AMOUNT").alias("MIN"),
            max_("AMOUNT").alias("MAX"),
            stddev("AMOUNT").alias("STDDEV")
        ).to_pandas()
        amount_stats["LABEL"] = amount_stats["IS_FRAUD"].map({0: "Legitimate", 1: "Fraud"})
        st.dataframe(amount_stats[["LABEL", "AVG", "MIN", "MAX", "STDDEV"]], use_container_width=True)

        # Sample data
        st.subheader("Sample Transactions")
        st.dataframe(raw.limit(100).to_pandas(), use_container_width=True)

    except Exception as e:
        st.error(f"Could not load data. Run notebook 00_setup first.\n\nError: {e}")


# ============================================================
# PAGE 2: Feature Store
# ============================================================
elif page == "Feature Store":
    st.title("Snowflake Feature Store")
    st.caption("Entities, Feature Views, and Dynamic Tables powering our ML pipeline")

    try:
        from snowflake.ml.feature_store import FeatureStore, CreationMode
        fs = FeatureStore(session=session, database=DB, name=SCHEMA,
                         default_warehouse="ML_DEMO_WH",
                         creation_mode=CreationMode.CREATE_IF_NOT_EXIST)

        # Entities
        st.subheader("Registered Entities")
        entities_df = fs.list_entities().to_pandas()
        if len(entities_df) > 0:
            st.dataframe(entities_df, use_container_width=True)
        else:
            st.info("No entities registered yet. Run notebook 01.")

        st.divider()

        # Feature Views
        st.subheader("Feature Views")
        fv_df = fs.list_feature_views().to_pandas()
        if len(fv_df) > 0:
            st.dataframe(fv_df, use_container_width=True)

            st.divider()

            # Sample data from each feature view
            st.subheader("Feature View Samples")
            for _, row in fv_df.iterrows():
                fv_name = row.get("NAME", row.get("name", ""))
                fv_version = row.get("VERSION", row.get("version", ""))
                if fv_name and fv_version:
                    with st.expander(f"{fv_name} / {fv_version}"):
                        try:
                            fv = fs.get_feature_view(fv_name, fv_version)
                            sample = fs.read_feature_view(fv).limit(10).to_pandas()
                            st.dataframe(sample, use_container_width=True)
                        except Exception as ex:
                            st.warning(f"Could not read: {ex}")
        else:
            st.info("No feature views registered yet. Run notebook 01.")

        st.divider()
        st.markdown("""
        **How it works:**
        - Feature Views are backed by **Dynamic Tables** that refresh incrementally
        - Entities define join keys for feature lookup at training and inference time
        - `generate_dataset()` creates versioned, reproducible training datasets
        - View in **Snowsight > AI & ML > Feature Store**
        """)

    except Exception as e:
        st.error(f"Could not connect to Feature Store.\n\nError: {e}")


# ============================================================
# PAGE 3: Experiments
# ============================================================
elif page == "Experiments":
    st.title("Experiment Tracking")
    st.caption("Compare model training runs side-by-side")

    try:
        # Display experiment comparison
        st.subheader("Model Comparison: FRAUD_DETECTION_EXPERIMENT")

        # Try to read from experiment tracking, fall back to hardcoded representative data
        comparison = pd.DataFrame({
            "Model": ["XGBoost Baseline", "XGBoost Tuned", "LightGBM"],
            "AUC": [0.95, 0.97, 0.96],
            "F1 Score": [0.72, 0.79, 0.76],
            "Precision": [0.80, 0.85, 0.82],
            "Recall": [0.65, 0.74, 0.71],
            "Estimators": [100, 300, 300],
            "Max Depth": [4, 6, 6],
            "Learning Rate": [0.1, 0.05, 0.05]
        })

        st.info("Metrics below are representative. Run notebook 02 for actual results on your data.")

        # Highlight best model
        best_idx = comparison["F1 Score"].idxmax()
        st.success(f"Best Model (by F1): **{comparison.loc[best_idx, 'Model']}** with F1={comparison.loc[best_idx, 'F1 Score']:.2f}")

        st.dataframe(
            comparison.style.highlight_max(subset=["AUC", "F1 Score", "Precision", "Recall"], color="#c6efce"),
            use_container_width=True
        )

        st.divider()

        # Metric charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("AUC Comparison")
            st.bar_chart(comparison.set_index("Model")["AUC"])
        with c2:
            st.subheader("F1 Score Comparison")
            st.bar_chart(comparison.set_index("Model")["F1 Score"])

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Precision")
            st.bar_chart(comparison.set_index("Model")["Precision"])
        with c4:
            st.subheader("Recall")
            st.bar_chart(comparison.set_index("Model")["Recall"])

        st.divider()
        st.markdown("""
        **Snowflake Experiment Tracking features:**
        - Autologging for XGBoost, LightGBM, and Keras
        - Manual metric and parameter logging
        - Compare up to 5 runs visually in Snowsight
        - View at **Snowsight > AI & ML > Experiments**
        """)

    except Exception as e:
        st.error(f"Error loading experiments: {e}")


# ============================================================
# PAGE 4: Model Registry
# ============================================================
elif page == "Model Registry":
    st.title("Snowflake Model Registry")
    st.caption("Versioned models with metadata, explainability, and lineage")

    try:
        from snowflake.ml.registry import Registry
        reg = Registry(session=session, database_name=DB, schema_name=SCHEMA)

        # List models
        st.subheader("Registered Models")
        models_df = reg.show_models()
        if models_df is not None and len(models_df) > 0:
            st.dataframe(models_df, use_container_width=True)

            # Model versions
            st.divider()
            st.subheader("Model Versions: FRAUD_DETECTION_MODEL")
            try:
                model = reg.get_model("FRAUD_DETECTION_MODEL")
                versions_df = model.show_versions()
                st.dataframe(versions_df, use_container_width=True)

                # Version details
                st.divider()
                st.subheader("Version Metrics")
                for ver_name in ["V1_BASELINE", "V2_TUNED"]:
                    try:
                        mv = model.version(ver_name)
                        metrics = mv.show_metrics()
                        with st.expander(f"Version: {ver_name} - {mv.comment}"):
                            if metrics:
                                metrics_df = pd.DataFrame([metrics])
                                st.dataframe(metrics_df, use_container_width=True)
                            else:
                                st.write("No metrics logged")
                    except Exception:
                        pass
            except Exception as ex:
                st.warning(f"Model not found: {ex}")
        else:
            st.info("No models registered yet. Run notebook 03.")

        st.divider()

        # Feature importance (representative)
        st.subheader("Feature Importance (SHAP)")
        importance = pd.DataFrame({
            "Feature": [
                "TXN_AMOUNT_RATIO", "TXN_IS_HIGH_AMOUNT", "AMOUNT",
                "MERCH_FRAUD_RATE", "TXN_IS_LATE_NIGHT", "CUST_STDDEV_AMOUNT",
                "CUST_AVG_TRANSACTION_AMT", "TXN_HOUR", "IS_ONLINE",
                "CUST_INCOME"
            ],
            "Importance": [0.35, 0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03]
        }).sort_values("Importance", ascending=True)

        st.bar_chart(importance.set_index("Feature")["Importance"])
        st.caption("Representative SHAP values. Run notebook 03 for actual values.")

        st.divider()
        st.markdown("""
        **Model Registry features:**
        - Version management with metrics and comments
        - Built-in SHAP explainability
        - Batch inference via `model.run()`
        - Lineage tracking to Feature Views
        - View at **Snowsight > AI & ML > Models**
        """)

    except Exception as e:
        st.error(f"Error loading registry: {e}")


# ============================================================
# PAGE 5: Live Fraud Scoring
# ============================================================
elif page == "Live Fraud Scoring":
    st.title("Live Fraud Scoring")
    st.caption("Score a transaction in real-time using the deployed model")

    st.markdown("Enter transaction details below to get a fraud risk assessment.")

    with st.form("fraud_scoring_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Transaction")
            amount = st.number_input("Amount ($)", min_value=1.0, max_value=50000.0, value=150.0, step=10.0)
            txn_hour = st.slider("Transaction Hour", 0, 23, 14)
            txn_dow = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            is_online = st.checkbox("Online Transaction", value=False)

        with c2:
            st.subheader("Customer Profile")
            cust_age = st.slider("Customer Age", 18, 80, 40)
            cust_income = st.number_input("Annual Income ($)", min_value=20000, max_value=500000, value=75000, step=5000)
            cust_avg_amt = st.number_input("Avg Transaction ($)", min_value=1.0, max_value=5000.0, value=120.0, step=10.0)
            cust_total_txns = st.number_input("Total Transactions", min_value=1, max_value=10000, value=200, step=10)

        with c3:
            st.subheader("Merchant")
            merchant_cat = st.selectbox("Category", [
                "grocery", "restaurant", "gas_station", "online_retail",
                "electronics", "travel", "entertainment", "healthcare"
            ])
            merch_avg_amt = st.number_input("Merchant Avg ($)", min_value=1.0, max_value=5000.0, value=150.0, step=10.0)
            merch_fraud_rate = st.number_input("Merchant Fraud Rate", min_value=0.0, max_value=1.0, value=0.01, step=0.005, format="%.3f")

        submitted = st.form_submit_button("Score Transaction", type="primary", use_container_width=True)

    if submitted:
        # Compute derived features
        dow_map = {"Monday": 1, "Tuesday": 2, "Wednesday": 3, "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 0}
        txn_day = dow_map[txn_dow]
        is_weekend = 1 if txn_day in [0, 6] else 0
        is_late_night = 1 if (txn_hour >= 22 or txn_hour <= 5) else 0
        amount_ratio = amount / max(cust_avg_amt, 1)
        is_high_amount = 1 if amount > 3 * cust_avg_amt else 0

        input_df = pd.DataFrame([{
            "AMOUNT": float(amount),
            "IS_ONLINE": int(is_online),
            "TXN_HOUR": txn_hour,
            "TXN_DAY_OF_WEEK": txn_day,
            "TXN_IS_WEEKEND": is_weekend,
            "TXN_IS_LATE_NIGHT": is_late_night,
            "TXN_AMOUNT_RATIO": amount_ratio,
            "TXN_IS_HIGH_AMOUNT": is_high_amount,
            "CUST_AVG_TRANSACTION_AMT": float(cust_avg_amt),
            "CUST_TOTAL_TRANSACTIONS": cust_total_txns,
            "CUST_STDDEV_AMOUNT": float(cust_avg_amt * 0.8),
            "CUST_MAX_AMOUNT": float(cust_avg_amt * 5),
            "CUST_UNIQUE_MERCHANTS": 30,
            "CUST_PCT_ONLINE": 0.3 if not is_online else 0.6,
            "CUST_AGE": cust_age,
            "CUST_INCOME": cust_income,
            "MERCH_AVG_TRANSACTION_AMT": float(merch_avg_amt),
            "MERCH_TOTAL_TRANSACTIONS": 1000,
            "MERCH_FRAUD_RATE": float(merch_fraud_rate)
        }])

        st.divider()

        try:
            from snowflake.ml.registry import Registry
            reg = Registry(session=session, database_name=DB, schema_name=SCHEMA)
            mv = reg.get_model("FRAUD_DETECTION_MODEL").default
            proba_result = mv.run(input_df, function_name="predict_proba")
            fraud_prob = float(proba_result.iloc[0, 1])
        except Exception as e:
            # Fallback: simple heuristic for demo if model not deployed
            risk_score = 0.0
            if is_late_night:
                risk_score += 0.25
            if amount_ratio > 3:
                risk_score += 0.30
            if is_online:
                risk_score += 0.05
            if merch_fraud_rate > 0.05:
                risk_score += 0.15
            fraud_prob = min(risk_score, 0.99)
            st.caption(f"Using heuristic scoring (model not deployed). Deploy via notebook 04 for ML scoring.")

        # Display results
        st.subheader("Risk Assessment")

        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud Probability", f"{fraud_prob:.1%}")

        if fraud_prob < 0.3:
            risk_level = "LOW RISK"
            r2.metric("Risk Level", risk_level)
            st.success(f"Transaction appears **legitimate**. Fraud probability: {fraud_prob:.1%}")
        elif fraud_prob < 0.7:
            risk_level = "MEDIUM RISK"
            r2.metric("Risk Level", risk_level)
            st.warning(f"Transaction is **suspicious**. Fraud probability: {fraud_prob:.1%}. Recommend manual review.")
        else:
            risk_level = "HIGH RISK"
            r2.metric("Risk Level", risk_level)
            st.error(f"Transaction is **likely fraudulent**. Fraud probability: {fraud_prob:.1%}. Recommend blocking.")

        r3.metric("Amount vs Avg", f"{amount_ratio:.1f}x")

        # Risk signals
        st.divider()
        st.subheader("Risk Signals")
        signals = []
        if is_late_night:
            signals.append("Late-night transaction (high-risk hours)")
        if amount_ratio > 3:
            signals.append(f"Amount is {amount_ratio:.1f}x customer average")
        if is_online:
            signals.append("Online transaction (card-not-present)")
        if merch_fraud_rate > 0.03:
            signals.append(f"Merchant has elevated fraud rate ({merch_fraud_rate:.1%})")
        if is_weekend:
            signals.append("Weekend transaction")

        if signals:
            for s in signals:
                st.markdown(f"- {s}")
        else:
            st.markdown("No significant risk signals detected.")
