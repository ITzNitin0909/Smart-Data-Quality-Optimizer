
import hashlib
import io

import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="Smart Data Quality Optimizer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Smart Data Quality Optimizer")
st.write("AI-powered data quality analysis for enterprise datasets.")

# -------------------- FILE UPLOAD --------------------

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Read the uploaded bytes once so we can identify dataset changes.
        uploaded_bytes = uploaded_file.getvalue()
        dataset_hash = hashlib.sha256(uploaded_bytes).hexdigest()
        df = pd.read_csv(io.BytesIO(uploaded_bytes))

        # Clear old AI results when a different dataset is uploaded.
        # Otherwise Streamlit session state could show results from a prior file.
        if st.session_state.get("dataset_hash") != dataset_hash:
            st.session_state.pop("ai_anomaly_results", None)
            st.session_state.pop("ai_anomaly_records", None)
            st.session_state["dataset_hash"] = dataset_hash

        if df.empty:
            st.warning("The uploaded CSV contains no data.")
            st.stop()

        # -------------------- DATASET OVERVIEW --------------------

        # -------------------- EXECUTIVE SUMMARY --------------------

        st.subheader("Executive Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:
            st.metric("Rows Loaded", f"{len(df):,}")

        with summary_col2:
            st.metric("Columns", f"{len(df.columns):,}")

        with summary_col3:
            st.metric(
                "Exact Duplicate Rows",
                f"{int(df.duplicated().sum()):,}"
            )

        st.caption(
            "Summary is based on the uploaded file. "
            "AI anomalies are reviewed separately and are not automatically "
            "counted as data errors."
        )

        # -------------------- DATASET OVERVIEW --------------------

        st.subheader("Dataset Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Rows", f"{len(df):,}")

        with col2:
            st.metric("Total Columns", len(df.columns))

        with col3:
            total_missing = int(df.isnull().sum().sum())
            st.metric("Missing Values", f"{total_missing:,}")

        # -------------------- DATASET PREVIEW --------------------

        st.subheader("Dataset Preview")
        st.dataframe(df.head(100), width="stretch")

        # -------------------- MISSING VALUES --------------------

        st.subheader("Missing Values by Column")

        missing_report = pd.DataFrame({
            "Column": df.columns,
            "Missing Count": df.isnull().sum().values,
            "Missing %": (
                df.isnull().sum().values / len(df) * 100
            ).round(2)
        })

        missing_report = missing_report[
            missing_report["Missing Count"] > 0
        ].sort_values("Missing Count", ascending=False)

        if missing_report.empty:
            st.success("No missing values detected.")
        else:
            st.dataframe(missing_report, width="stretch")
            st.bar_chart(
                missing_report.set_index("Column")["Missing Count"]
            )

        # -------------------- MISSING DATA RECOMMENDATIONS --------------------

        st.subheader("Missing Data Recommendations")

        recommendations = []

        for _, row in missing_report.iterrows():
            column = row["Column"]
            count = int(row["Missing Count"])

            if column == "Customer ID":
                action = (
                    "Review whether customer ID is optional. "
                    "Do not delete transactions automatically."
                )
            elif column == "Description":
                action = (
                    "Check the product catalogue using StockCode "
                    "before filling the description."
                )
            else:
                action = "Investigate the missing-value handling policy."

            recommendations.append({
                "Column": column,
                "Missing Rows": count,
                "Recommended Action": action,
                "Auto Fix": "No"
            })

        if recommendations:
            recommendation_df = pd.DataFrame(recommendations)
            st.dataframe(recommendation_df, width="stretch")

            st.download_button(
                "Download Missing Data Recommendations",
                data=recommendation_df.to_csv(index=False).encode("utf-8"),
                file_name="missing_data_recommendations.csv",
                mime="text/csv"
            )
        else:
            st.success("No missing-data recommendations needed.")

        # -------------------- COLUMN DATA TYPES --------------------

        st.subheader("Column Data Types")

        dtype_report = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values
        })

        st.dataframe(dtype_report, width="stretch")

        # -------------------- RULE-BASED CHECKS --------------------

        st.subheader("Basic Data Quality Checks")

        if {"Quantity", "Price"}.issubset(df.columns):
            quantity = pd.to_numeric(df["Quantity"], errors="coerce")
            price = pd.to_numeric(df["Price"], errors="coerce")

            rule_report = pd.DataFrame({
                "Issue": [
                    "Exact Duplicate Rows",
                    "Negative Quantity",
                    "Zero Price",
                    "Negative Price"
                ],
                "Affected Records": [
                    int(df.duplicated().sum()),
                    int((quantity < 0).sum()),
                    int((price == 0).sum()),
                    int((price < 0).sum())
                ]
            })

            rule_report["Affected %"] = (
                rule_report["Affected Records"] / len(df) * 100
            ).round(2)

            st.dataframe(rule_report, width="stretch")

            # Provisional score: rows with any selected rule issue
            # are counted once. This is not a validated business-quality score.
            score_issue_mask = (
                df.duplicated()
                | (quantity < 0).fillna(False)
                | (price <= 0).fillna(False)
            )
            issue_rows = int(score_issue_mask.sum())
            quality_score = max(
                0.0,
                (1 - issue_rows / len(df)) * 100
            )

            st.subheader("Provisional Data Quality Score")
            score_col1, score_col2 = st.columns(2)

            with score_col1:
                st.metric("Provisional Score", f"{quality_score:.2f}%")

            with score_col2:
                st.metric("Rows Flagged by Selected Rules", f"{issue_rows:,}")

            st.caption(
                "Provisional only: flags duplicate rows, negative quantity, "
                "and zero/negative price. Some may be valid business "
                "transactions; confirm business rules before treating this "
                "as a final data quality score."
            )

            st.download_button(
                "Download Rule-Based Quality Report",
                data=rule_report.to_csv(index=False).encode("utf-8"),
                file_name="quality_report.csv",
                mime="text/csv"
            )
        else:
            st.info("Rule checks require Quantity and Price columns.")

        # -------------------- AI ANOMALY DETECTION --------------------

        st.subheader("AI Anomaly Detection — Isolation Forest")

        if {"Quantity", "Price"}.issubset(df.columns):
            ai_data = pd.DataFrame({
                "Quantity": pd.to_numeric(
                    df["Quantity"], errors="coerce"
                ),
                "Price": pd.to_numeric(
                    df["Price"], errors="coerce"
                )
            }, index=df.index)

            ai_data = ai_data.replace(
                [float("inf"), float("-inf")], float("nan")
            )

            valid_mask = ai_data.notna().all(axis=1)
            valid_features = ai_data.loc[valid_mask]

            if len(valid_features) < 10:
                st.warning(
                    "At least 10 valid Quantity/Price records "
                    "are needed for AI detection."
                )
            else:
                contamination = st.slider(
                    "Expected anomaly proportion (%)",
                    min_value=1,
                    max_value=10,
                    value=5,
                    help=(
                        "Model setting, not the actual percentage "
                        "of incorrect records."
                    )
                )

                if st.button("Run AI Anomaly Detection"):
                    with st.spinner("Training Isolation Forest..."):
                        model = IsolationForest(
                            n_estimators=100,
                            contamination=contamination / 100,
                            random_state=42,
                            n_jobs=-1
                        )

                        predictions = model.fit_predict(valid_features)

                        anomaly_df = df.loc[
                            valid_features.index
                        ].copy()

                        anomaly_df["AI_Anomaly"] = predictions
                        anomaly_df["AI_Status"] = anomaly_df[
                            "AI_Anomaly"
                        ].map({
                            -1: "Anomaly — Review",
                            1: "Normal"
                        })

                        anomalies = anomaly_df[
                            anomaly_df["AI_Anomaly"] == -1
                        ].copy()

                    st.session_state["ai_anomaly_results"] = anomaly_df
                    st.session_state["ai_anomaly_records"] = anomalies
                    st.session_state["ai_results_dataset_hash"] = dataset_hash
                    st.session_state["ai_results_contamination"] = contamination

                # -------------------- AI RESULTS --------------------

                results_match_current_settings = (
                    "ai_anomaly_results" in st.session_state
                    and st.session_state.get("ai_results_dataset_hash") == dataset_hash
                    and st.session_state.get("ai_results_contamination") == contamination
                )

                if results_match_current_settings:
                    anomaly_df = st.session_state["ai_anomaly_results"]
                    anomalies = st.session_state["ai_anomaly_records"]

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Records Analysed",
                            f"{len(anomaly_df):,}"
                        )

                    with col2:
                        st.metric(
                            "AI Anomalies",
                            f"{len(anomalies):,}"
                        )

                    with col3:
                        anomaly_pct = (
                            len(anomalies) / len(anomaly_df) * 100
                        )
                        st.metric(
                            "Anomaly Percentage",
                            f"{anomaly_pct:.2f}%"
                        )

                    st.warning(
                        "AI anomalies are unusual records, not confirmed "
                        "errors. Review them before making changes."
                    )

                    st.subheader("AI-Detected Anomaly Records")
                    st.dataframe(
                        anomalies.head(500),
                        width="stretch"
                    )

                    st.download_button(
                        "Download All AI Anomalies",
                        data=anomalies.to_csv(index=False).encode("utf-8"),
                        file_name="ai_anomalies.csv",
                        mime="text/csv"
                    )

                    # -------------------- AI VS RULE COMPARISON --------------------

                    st.subheader(
                        "AI vs Rule-Based Detection Comparison"
                    )

                    # Rule: negative quantity OR zero/negative price
                    rule_issue_mask = (
                        (quantity < 0) | (price <= 0)
                    ).fillna(False)

                    # Mark AI anomalies using original dataframe indices
                    ai_anomaly_mask = pd.Series(
                        False, index=df.index
                    )

                    ai_anomaly_mask.loc[anomaly_df.index] = (
                        anomaly_df["AI_Anomaly"].eq(-1).to_numpy()
                    )

                    both_mask = ai_anomaly_mask & rule_issue_mask
                    ai_only_mask = ai_anomaly_mask & ~rule_issue_mask
                    rule_only_mask = rule_issue_mask & ~ai_anomaly_mask

                    comparison_df = pd.DataFrame({
                        "Detection Category": [
                            "AI Anomalies",
                            "Rule-Based Issues",
                            "Detected by Both",
                            "AI Only",
                            "Rule-Based Only"
                        ],
                        "Records": [
                            int(ai_anomaly_mask.sum()),
                            int(rule_issue_mask.sum()),
                            int(both_mask.sum()),
                            int(ai_only_mask.sum()),
                            int(rule_only_mask.sum())
                        ]
                    })

                    st.dataframe(
                        comparison_df,
                        width="stretch",
                        hide_index=True
                    )

                    st.bar_chart(
                        comparison_df.set_index("Detection Category")
                    )

                    st.caption(
                        "Rule-based issues mean Quantity < 0 or Price <= 0. "
                        "These may be valid business transactions. "
                        "AI-only means flagged by AI but not by these rules. "
                        "Counts are record-level and categories may overlap."
                    )

                    # Download comparison report
                    st.download_button(
                        "Download AI vs Rule Comparison",
                        data=comparison_df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="ai_vs_rule_comparison.csv",
                        mime="text/csv"
                    )

                    # -------------------- ANOMALY PATTERN ANALYSIS --------------------

                    st.subheader("AI Anomaly Pattern Analysis")

                    if not anomalies.empty:
                        anomaly_quantity = pd.to_numeric(
                            anomalies["Quantity"],
                            errors="coerce"
                        )
                        anomaly_price = pd.to_numeric(
                            anomalies["Price"],
                            errors="coerce"
                        )

                        pattern_report = pd.DataFrame({
                            "Metric": [
                                "Negative Quantity",
                                "Zero Price",
                                "Negative Price",
                                "Positive Quantity & Price"
                            ],
                            "Records": [
                                int((anomaly_quantity < 0).sum()),
                                int((anomaly_price == 0).sum()),
                                int((anomaly_price < 0).sum()),
                                int((
                                    (anomaly_quantity > 0)
                                    & (anomaly_price > 0)
                                ).sum())
                            ]
                        })

                        st.dataframe(
                            pattern_report,
                            width="stretch",
                            hide_index=True
                        )

                        st.bar_chart(
                            pattern_report.set_index("Metric")
                        )

                        st.download_button(
                            "Download Anomaly Pattern Report",
                            data=pattern_report.to_csv(
                                index=False
                            ).encode("utf-8"),
                            file_name="anomaly_pattern_report.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info("No AI anomalies found.")
                elif "ai_anomaly_results" in st.session_state:
                    st.info(
                        "The dataset or contamination setting has changed. "
                        "Click **Run AI Anomaly Detection** to refresh the results."
                    )

        else:
            st.info(
                "AI detection requires Quantity and Price columns."
            )

        st.success("Dataset analysis completed.")

    except Exception as error:
        st.error(f"Could not process the CSV: {error}")

else:
    st.info("Upload a CSV file to begin data quality analysis.")
