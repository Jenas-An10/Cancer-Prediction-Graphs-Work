import streamlit as st
import pandas as pd
from ml_tools import run_ML_pipeline

def main():
    st.title("Machine Learning Pipeline")

    # File uploader
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "tsv"])

    # Select model
    ml_model = st.selectbox(
        "Choose a machine learning model",
        ["NB", "kNN", "LR", "RF", "SVM"]
    )

    # Select report type
    report_type = st.selectbox(
        "Choose a report type",
        ["confusion_matrix", "roc_auc_curve", "feature_importance", "prediction_result", "pca_plot"]
    )

    if uploaded_file and ml_model and report_type:
        # Convert file to a pandas DataFrame
        file_path = uploaded_file.name  # Use the uploaded file's name

        # Save the uploaded file temporarily
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run the ML pipeline
        if report_type in ["confusion_matrix", "roc_auc_curve", "feature_importance"]:
            result = run_ML_pipeline(report_type, file_path, ml_model)
            if report_type == "feature_importance":
                st.bar_chart(result)  # Display feature importance
            else:
                st.pyplot(result)  # Display plots
        elif report_type == "prediction_result":
            result = run_ML_pipeline(report_type, file_path, ml_model)
            st.write(result)  # Display prediction results
        elif report_type == "pca_plot":
            result = run_ML_pipeline(report_type, file_path, ml_model)
            st.pyplot(result)  # Display PCA plot

if __name__ == "__main__":
    main()
