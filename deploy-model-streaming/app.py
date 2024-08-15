import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from anomaly_detection import features
import plotly.express as px
from sklearn.decomposition import PCA

st.title("Why anomalous?")
st.subheader("Analyze customer spending data for anomalies and identify the reasons behind each anomaly")

with st.sidebar:
    st.header("Data requirements")
    st.caption("To run this app, you need to upload a dataframe in CSV format (only numerical features are taken into account)")
    with st.expander("Data format"):
        st.markdown(" - UTF-8")
        st.markdown(" - Comma-separated")
        st.markdown(" - No missing values")
        st.markdown(" - First row - header")
        st.markdown(" - Categorical features must be 'object' or 'category'")

    st.divider()
    st.caption("Developed by Emad Chelhi")

if "clicked" not in st.session_state:
    st.session_state.clicked = {1: False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Let's get started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    uploaded_file = st.file_uploader("Choose a file", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, low_memory=False)
            st.header("Uploaded Data Sample")
            st.write(df.head())

            # Apply preprocessing
            categorical_features = df.select_dtypes(include=['object', 'category']).columns
            df_num = df.drop(columns=categorical_features)
            df_log = features.logarithm(df_num, 100)  # Apply log transformation

            # Define gamma values
            gamma_values = np.logspace(-4, 2, 7)

            # Split data into training and test sets
            train_data, test_data = train_test_split(df_log, test_size=0.2, random_state=74)

            # Initialize variable to track the best gamma and its metric
            best_gamma = None
            best_metric = -np.inf

            # Iterate over each gamma value
            for gamma in gamma_values:
                # Initialize arrays to store mean and deviation of each datapoint across bootstraps
                mean_scores = np.zeros(len(test_data))
                deviation_scores = np.zeros(len(test_data))

                # Bootstrap and evaluate
                for _ in range(100):  # Perform 100 bootstraps
                    # Bootstrap the training data
                    bootstrapped_train_data = resample(train_data, replace=True, n_samples=np.ceil(0.7 * len(train_data)).astype(int))
                    # Compute anomaly scores on the test data using bootstrapped training data
                    anomaly_scores = features.compute_anomaly_scores(bootstrapped_train_data, test_data, gamma)

                    # Accumulate anomaly scores for each datapoint
                    mean_scores += anomaly_scores
                    deviation_scores += anomaly_scores ** 2

                # Calculate mean and deviation of each datapoint across bootstraps
                mean_scores /= 100
                deviation_scores /= 100
                deviation_scores -= mean_scores ** 2
                deviation_scores = np.sqrt(deviation_scores)

                # Normalize the mean and deviation scores
                mean_scores_norm = (mean_scores - np.mean(mean_scores)) / np.std(mean_scores)
                deviation_scores_norm = (deviation_scores - np.mean(deviation_scores)) / np.std(deviation_scores)

                # Calculate metric that benefits low deviation and high difference in mean between normal and anomalous data
                outliers = np.where(mean_scores_norm > np.percentile(mean_scores_norm, 97))[0]
                non_outliers = np.where(mean_scores_norm <= np.percentile(mean_scores_norm, 97))[0]
                metric = np.mean(np.abs(np.mean(outliers) - np.mean(non_outliers))) / np.mean((deviation_scores_norm) ** 2)

                # Check if this gamma value has the highest metric so far
                if metric > best_metric:
                    best_metric = metric
                    best_gamma = gamma

            # Compute final anomaly scores with the best gamma
            final_anomaly_scores = features.compute_anomaly_scores(df_log, df_log, best_gamma)
            final_anomaly_scores_df = pd.DataFrame(final_anomaly_scores, columns=["Anomaly Score"])

            st.header("Anomaly Scores")
            st.write(final_anomaly_scores_df.head())

            # Compute anomaly contributions and feature contributions
            anomaly_contributions = features.compute_anomaly_contributions(df_log.to_numpy(), best_gamma, final_anomaly_scores)
            feature_contributions = features.propagate_feature_contributions(anomaly_contributions, df_log.to_numpy())

            # Perform PCA on the data
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(df_log)
            pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])

            # Create a DataFrame with the PCA results and the anomaly scores
            pca_df['Anomaly Score'] = final_anomaly_scores
            pca_df['Anomaly Score'] = (pca_df['Anomaly Score'] - pca_df['Anomaly Score'].min()) / (pca_df['Anomaly Score'].max() - pca_df['Anomaly Score'].min())

            # Add the original index to the DataFrame
            pca_df['Index'] = df_log.index

            # Feature names
            feature_names = df_log.columns

            # Add the feature contributions to the DataFrame
            for i in range(df_log.shape[1]):
                pca_df[feature_names[i]] = feature_contributions[:, i]

            # Plot the scatter plot
            fig = px.scatter(pca_df, x='PC1', y='PC2', color='Anomaly Score', color_continuous_scale='Viridis', hover_data=['Index'] + list(pca_df.columns))
            fig.update_traces(marker=dict(size=5))

            st.header("Interactive Scatter Plot of Anomaly Scores")
            st.plotly_chart(fig)

            # Provide download option for results
            final_anomaly_scores_csv = final_anomaly_scores_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Anomaly Scores",
                               final_anomaly_scores_csv,
                               "anomaly_scores.csv",
                               "text/csv",
                               key="download-csv")
        except Exception as e:
            st.error(f"An error occurred: {e}")
