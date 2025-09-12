import pandas as pd
from sklearn.ensemble import IsolationForest
from preprocess_data import preprocess_wadi_data

def train_isolation_forest(file_path):
    """
    Trains an Isolation Forest model to detect anomalies in the dataset.

    Args:
        file_path (str): The path to the CSV dataset file.
    """
    # 1. Preprocess the data using the existing script
    preprocessed_data = preprocess_wadi_data(file_path)

    if preprocessed_data is None:
        print("Preprocessing failed. Exiting.")
        return

    # 2. Initialize the Isolation Forest model
    # The 'contamination' parameter is a key hyperparameter. It's the expected
    # proportion of outliers in the dataset. A value of 'auto' can be used,
    # but for time-series data, it's often better to specify a small value.
    model = IsolationForest(contamination=0.01, random_state=42)

    # 3. Fit the model to the preprocessed data
    print("Training the Isolation Forest model...")
    model.fit(preprocessed_data)
    print("Training complete.")

    # 4. Predict anomalies
    # The predict method returns 1 for inliers (normal) and -1 for outliers (anomalies)
    predictions = model.predict(preprocessed_data)

    # 5. Add anomaly scores and predictions to the original DataFrame for analysis
    preprocessed_data['anomaly_score'] = model.decision_function(preprocessed_data)
    preprocessed_data['anomaly_label'] = predictions
    
    # Count and display the number of anomalies detected
    num_anomalies = (predictions == -1).sum()
    total_records = len(predictions)
    
    print(f"\nTotal records: {total_records}")
    print(f"Anomalies detected: {num_anomalies}")
    print(f"Percentage of anomalies: {(num_anomalies / total_records) * 100:.2f}%")
    
    # Display a sample of the detected anomalies
    anomalies = preprocessed_data[preprocessed_data['anomaly_label'] == -1]
    if not anomalies.empty:
        print("\nSample of detected anomalies:")
        print(anomalies.head().to_markdown(index=True, numalign="left", stralign="left"))
    else:
        print("\nNo anomalies were detected based on the current contamination setting.")
        
if __name__ == "__main__":
    file_name = "WADI_dataset.csv"
    train_isolation_forest(file_name)
