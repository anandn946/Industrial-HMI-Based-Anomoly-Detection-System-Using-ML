import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_wadi_data(file_path):
    """
    Preprocesses the WaDi dataset for use with an Isolation Forest model.

    Args:
        file_path (str): The path to the CSV dataset file.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame ready for modeling.
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

        # Drop the first column if it's an unnamed index
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Timestamp'}, inplace=True)
        else:
            print("No 'Unnamed: 0' column found. Assuming the first column is the timestamp.")

        # Convert the timestamp column to datetime objects and set as index
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)

        # Identify categorical columns to handle them separately
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

        # Drop categorical columns for now, as Isolation Forest works best with numerical data.
        # This simplifies the model but you may want to encode them later if they are relevant.
        numerical_df = df.drop(columns=categorical_cols)

        # Replace 'Bad Input' values with NaN
        numerical_df.replace('Bad Input', float('nan'), inplace=True)

        # Convert all remaining columns to a numerical data type
        numerical_df = numerical_df.apply(pd.to_numeric, errors='coerce')

        # Fill any remaining NaN values with the mean of their column
        numerical_df.fillna(numerical_df.mean(), inplace=True)

        # Apply MinMaxScaler to normalize the data
        scaler = MinMaxScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(numerical_df),
                                 columns=numerical_df.columns,
                                 index=numerical_df.index)

        return scaled_df

    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        return None

if __name__ == "__main__":
    file_name = "WADI_dataset.csv"
    preprocessed_data = preprocess_wadi_data(file_name)

    if preprocessed_data is not None:
        print("Preprocessed DataFrame created successfully.")
        print("\nFirst 5 rows of the preprocessed data:")
        print(preprocessed_data.head())
        print("\nInformation about the preprocessed DataFrame:")
        preprocessed_data.info()
