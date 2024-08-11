import requests
import pandas as pd

def download_data(url: str, save_path: str):
    """Download data from a URL and save it to a file.

    Args:
        url (str): URL to download data from.
        save_path (str): Path to save the downloaded file.

    Raises:
        RequestException: If the request to download the data fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f"Data successfully downloaded from {url} to {save_path}.")
    except requests.RequestException as e:
        print(f"Failed to download data: {e}")

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the file is not found.
        pd.errors.EmptyDataError: If the file is empty.
        pd.errors.ParserError: If there is a parsing error.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data successfully loaded from {file_path}.")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame
    except pd.errors.EmptyDataError:
        print(f"File is empty: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame
    except pd.errors.ParserError:
        print(f"Error parsing file: {file_path}")
        return pd.DataFrame()  # Return an empty DataFrame
