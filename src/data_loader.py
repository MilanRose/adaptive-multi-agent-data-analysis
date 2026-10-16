import pandas as pd

def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.
    """
    df = pd.read_csv(file_path)
    return df


if __name__ == "__main__":
    # Quick test: load the dataset and show basic info
    df = load_dataset("data/titanic.csv")
    print("Loaded dataset with shape:", df.shape)
    print(df.head())