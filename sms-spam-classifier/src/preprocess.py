import pandas as pd

def load_data(path):
    # Support both the original tab-separated dataset and the provided CSV
    if str(path).endswith(".csv"):
        # Use latin-1 to handle legacy encodings in SMS dataset
        df = pd.read_csv(path, encoding='latin-1')
        # Normalize column names (some CSVs use 'text' vs 'message')
        if 'text' in df.columns:
            df = df.rename(columns={'text': 'message'})
        elif 'message' not in df.columns and df.shape[1] >= 2:
            # Fallback: if file has two columns without proper headers
            df.columns = ['label', 'message']
    else:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding='latin-1'
        )

    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df
