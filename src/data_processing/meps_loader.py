"""
Load raw MEPS data.
"""
from pathlib import Path
import sys
import pandas as pd
import pyreadstat


sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import MEPS_FILE


def load_meps_data(file_path: Path = None) -> pd.DataFrame:
    """
    Load MEPS .dta file.
    
    Args:
        file_path: Path to .dta file. Uses default if None.
        
    Returns:
        DataFrame with MEPS data
    """
    if file_path is None:
        file_path = MEPS_FILE
    
    if not file_path.exists():
        raise FileNotFoundError(f"MEPS file not found: {file_path}")
    
    print(f"Loading MEPS data from: {file_path}")
    df, meta = pyreadstat.read_dta(file_path)
    print(f"Loaded {len(df):,} records with {len(df.columns)} variables")
    
    return df


if __name__ == "__main__":
    df = load_meps_data()
    print(f"\nShape: {df.shape}")
    print(f"\nFirst few columns: {df.columns[:10].tolist()}")