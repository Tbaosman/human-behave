"""
Generate a shared patient sample for fair comparison.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import PROCESSED_DATA_DIR, OUTPUTS_DIR


def generate_shared_sample(
    n_per_group: int = 50,
    random_seed: int = 42,
    output_name: str = "shared_sample"
) -> pd.DataFrame:
    """
    Generate balanced sample and save patient IDs.
    Both runners use these exact patients.
    """
    data_path = PROCESSED_DATA_DIR / "patient_profiles.parquet"
    df = pd.read_parquet(data_path)
    
    # Balanced sampling
    with_barrier = df[df["any_barrier"] == 1].sample(
        n=min(n_per_group, len(df[df["any_barrier"] == 1])),
        random_state=random_seed
    )
    
    without_barrier = df[df["any_barrier"] == 0].sample(
        n=min(n_per_group, len(df[df["any_barrier"] == 0])),
        random_state=random_seed
    )
    
    sample = pd.concat([with_barrier, without_barrier]).sample(
        frac=1, random_state=random_seed
    )
    
    # Save sample
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = OUTPUTS_DIR / f"{output_name}.parquet"
    sample.to_parquet(sample_path, index=False)
    
    print(f"Shared sample generated: {len(sample)} patients")
    print(f"  With barrier: {len(with_barrier)}")
    print(f"  Without barrier: {len(without_barrier)}")
    print(f"  Saved to: {sample_path}")
    
    return sample


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=50, help="Patients per group")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    generate_shared_sample(n_per_group=args.n, random_seed=args.seed)