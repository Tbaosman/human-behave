"""
Process MEPS data into patient profiles.
"""


from pathlib import Path
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import PROCESSED_DATA_DIR, GROUND_TRUTH_VARS
from src.data_processing.meps_loader import load_meps_data


# Variables to extract from MEPS
VARIABLES_TO_EXTRACT = {
    # Identifiers
    "DUPERSID": "person_id",
    
    # Demographics
    "AGE23X": "age",
    "SEX": "sex",
    "MARRY23X": "marital_status",
    "RACETHX": "race_ethnicity",
    "REGION23": "region",
    "FAMSZEYR": "family_size",
    
    # Economic
    "POVCAT23": "poverty_category",
    "POVLEV23": "poverty_level_pct",
    "FAMINC23": "family_income",
    "TTLP23X": "personal_income",
    
    # Insurance
    "INSCOV23": "insurance_coverage",
    "PRVEV23": "has_private",
    "MCREV23": "has_medicare",
    "MCDEV23": "has_medicaid",
    "UNINS23": "uninsured_full_year",
    
    # Health status
    "RTHLTH53": "health_status",
    "MNHLTH53": "mental_health_status",
    
    # Chronic conditions
    "HIBPDX": "has_hypertension",
    "DIABDX_M18": "has_diabetes",
    "ASTHDX": "has_asthma",
    "ARTHDX": "has_arthritis",
    "CHDDX": "has_heart_disease",
    "STRKDX": "has_stroke",
    "CANCERDX": "has_cancer",
    
    # Healthcare access
    "HAVEUS42": "has_usual_care_source",
    
    # Utilization
    "OBTOTV23": "office_visits",
    "OPTOTV23": "outpatient_visits",
    "ERTOT23": "er_visits",
    "IPDIS23": "inpatient_stays",
    "RXTOT23": "prescriptions",
    
    # Expenditures
    "TOTEXP23": "total_expenditure",
    "TOTSLF23": "out_of_pocket",
    
    # Employment
    "EMPST53": "employment_status",
    
    # Ground truth
    "DLAYCA42": "delayed_care",
    "AFRDCA42": "forgone_care",
}


# MEPS Yes/No coding
MEPS_YES = 1
MEPS_NO = 2


def process_meps_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw MEPS data into clean patient profiles.
    """
    # Extract relevant variables
    available_vars = [v for v in VARIABLES_TO_EXTRACT.keys() if v in df.columns]
    missing_vars = [v for v in VARIABLES_TO_EXTRACT.keys() if v not in df.columns]
    
    if missing_vars:
        print(f"Warning: Missing variables: {missing_vars}")
    
    processed = df[available_vars].copy()
    
    # Rename columns
    rename_map = {k: v for k, v in VARIABLES_TO_EXTRACT.items() if k in available_vars}
    processed = processed.rename(columns=rename_map)
    
    # Filter to adults (18+)
    if "age" in processed.columns:
        processed = processed[processed["age"] >= 18].copy()
        print(f"Filtered to adults (18+): {len(processed):,} records")
    
    # Convert MEPS coding to binary
    processed = _convert_meps_coding(processed)
    
    # Calculate derived variables
    processed = _add_derived_variables(processed)
    
    return processed


def _convert_meps_coding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert MEPS Yes/No coding to binary (0/1).
    
    MEPS coding:
        1 = Yes → 1
        2 = No  → 0
        Negative values = Missing → NaN
    """
    
    # Ground truth variables
    binary_vars = ["delayed_care", "forgone_care"]
    
    for var in binary_vars:
        if var in df.columns:
            df[var] = df[var].apply(_meps_to_binary)
            
            valid_count = df[var].notna().sum()
            yes_count = (df[var] == 1).sum()
            print(f"  {var}: {yes_count:,} yes / {valid_count:,} valid ({yes_count/valid_count*100:.1f}%)")
    
    # Other Yes/No variables (e.g., has_usual_care_source, chronic conditions)
    other_binary = [
        "has_usual_care_source",
        "has_hypertension", "has_diabetes", "has_asthma",
        "has_arthritis", "has_heart_disease", "has_stroke", "has_cancer",
        "has_private", "has_medicare", "has_medicaid", "uninsured_full_year"
    ]
    
    for var in other_binary:
        if var in df.columns:
            df[var] = df[var].apply(_meps_to_binary)
    
    return df


def _meps_to_binary(value) -> int:
    """
    Convert single MEPS value to binary.
    
    1 → 1 (Yes)
    2 → 0 (No)
    Negative/Other → None (Missing)
    """
    if pd.isna(value):
        return None
    if value == MEPS_YES:
        return 1
    if value == MEPS_NO:
        return 0
    return None  # Negative values or other


def _add_derived_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived variables."""
    
    # Chronic condition count
    chronic_cols = [
        "has_hypertension", "has_diabetes", "has_asthma",
        "has_arthritis", "has_heart_disease", "has_stroke", "has_cancer"
    ]
    available_chronic = [c for c in chronic_cols if c in df.columns]
    
    if available_chronic:
        df["chronic_count"] = df[available_chronic].apply(
            lambda row: sum(1 for v in row if v == 1), axis=1
        )
    
    # OOP burden ratio
    if "out_of_pocket" in df.columns and "family_income" in df.columns:
        df["oop_burden_ratio"] = df.apply(
            lambda row: row["out_of_pocket"] / row["family_income"] 
            if pd.notna(row["family_income"]) and row["family_income"] > 0 else None,
            axis=1
        )
    
    # Any barrier (combined ground truth)
    if "delayed_care" in df.columns and "forgone_care" in df.columns:
        df["any_barrier"] = df.apply(
            lambda row: (
                1 if (row["delayed_care"] == 1 or row["forgone_care"] == 1)
                else (0 if (pd.notna(row["delayed_care"]) or pd.notna(row["forgone_care"]))
                else None)
            ),
            axis=1
        )
    
    return df


def save_processed_data(df: pd.DataFrame, output_path: Path = None):
    """Save processed data to parquet."""
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "patient_profiles.parquet"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved processed data to: {output_path}")
    
    # Also save as CSV for easy inspection
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV copy to: {csv_path}")


def main():
    """Main processing pipeline."""
    print("=" * 70)
    print("MEPS DATA PROCESSING")
    print("=" * 70)
    
    # Load
    df = load_meps_data()
    
    # Process
    print("\nConverting MEPS coding...")
    processed = process_meps_data(df)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("PROCESSED DATA SUMMARY")
    print("=" * 70)
    print(f"Total records: {len(processed):,}")
    print(f"Variables: {len(processed.columns)}")
    
    print(f"\n--- Ground Truth Rates (Adults 18+) ---")
    
    if "delayed_care" in processed.columns:
        valid = processed["delayed_care"].dropna()
        delay_rate = valid.mean() * 100
        print(f"Delay rate: {delay_rate:.1f}% ({int(valid.sum()):,} / {len(valid):,})")
    
    if "forgone_care" in processed.columns:
        valid = processed["forgone_care"].dropna()
        forgo_rate = valid.mean() * 100
        print(f"Forgo rate: {forgo_rate:.1f}% ({int(valid.sum()):,} / {len(valid):,})")
    
    if "any_barrier" in processed.columns:
        valid = processed["any_barrier"].dropna()
        barrier_rate = valid.mean() * 100
        print(f"Any barrier: {barrier_rate:.1f}% ({int(valid.sum()):,} / {len(valid):,})")
    
    # Save
    save_processed_data(processed)
    
    return processed


if __name__ == "__main__":
    main()