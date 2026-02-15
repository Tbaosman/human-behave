"""
Patient profile for LLM input.

CRITICAL: This module ensures ground truth (delay/forgo) is NEVER exposed to the LLM.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import GROUND_TRUTH_VARS


# Value mappings
SEX_MAP = {1: "male", 2: "female"}
MARITAL_MAP = {1: "married", 2: "widowed", 3: "divorced", 4: "separated", 5: "never married"}
REGION_MAP = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}
POVERTY_MAP = {1: "poor", 2: "near poor", 3: "low income", 4: "middle income", 5: "high income"}
INSURANCE_MAP = {1: "private", 2: "public only", 3: "uninsured"}
HEALTH_MAP = {1: "excellent", 2: "very good", 3: "good", 4: "fair", 5: "poor"}
EMPLOYMENT_MAP = {1: "employed", 2: "have a job but not at work", 3: "unemployed", 4: "not in labor force"}


def safe_map(value, mapping: dict, default: str = "unknown") -> str:
    """Safely map a value."""
    if pd.isna(value) or value < 0:
        return default
    return mapping.get(int(value), default)


def safe_int(value, default: int = 0) -> int:
    """Safely convert to int."""
    if pd.isna(value) or value < 0:
        return default
    return int(value)


def safe_float(value, default: float = 0.0) -> float:
    """Safely convert to float."""
    if pd.isna(value) or value < 0:
        return default
    return float(value)


@dataclass
class PatientProfile:
    """
    Patient profile for LLM input.
    Contains NO ground truth information.
    """
    
    # === Fields WITHOUT defaults (must come first) ===
    
    # Identifier
    person_id: str
    
    # Demographics
    age: int
    sex: str
    marital_status: str
    region: str
    family_size: int
    
    # Economic
    poverty_category: str
    poverty_level_pct: float
    family_income: float
    employment_status: str
    
    # Insurance
    insurance_status: str
    
    # Health
    health_status: str
    mental_health_status: str
    chronic_count: int
    
    # Healthcare access & utilization
    has_usual_care_source: bool
    office_visits: int
    er_visits: int
    inpatient_stays: int
    
    # Costs
    total_expenditure: float
    out_of_pocket: float
    oop_burden_ratio: float
    
    # === Fields WITH defaults (must come last) ===
    chronic_conditions: List[str] = field(default_factory=list)
    
    def to_narrative(self) -> str:
        """
        Convert to first-person narrative for LLM.
        
        CRITICAL: No ground truth information included.
        """
        parts = []
        
        # Demographics
        parts.append(f"I am a {self.age}-year-old {self.sex}")
        if self.marital_status != "unknown":
            parts.append(f", {self.marital_status}")
        parts.append(f", living in the {self.region}")
        if self.family_size > 1:
            parts.append(f" with a household of {self.family_size} people")
        parts.append(".")
        
        # Economic
        parts.append(f"\n\nMy household income is ${self.family_income:,.0f} per year")
        parts.append(f", putting me in the {self.poverty_category} category")
        parts.append(f" ({self.poverty_level_pct:.0f}% of the federal poverty level).")
        
        if self.employment_status != "unknown":
            parts.append(f" I am currently {self.employment_status}.")
        
        # Insurance
        parts.append(f"\n\nFor health insurance, I have {self.insurance_status} insurance.")
        
        # Health status
        parts.append(f"\n\nI would rate my physical health as {self.health_status}")
        parts.append(f" and my mental health as {self.mental_health_status}.")
        
        if self.chronic_count > 0:
            parts.append(f" I have {self.chronic_count} chronic condition(s)")
            if self.chronic_conditions:
                parts.append(f": {', '.join(self.chronic_conditions)}")
            parts.append(".")
        
        # Healthcare access
        if self.has_usual_care_source:
            parts.append("\n\nI have a usual place where I go for medical care.")
        else:
            parts.append("\n\nI do not have a usual doctor or place for medical care.")
        
        # Utilization
        usage = []
        if self.office_visits > 0:
            usage.append(f"{self.office_visits} office visit(s)")
        if self.er_visits > 0:
            usage.append(f"{self.er_visits} emergency room visit(s)")
        if self.inpatient_stays > 0:
            usage.append(f"{self.inpatient_stays} hospital stay(s)")
        
        if usage:
            parts.append(f" In the past year, I had {', '.join(usage)}.")
        else:
            parts.append(" I did not use any healthcare services in the past year.")
        
        # Costs
        parts.append(f"\n\nMy total healthcare costs last year were ${self.total_expenditure:,.0f}")
        parts.append(f", of which I paid ${self.out_of_pocket:,.0f} out of pocket.")
        
        if self.oop_burden_ratio > 0.1:
            parts.append(f" This was {self.oop_burden_ratio * 100:.1f}% of my household income.")
        
        return "".join(parts)


@dataclass
class GroundTruth:
    """
    Ground truth for evaluation.
    Kept separate from PatientProfile.
    """
    person_id: str
    delayed_care: int  # 0 or 1
    forgone_care: int  # 0 or 1
    any_barrier: int   # 0 or 1


def create_patient_profile(row: pd.Series) -> PatientProfile:
    """
    Create PatientProfile from MEPS row.
    
    Args:
        row: Pandas Series with MEPS data
        
    Returns:
        PatientProfile (without ground truth)
    """
    # Build chronic conditions list
    chronic_conditions = []
    condition_map = {
        "has_hypertension": "high blood pressure",
        "has_diabetes": "diabetes",
        "has_asthma": "asthma",
        "has_arthritis": "arthritis",
        "has_heart_disease": "heart disease",
        "has_stroke": "history of stroke",
        "has_cancer": "cancer",
    }
    
    for var, name in condition_map.items():
        if var in row.index and row.get(var) == 1:
            chronic_conditions.append(name)
    
    return PatientProfile(
        person_id=str(row.get("person_id", "unknown")),
        age=safe_int(row.get("age")),
        sex=safe_map(row.get("sex"), SEX_MAP),
        marital_status=safe_map(row.get("marital_status"), MARITAL_MAP),
        region=safe_map(row.get("region"), REGION_MAP),
        family_size=safe_int(row.get("family_size"), default=1),
        poverty_category=safe_map(row.get("poverty_category"), POVERTY_MAP),
        poverty_level_pct=safe_float(row.get("poverty_level_pct")),
        family_income=safe_float(row.get("family_income")),
        employment_status=safe_map(row.get("employment_status"), EMPLOYMENT_MAP),
        insurance_status=safe_map(row.get("insurance_coverage"), INSURANCE_MAP),
        health_status=safe_map(row.get("health_status"), HEALTH_MAP),
        mental_health_status=safe_map(row.get("mental_health_status"), HEALTH_MAP),
        chronic_count=safe_int(row.get("chronic_count")),
        has_usual_care_source=row.get("has_usual_care_source") == 1,
        office_visits=safe_int(row.get("office_visits")),
        er_visits=safe_int(row.get("er_visits")),
        inpatient_stays=safe_int(row.get("inpatient_stays")),
        total_expenditure=safe_float(row.get("total_expenditure")),
        out_of_pocket=safe_float(row.get("out_of_pocket")),
        oop_burden_ratio=safe_float(row.get("oop_burden_ratio")),
        chronic_conditions=chronic_conditions,
    )


def extract_ground_truth(row: pd.Series) -> GroundTruth:
    """
    Extract ground truth from MEPS row.
    
    Args:
        row: Pandas Series with MEPS data
        
    Returns:
        GroundTruth
    """
    delayed = 1 if row.get("delayed_care") == 1 else 0
    forgone = 1 if row.get("forgone_care") == 1 else 0
    any_barrier = 1 if (delayed == 1 or forgone == 1) else 0
    
    return GroundTruth(
        person_id=str(row.get("person_id", "unknown")),
        delayed_care=delayed,
        forgone_care=forgone,
        any_barrier=any_barrier,
    )


if __name__ == "__main__":
    from configs.config import PROCESSED_DATA_DIR
    
    print("=" * 70)
    print("PATIENT PROFILE TEST")
    print("=" * 70)
    
    # Load processed data
    data_path = PROCESSED_DATA_DIR / "patient_profiles.parquet"
    
    if not data_path.exists():
        print(f"Error: Run data_processor.py first")
        exit(1)
    
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} records")
    
    # Test with a few patients
    for i in range(3):
        row = df.iloc[i]
        
        profile = create_patient_profile(row)
        truth = extract_ground_truth(row)
        
        print(f"\n{'=' * 70}")
        print(f"PATIENT {i + 1}: {profile.person_id}")
        print("=" * 70)
        
        print("\n--- NARRATIVE (sent to LLM) ---")
        print(profile.to_narrative())
        
        print("\n--- GROUND TRUTH (for evaluation only) ---")
        print(f"Delayed care: {truth.delayed_care}")
        print(f"Forgone care: {truth.forgone_care}")
        print(f"Any barrier: {truth.any_barrier}")