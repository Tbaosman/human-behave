"""
Simulation runner: Run predictions across multiple patients.
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import PROCESSED_DATA_DIR, OUTPUTS_DIR, DEFAULT_MODEL, DEFAULT_TEMPERATURE
from src.profiles.patient_profile import create_patient_profile, extract_ground_truth
from src.agents.patient_agent import PatientAgent
from src.agents.base_agent import test_ollama_connection


def get_balanced_sample(df: pd.DataFrame, n_per_group: int = 10, random_seed: int = 42) -> pd.DataFrame:
    """
    Get balanced sample: n with barriers, n without.
    
    Args:
        df: Full dataset
        n_per_group: Number per group
        random_seed: For reproducibility
    
    Returns:
        Balanced DataFrame
    """
    with_barrier = df[df["any_barrier"] == 1].sample(
        n=min(n_per_group, len(df[df["any_barrier"] == 1])),
        random_state=random_seed
    )
    
    without_barrier = df[df["any_barrier"] == 0].sample(
        n=min(n_per_group, len(df[df["any_barrier"] == 0])),
        random_state=random_seed
    )
    
    balanced = pd.concat([with_barrier, without_barrier]).sample(
        frac=1, random_state=random_seed  # Shuffle
    )
    
    print(f"Balanced sample: {len(with_barrier)} with barrier, {len(without_barrier)} without")
    
    return balanced


def run_simulation(
    n_patients: int = 100,
    model_name: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    random_seed: int = 42,
    output_name: Optional[str] = None,
    balanced: bool = False,
) -> pd.DataFrame:
    """
    Run simulation on n patients.
    
    Args:
        n_patients: Number of patients to simulate
        model_name: Ollama model to use
        temperature: Model temperature
        random_seed: For reproducible sampling
        output_name: Output filename (auto-generated if None)
        balanced: If True, use balanced sampling (equal barriers/no barriers)
    
    Returns:
        DataFrame with results
    """
    
    # Load data
    data_path = PROCESSED_DATA_DIR / "patient_profiles.parquet"
    df = pd.read_parquet(data_path)
    
    # Sample patients
    if balanced:
        sample_df = get_balanced_sample(df, n_per_group=n_patients // 2, random_seed=random_seed)
    elif n_patients >= len(df):
        sample_df = df.copy()
    else:
        sample_df = df.sample(n=n_patients, random_state=random_seed)
    
    print(f"Running simulation: {len(sample_df)} patients, model={model_name}")
    
    # Initialize agent
    agent = PatientAgent(model_name=model_name, temperature=temperature)
    
    # Run predictions
    results = []
    
    for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Simulating"):
        profile = create_patient_profile(row)
        truth = extract_ground_truth(row)
        
        try:
            response = agent.predict(profile)
            
            results.append({
                # Identifiers
                "person_id": profile.person_id,
                
                # Demographics
                "age": profile.age,
                "sex": profile.sex,
                "insurance_status": profile.insurance_status,
                "poverty_category": profile.poverty_category,
                "family_income": profile.family_income,
                "health_status": profile.health_status,
                "chronic_count": profile.chronic_count,
                
                # Predictions
                "pred_delay": response.delay,
                "pred_forgo": response.forgo,
                "pred_any_barrier": 1 if (response.delay == 1 or response.forgo == 1) else 0,
                "reasoning": response.reasoning,
                
                # Ground truth
                "true_delay": truth.delayed_care,
                "true_forgo": truth.forgone_care,
                "true_any_barrier": truth.any_barrier,
                
                # Accuracy
                "delay_correct": int(response.delay == truth.delayed_care),
                "forgo_correct": int(response.forgo == truth.forgone_care),
                "any_barrier_correct": int(
                    (1 if (response.delay == 1 or response.forgo == 1) else 0) == truth.any_barrier
                ),
                
                # Metadata
                "model": model_name,
                "error": None,
            })
            
        except Exception as e:
            results.append({
                "person_id": profile.person_id,
                "error": str(e),
                "model": model_name,
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        balance_tag = "_balanced" if balanced else ""
        output_name = f"sim_{model_name.replace(':', '_')}_{len(sample_df)}{balance_tag}_{timestamp}"
    
    csv_path = OUTPUTS_DIR / f"{output_name}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save config
    config = {
        "n_patients": len(sample_df),
        "model_name": model_name,
        "temperature": temperature,
        "random_seed": random_seed,
        "balanced": balanced,
        "timestamp": datetime.now().isoformat(),
    }
    config_path = OUTPUTS_DIR / f"{output_name}_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Print summary
    print_summary(results_df)
    
    return results_df


def print_summary(df: pd.DataFrame):
    """Print simulation summary."""
    
    print(f"\n{'=' * 70}")
    print("SIMULATION SUMMARY")
    print("=" * 70)
    
    valid = df[df["error"].isna()]
    errors = df[df["error"].notna()]
    
    print(f"\nTotal: {len(df)}, Valid: {len(valid)}, Errors: {len(errors)}")
    
    if len(valid) == 0:
        print("No valid results!")
        return
    
    # Ground truth rates
    print(f"\n--- Ground Truth Rates ---")
    print(f"Delay rate: {valid['true_delay'].mean() * 100:.1f}%")
    print(f"Forgo rate: {valid['true_forgo'].mean() * 100:.1f}%")
    print(f"Any barrier: {valid['true_any_barrier'].mean() * 100:.1f}%")
    
    # Predicted rates
    print(f"\n--- Predicted Rates ---")
    print(f"Delay rate: {valid['pred_delay'].mean() * 100:.1f}%")
    print(f"Forgo rate: {valid['pred_forgo'].mean() * 100:.1f}%")
    print(f"Any barrier: {valid['pred_any_barrier'].mean() * 100:.1f}%")
    
    # Accuracy
    print(f"\n--- Accuracy ---")
    print(f"Delay accuracy: {valid['delay_correct'].mean() * 100:.1f}%")
    print(f"Forgo accuracy: {valid['forgo_correct'].mean() * 100:.1f}%")
    print(f"Any barrier accuracy: {valid['any_barrier_correct'].mean() * 100:.1f}%")
    
    # Confusion matrix style breakdown
    print(f"\n--- Prediction Breakdown ---")
    
    # True positives, false positives, etc. for any_barrier
    tp = len(valid[(valid['pred_any_barrier'] == 1) & (valid['true_any_barrier'] == 1)])
    fp = len(valid[(valid['pred_any_barrier'] == 1) & (valid['true_any_barrier'] == 0)])
    tn = len(valid[(valid['pred_any_barrier'] == 0) & (valid['true_any_barrier'] == 0)])
    fn = len(valid[(valid['pred_any_barrier'] == 0) & (valid['true_any_barrier'] == 1)])
    
    print(f"True Positives (predicted barrier, had barrier): {tp}")
    print(f"False Positives (predicted barrier, no barrier): {fp}")
    print(f"True Negatives (predicted no barrier, no barrier): {tn}")
    print(f"False Negatives (predicted no barrier, had barrier): {fn}")
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run healthcare decision simulation")
    parser.add_argument("-n", "--n_patients", type=int, default=10, help="Number of patients")
    parser.add_argument("-m", "--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("-t", "--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Temperature")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-b", "--balanced", action="store_true", help="Use balanced sampling")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HEALTHCARE DECISION SIMULATION")
    print("=" * 70)
    
    # Test connection
    print(f"\nTesting Ollama ({args.model})...")
    if not test_ollama_connection(args.model):
        print("Error: Ollama not available")
        exit(1)
    print("✓ Connected\n")
    
    # Run simulation
    results = run_simulation(
        n_patients=args.n_patients,
        model_name=args.model,
        temperature=args.temperature,
        random_seed=args.seed,
        balanced=args.balanced,
    )