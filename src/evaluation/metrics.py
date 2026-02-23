"""
Evaluation metrics for comparing LLM predictions to MEPS ground truth.

Supports:
- Individual model evaluation
- Single vs Debate comparison
- Multi-model comparison
- Subgroup analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from configs.config import OUTPUTS_DIR


# ============================================
# DATA CLASSES
# ============================================

@dataclass
class BinaryMetrics:
    """Metrics for a single binary prediction task."""
    name: str
    n_samples: int
    
    # Counts
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Rates
    accuracy: float
    precision: float
    recall: float
    f1: float
    
    # Distribution
    true_rate: float
    predicted_rate: float
    rate_difference: float
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "tp": self.true_positives,
            "fp": self.false_positives,
            "tn": self.true_negatives,
            "fn": self.false_negatives,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "true_rate": round(self.true_rate, 4),
            "predicted_rate": round(self.predicted_rate, 4),
            "rate_difference": round(self.rate_difference, 4),
        }


@dataclass
class ModelEvaluation:
    """Complete evaluation for one model/approach."""
    model_name: str
    approach: str  # "single" or "debate"
    n_patients: int
    
    delay_metrics: BinaryMetrics
    forgo_metrics: BinaryMetrics
    any_barrier_metrics: BinaryMetrics
    
    subgroup_results: Optional[Dict] = None
    
    def to_dict(self) -> dict:
        result = {
            "model_name": self.model_name,
            "approach": self.approach,
            "n_patients": self.n_patients,
            "delay": self.delay_metrics.to_dict(),
            "forgo": self.forgo_metrics.to_dict(),
            "any_barrier": self.any_barrier_metrics.to_dict(),
        }
        if self.subgroup_results:
            result["subgroups"] = self.subgroup_results
        return result


# ============================================
# CORE METRICS FUNCTIONS
# ============================================

def calculate_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "metric"
) -> BinaryMetrics:
    """
    Calculate all metrics for a binary prediction task.
    
    Args:
        y_true: Ground truth (0/1)
        y_pred: Predictions (0/1)
        name: Label for this metric
    
    Returns:
        BinaryMetrics
    """
    # Remove NaN
    mask = ~(pd.isna(y_true) | pd.isna(y_pred))
    y_true = np.array(y_true[mask], dtype=int)
    y_pred = np.array(y_pred[mask], dtype=int)
    
    n = len(y_true)
    
    if n == 0:
        return BinaryMetrics(
            name=name, n_samples=0,
            true_positives=0, false_positives=0,
            true_negatives=0, false_negatives=0,
            accuracy=0, precision=0, recall=0, f1=0,
            true_rate=0, predicted_rate=0, rate_difference=0,
        )
    
    # Confusion matrix
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    
    # Metrics
    accuracy = (tp + tn) / n if n > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Rates
    true_rate = np.mean(y_true)
    predicted_rate = np.mean(y_pred)
    rate_difference = predicted_rate - true_rate
    
    return BinaryMetrics(
        name=name,
        n_samples=n,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        true_rate=true_rate,
        predicted_rate=predicted_rate,
        rate_difference=rate_difference,
    )


def evaluate_results(df: pd.DataFrame, model_name: str = "", approach: str = "") -> ModelEvaluation:
    """
    Evaluate a full results DataFrame.
    
    Args:
        df: Results DataFrame with pred_* and true_* columns
        model_name: Model identifier
        approach: "single" or "debate"
    
    Returns:
        ModelEvaluation
    """
    # Filter valid results
    valid = df[df["error"].isna()].copy() if "error" in df.columns else df.copy()
    
    # Calculate metrics for each task
    delay_metrics = calculate_binary_metrics(
        valid["true_delay"].values,
        valid["pred_delay"].values,
        name="delay"
    )
    
    forgo_metrics = calculate_binary_metrics(
        valid["true_forgo"].values,
        valid["pred_forgo"].values,
        name="forgo"
    )
    
    any_barrier_metrics = calculate_binary_metrics(
        valid["true_any_barrier"].values,
        valid["pred_any_barrier"].values,
        name="any_barrier"
    )
    
    # Subgroup analysis
    subgroups = calculate_subgroup_metrics(valid)
    
    return ModelEvaluation(
        model_name=model_name or valid["model"].iloc[0] if "model" in valid.columns else "unknown",
        approach=approach,
        n_patients=len(valid),
        delay_metrics=delay_metrics,
        forgo_metrics=forgo_metrics,
        any_barrier_metrics=any_barrier_metrics,
        subgroup_results=subgroups,
    )


# ============================================
# SUBGROUP ANALYSIS
# ============================================

def calculate_subgroup_metrics(df: pd.DataFrame) -> Dict:
    """
    Calculate metrics by subgroup.
    
    Subgroups: insurance status, poverty category, health status
    """
    subgroups = {}
    
    # By insurance status
    if "insurance_status" in df.columns:
        subgroups["by_insurance"] = _subgroup_analysis(
            df, group_col="insurance_status"
        )
    
    # By poverty category
    if "poverty_category" in df.columns:
        subgroups["by_poverty"] = _subgroup_analysis(
            df, group_col="poverty_category"
        )
    
    # By health status
    if "health_status" in df.columns:
        subgroups["by_health"] = _subgroup_analysis(
            df, group_col="health_status"
        )
    
    # By chronic condition count (grouped)
    if "chronic_count" in df.columns:
        df = df.copy()
        df["chronic_group"] = df["chronic_count"].apply(
            lambda x: "none" if x == 0 else ("1-2" if x <= 2 else "3+")
        )
        subgroups["by_chronic"] = _subgroup_analysis(
            df, group_col="chronic_group"
        )
    
    return subgroups


def _subgroup_analysis(df: pd.DataFrame, group_col: str) -> Dict:
    """Calculate any_barrier metrics for each subgroup."""
    results = {}
    
    for group_name, group_df in df.groupby(group_col):
        if len(group_df) < 2:
            continue
        
        metrics = calculate_binary_metrics(
            group_df["true_any_barrier"].values,
            group_df["pred_any_barrier"].values,
            name=str(group_name)
        )
        
        results[str(group_name)] = {
            "n": metrics.n_samples,
            "accuracy": round(metrics.accuracy, 4),
            "precision": round(metrics.precision, 4),
            "recall": round(metrics.recall, 4),
            "f1": round(metrics.f1, 4),
            "true_rate": round(metrics.true_rate, 4),
            "predicted_rate": round(metrics.predicted_rate, 4),
        }
    
    return results


# ============================================
# COMPARISON FUNCTIONS
# ============================================

def compare_approaches(
    single_df: pd.DataFrame,
    debate_df: pd.DataFrame,
    model_name: str = ""
) -> Dict:
    """
    Compare single agent vs debate agent results.
    
    Args:
        single_df: Single agent results
        debate_df: Debate agent results
        model_name: Model used
    
    Returns:
        Comparison dictionary
    """
    single_eval = evaluate_results(single_df, model_name, "single")
    debate_eval = evaluate_results(debate_df, model_name, "debate")
    
    comparison = {
        "model": model_name,
        "single": single_eval.to_dict(),
        "debate": debate_eval.to_dict(),
        "improvement": {
            "delay_f1": debate_eval.delay_metrics.f1 - single_eval.delay_metrics.f1,
            "forgo_f1": debate_eval.forgo_metrics.f1 - single_eval.forgo_metrics.f1,
            "any_barrier_f1": debate_eval.any_barrier_metrics.f1 - single_eval.any_barrier_metrics.f1,
            "delay_accuracy": debate_eval.delay_metrics.accuracy - single_eval.delay_metrics.accuracy,
            "forgo_accuracy": debate_eval.forgo_metrics.accuracy - single_eval.forgo_metrics.accuracy,
            "any_barrier_accuracy": debate_eval.any_barrier_metrics.accuracy - single_eval.any_barrier_metrics.accuracy,
        }
    }
    
    return comparison


def compare_models(evaluations: List[ModelEvaluation]) -> pd.DataFrame:
    """
    Compare multiple models side by side.
    
    Args:
        evaluations: List of ModelEvaluation objects
    
    Returns:
        DataFrame with model comparison
    """
    rows = []
    
    for eval in evaluations:
        rows.append({
            "model": eval.model_name,
            "approach": eval.approach,
            "n_patients": eval.n_patients,
            "delay_accuracy": eval.delay_metrics.accuracy,
            "delay_f1": eval.delay_metrics.f1,
            "delay_precision": eval.delay_metrics.precision,
            "delay_recall": eval.delay_metrics.recall,
            "forgo_accuracy": eval.forgo_metrics.accuracy,
            "forgo_f1": eval.forgo_metrics.f1,
            "forgo_precision": eval.forgo_metrics.precision,
            "forgo_recall": eval.forgo_metrics.recall,
            "barrier_accuracy": eval.any_barrier_metrics.accuracy,
            "barrier_f1": eval.any_barrier_metrics.f1,
            "barrier_precision": eval.any_barrier_metrics.precision,
            "barrier_recall": eval.any_barrier_metrics.recall,
            "true_barrier_rate": eval.any_barrier_metrics.true_rate,
            "pred_barrier_rate": eval.any_barrier_metrics.predicted_rate,
            "rate_difference": eval.any_barrier_metrics.rate_difference,
        })
    
    return pd.DataFrame(rows)


# ============================================
# PRINTING FUNCTIONS
# ============================================

def print_evaluation(eval: ModelEvaluation):
    """Print detailed evaluation results."""
    
    print(f"\n{'=' * 70}")
    print(f"EVALUATION: {eval.model_name} ({eval.approach})")
    print(f"{'=' * 70}")
    print(f"Patients: {eval.n_patients}")
    
    for metrics in [eval.delay_metrics, eval.forgo_metrics, eval.any_barrier_metrics]:
        print(f"\n--- {metrics.name.upper()} ---")
        print(f"True rate: {metrics.true_rate * 100:.1f}%  |  Predicted rate: {metrics.predicted_rate * 100:.1f}%  |  Diff: {metrics.rate_difference * 100:+.1f}%")
        print(f"Accuracy: {metrics.accuracy * 100:.1f}%")
        print(f"Precision: {metrics.precision:.3f}  |  Recall: {metrics.recall:.3f}  |  F1: {metrics.f1:.3f}")
        print(f"TP: {metrics.true_positives}  FP: {metrics.false_positives}  TN: {metrics.true_negatives}  FN: {metrics.false_negatives}")
    
    # Subgroup results
    if eval.subgroup_results:
        print(f"\n--- SUBGROUP ANALYSIS ---")
        for group_type, groups in eval.subgroup_results.items():
            print(f"\n{group_type}:")
            print(f"  {'Group':<20} {'N':>5} {'Acc':>7} {'F1':>7} {'True%':>7} {'Pred%':>7}")
            print(f"  {'-' * 55}")
            for group_name, metrics in groups.items():
                print(f"  {group_name:<20} {metrics['n']:>5} {metrics['accuracy'] * 100:>6.1f}% {metrics['f1']:>6.3f} {metrics['true_rate'] * 100:>6.1f}% {metrics['predicted_rate'] * 100:>6.1f}%")


def print_comparison(single_eval: ModelEvaluation, debate_eval: ModelEvaluation):
    """Print side-by-side comparison of single vs debate."""
    
    print(f"\n{'=' * 70}")
    print(f"COMPARISON: Single Agent vs Debate Agent")
    print(f"Model: {single_eval.model_name}")
    print(f"{'=' * 70}")
    
    print(f"\n{'Metric':<25} {'Single':>10} {'Debate':>10} {'Diff':>10}")
    print(f"{'-' * 55}")
    
    pairs = [
        ("Delay Accuracy", single_eval.delay_metrics.accuracy, debate_eval.delay_metrics.accuracy),
        ("Delay F1", single_eval.delay_metrics.f1, debate_eval.delay_metrics.f1),
        ("Delay Precision", single_eval.delay_metrics.precision, debate_eval.delay_metrics.precision),
        ("Delay Recall", single_eval.delay_metrics.recall, debate_eval.delay_metrics.recall),
        ("", None, None),
        ("Forgo Accuracy", single_eval.forgo_metrics.accuracy, debate_eval.forgo_metrics.accuracy),
        ("Forgo F1", single_eval.forgo_metrics.f1, debate_eval.forgo_metrics.f1),
        ("Forgo Precision", single_eval.forgo_metrics.precision, debate_eval.forgo_metrics.precision),
        ("Forgo Recall", single_eval.forgo_metrics.recall, debate_eval.forgo_metrics.recall),
        ("", None, None),
        ("Barrier Accuracy", single_eval.any_barrier_metrics.accuracy, debate_eval.any_barrier_metrics.accuracy),
        ("Barrier F1", single_eval.any_barrier_metrics.f1, debate_eval.any_barrier_metrics.f1),
        ("Barrier Precision", single_eval.any_barrier_metrics.precision, debate_eval.any_barrier_metrics.precision),
        ("Barrier Recall", single_eval.any_barrier_metrics.recall, debate_eval.any_barrier_metrics.recall),
        ("", None, None),
        ("Barrier True Rate", single_eval.any_barrier_metrics.true_rate, debate_eval.any_barrier_metrics.true_rate),
        ("Barrier Pred Rate", single_eval.any_barrier_metrics.predicted_rate, debate_eval.any_barrier_metrics.predicted_rate),
    ]
    
    for name, single_val, debate_val in pairs:
        if single_val is None:
            print()
            continue
        diff = debate_val - single_val
        diff_str = f"{diff:+.3f}"
        print(f"{name:<25} {single_val:>10.3f} {debate_val:>10.3f} {diff_str:>10}")


def print_model_comparison(comparison_df: pd.DataFrame):
    """Print multi-model comparison table."""
    
    print(f"\n{'=' * 70}")
    print("MULTI-MODEL COMPARISON")
    print(f"{'=' * 70}")
    
    print(f"\n{'Model':<20} {'Approach':<10} {'N':>5} {'Barrier Acc':>12} {'Barrier F1':>12} {'Pred Rate':>12} {'True Rate':>12}")
    print(f"{'-' * 85}")
    
    for _, row in comparison_df.iterrows():
        print(
            f"{row['model']:<20} "
            f"{row['approach']:<10} "
            f"{row['n_patients']:>5} "
            f"{row['barrier_accuracy'] * 100:>11.1f}% "
            f"{row['barrier_f1']:>11.3f} "
            f"{row['pred_barrier_rate'] * 100:>11.1f}% "
            f"{row['true_barrier_rate'] * 100:>11.1f}%"
        )


# ============================================
# FILE LOADING
# ============================================

def load_results(filename: str) -> pd.DataFrame:
    """Load results CSV from outputs directory."""
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    return pd.read_csv(path)


# ============================================
# MAIN — Example Usage
# ============================================

if __name__ == "__main__":
    import glob
    
    print("=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)
    
    # Find all result files
    result_files = list(OUTPUTS_DIR.glob("*.csv"))
    
    if not result_files:
        print("\nNo result files found in outputs/")
        print("Run simulations first:")
        print("  python -m src.simulation.runner -n 20 -b")
        print("  python -m src.simulation.debate_runner -n 20 -b")
        exit(0)
    
    print(f"\nFound {len(result_files)} result files:")
    for f in result_files:
        print(f"  {f.name}")
    
    # Evaluate each file
    evaluations = []
    single_evals = []
    debate_evals = []
    
    for f in result_files:
        df = pd.read_csv(f)
        
        # Determine approach from filename
        if "debate" in f.name:
            approach = "debate"
        else:
            approach = "single"
        
        # Get model name
        model = df["model"].iloc[0] if "model" in df.columns else "unknown"
        
        # Evaluate
        eval_result = evaluate_results(df, model, approach)
        evaluations.append(eval_result)
        
        if approach == "single":
            single_evals.append(eval_result)
        else:
            debate_evals.append(eval_result)
        
        # Print individual evaluation
        print_evaluation(eval_result)
    
    # Compare single vs debate if both exist
    if single_evals and debate_evals:
        print_comparison(single_evals[0], debate_evals[0])
    
    # Multi-model comparison
    if len(evaluations) > 1:
        comparison_df = compare_models(evaluations)
        print_model_comparison(comparison_df)
        
        # Save comparison
        comparison_path = OUTPUTS_DIR / "model_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison saved to: {comparison_path}")