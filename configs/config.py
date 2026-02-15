"""
Project configuration.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# MEPS file
MEPS_FILE = RAW_DATA_DIR / "h251.dta"
PROCESSED_FILE = PROCESSED_DATA_DIR / "patient_profiles.parquet"

# Ground truth variables (DO NOT expose to LLM)
GROUND_TRUTH_VARS = [
    "DLAYCA42",      # Delayed care due to cost
    "AFRDCA42",      # Forgone care due to cost
]

# Default model
DEFAULT_MODEL = "llama3.2:latest"
DEFAULT_TEMPERATURE = 0.7

# Ollama settings
OLLAMA_HOST = "http://localhost:11434"