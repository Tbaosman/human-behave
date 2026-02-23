"""
Patient Agent: Simulates healthcare decision-making.

Minimal prompt design - tests natural LLM behavior.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agents.base_agent import BaseAgent, AgentConfig, AgentResponse, extract_json_from_response, test_ollama_connection

from src.profiles.patient_profile import PatientProfile
from configs.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE


class PatientAgent(BaseAgent):
    """
    Agent that simulate (predict) healthcare delay/forgo decisions.
    
    Uses MEPS-style retrospective questions with minimal prompting.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE):
        config = AgentConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=512,
            system_prompt=""  # Minimal - no system prompt
        )
        super().__init__(config)
    
    def build_prompt(self, profile: PatientProfile) -> str:
        """
        Build MEPS-style prompt.
        """
        narrative = profile.to_narrative()
        
        prompt = f"""You are this person:

    {narrative}

    ---

    Answer as this person would on a health survey.

    Question 1: In the past 12 months, did you DELAY getting medical care because you couldn't afford it?
    Question 2: In the past 12 months, was there any time you NEEDED medical care but DID NOT GET IT because you couldn't afford it?

    Think about your financial situation, health needs, and insurance coverage. Then respond in this JSON format:

    {{
    "delay": <0 for No or 1 for Yes>,
    "forgo": <0 for No or 1 for Yes>,
    "reasoning": "<your explanation here>"
    }}"""
        
        return prompt
    
    def parse_response(self, raw_response: str) -> AgentResponse:
        """Parse LLM response into structured output."""
        
        parsed = extract_json_from_response(raw_response)
        
        if parsed:
            delay = self._normalize_binary(parsed.get("delay", 0))
            forgo = self._normalize_binary(parsed.get("forgo", 0))
            reasoning = parsed.get("reasoning", "")
        else:
            # Fallback parsing
            delay, forgo, reasoning = self._fallback_parse(raw_response)
        
        return AgentResponse(
            delay=delay,
            forgo=forgo,
            reasoning=reasoning,
            raw_response=raw_response
        )
    
    def _normalize_binary(self, value) -> int:
        """Normalize to 0 or 1."""
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, int):
            return 1 if value >= 1 else 0
        if isinstance(value, str):
            return 1 if value.lower().strip() in ["1", "yes", "true"] else 0
        return 0
    
    def _fallback_parse(self, response: str) -> tuple:
        """Fallback when JSON parsing fails."""
        response_lower = response.lower()
        
        delay = 0
        forgo = 0
        
        # Look for delay indicators
        if any(word in response_lower for word in ["delay", "waited", "postponed", "put off"]):
            if "did not" not in response_lower and "didn't" not in response_lower:
                delay = 1
        
        # Look for forgo indicators
        if any(word in response_lower for word in ["forgo", "skip", "didn't get", "did not get", "went without"]):
            forgo = 1
        
        return delay, forgo, response[:200]
    
    def predict(self, profile: PatientProfile) -> AgentResponse:
        """
        Predict delay/forgo for a patient.
        
        Args:
            profile: Patient profile (no ground truth)
            
        Returns:
            AgentResponse with prediction
        """
        prompt = self.build_prompt(profile)
        raw_response = self.call_llm(prompt)
        print(f"DEBUG RAW RESPONSE:\n{raw_response}\n")
        return self.parse_response(raw_response)


if __name__ == "__main__":
    import pandas as pd
    from configs.config import PROCESSED_DATA_DIR
    from src.profiles.patient_profile import create_patient_profile, extract_ground_truth
    
    print("=" * 70)
    print("PATIENT AGENT TEST")
    print("=" * 70)
    
    # Test connection
    model = DEFAULT_MODEL
    print(f"\nTesting Ollama ({model})...")
    
    if not test_ollama_connection(model):
        print("Error: Ollama not available")
        exit(1)
    
    print("✓ Connected\n")
    
    # Load data
    data_path = PROCESSED_DATA_DIR / "patient_profiles.parquet"
    df = pd.read_parquet(data_path)
    
    # Initialize agent
    agent = PatientAgent()
    print(f"Agent: {agent}")
    
    # Test on a few diverse patients
    # Find one with barrier and one without
    with_barrier = df[df["any_barrier"] == 1].iloc[0] if "any_barrier" in df.columns else df.iloc[0]
    without_barrier = df[df["any_barrier"] == 0].iloc[0] if "any_barrier" in df.columns else df.iloc[1]
    
    test_cases = [
        ("With barrier (ground truth)", with_barrier),
        ("Without barrier (ground truth)", without_barrier),
    ]
    
    for label, row in test_cases:
        profile = create_patient_profile(row)
        truth = extract_ground_truth(row)
        
        print(f"\n{'=' * 70}")
        print(f"TEST: {label}")
        print("=" * 70)
        
        print(f"\n--- PROFILE SUMMARY ---")
        print(f"Age: {profile.age}, Sex: {profile.sex}")
        print(f"Income: ${profile.family_income:,.0f}, Insurance: {profile.insurance_status}")
        print(f"Health: {profile.health_status}, Chronic conditions: {profile.chronic_count}")
        
        print(f"\n--- GROUND TRUTH ---")
        print(f"Delayed: {truth.delayed_care}, Forgone: {truth.forgone_care}")
        
        print(f"\n--- LLM PREDICTION ---")
        response = agent.predict(profile)
        print(f"Delayed: {response.delay}, Forgone: {response.forgo}")
        print(f"Reasoning: {response.reasoning[:200]}")
        
        # Check accuracy
        delay_correct = response.delay == truth.delayed_care
        forgo_correct = response.forgo == truth.forgone_care
        print(f"\n--- ACCURACY ---")
        print(f"Delay correct: {'✓' if delay_correct else '✗'}")
        print(f"Forgo correct: {'✓' if forgo_correct else '✗'}")