"""
Debate Agent: Two agents independently predict, then an aggregator decides.

Architecture:
  Patient Profile → Agent A (temp=0.5) → prediction + reasoning
  Patient Profile → Agent B (temp=0.9) → prediction + reasoning
                        ↓
                    Aggregator Agent → final prediction

Tests whether debate/ensemble improves accuracy over single agent.
"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.base_agent import BaseAgent, AgentConfig, AgentResponse, extract_json_from_response
from src.profiles.patient_profile import PatientProfile
from configs.config import DEFAULT_MODEL


@dataclass
class DebateResult:
    """Full debate output including individual and final predictions."""
    
    # Agent A
    agent_a_delay: int
    agent_a_forgo: int
    agent_a_reasoning: str
    
    # Agent B
    agent_b_delay: int
    agent_b_forgo: int
    agent_b_reasoning: str
    
    # Aggregator
    final_delay: int
    final_forgo: int
    final_reasoning: str
    
    # Agreement
    agents_agreed_delay: bool
    agents_agreed_forgo: bool
    
    # Raw responses
    raw_a: str
    raw_b: str
    raw_aggregator: str


class IndividualAgent(BaseAgent):
    """Single prediction agent (used as Agent A or Agent B)."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, temperature: float = 0.7, agent_id: str = "A"):
        config = AgentConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=512,
            system_prompt=""
        )
        super().__init__(config)
        self.agent_id = agent_id
    
    def build_prompt(self, profile: PatientProfile) -> str:
        """Same minimal prompt as single agent."""
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
        """Parse response."""
        parsed = extract_json_from_response(raw_response)
        
        if parsed:
            delay = self._normalize_binary(parsed.get("delay", 0))
            forgo = self._normalize_binary(parsed.get("forgo", 0))
            reasoning = parsed.get("reasoning", "")
        else:
            delay, forgo, reasoning = self._fallback_parse(raw_response)
        
        return AgentResponse(
            delay=delay,
            forgo=forgo,
            reasoning=reasoning,
            raw_response=raw_response
        )
    
    def _normalize_binary(self, value) -> int:
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, int):
            return 1 if value >= 1 else 0
        if isinstance(value, str):
            return 1 if value.lower().strip() in ["1", "yes", "true"] else 0
        return 0
    
    def _fallback_parse(self, response: str) -> tuple:
        response_lower = response.lower()
        delay = 0
        forgo = 0
        
        if any(w in response_lower for w in ["delay", "waited", "postponed", "put off"]):
            if "did not" not in response_lower and "didn't" not in response_lower:
                delay = 1
        
        if any(w in response_lower for w in ["forgo", "skip", "didn't get", "did not get", "went without"]):
            forgo = 1
        
        return delay, forgo, response[:200]
    
    def predict(self, profile: PatientProfile) -> AgentResponse:
        prompt = self.build_prompt(profile)
        raw_response = self.call_llm(prompt)
        return self.parse_response(raw_response)


class AggregatorAgent(BaseAgent):
    """Weighs two agent opinions and makes final decision."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, temperature: float = 0.3):
        config = AgentConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=512,
            system_prompt=""
        )
        super().__init__(config)
    
    def build_prompt(
        self,
        profile: PatientProfile,
        response_a: AgentResponse,
        response_b: AgentResponse
    ) -> str:
        """Build aggregator prompt with both opinions."""
        narrative = profile.to_narrative()
        
        prompt = f"""You are this person:

{narrative}

---

Two responses were given on your behalf to a health survey about whether you delayed or went without medical care due to cost. Review both and decide which better represents what you would actually do.

Response 1:
- Delay: {"Yes" if response_a.delay == 1 else "No"}
- Forgo: {"Yes" if response_a.forgo == 1 else "No"}
- Reasoning: {response_a.reasoning}

Response 2:
- Delay: {"Yes" if response_b.delay == 1 else "No"}
- Forgo: {"Yes" if response_b.forgo == 1 else "No"}
- Reasoning: {response_b.reasoning}

Based on your actual circumstances, which response is more accurate? Give your final answer.

{{
  "delay": <0 for No or 1 for Yes>,
  "forgo": <0 for No or 1 for Yes>,
  "reasoning": "<explain which response you agree with and why>"
}}"""
        
        return prompt
    
    def parse_response(self, raw_response: str) -> AgentResponse:
        """Parse aggregator response."""
        parsed = extract_json_from_response(raw_response)
        
        if parsed:
            delay = self._normalize_binary(parsed.get("delay", 0))
            forgo = self._normalize_binary(parsed.get("forgo", 0))
            reasoning = parsed.get("reasoning", "")
        else:
            delay, forgo, reasoning = self._fallback_parse(raw_response)
        
        return AgentResponse(
            delay=delay,
            forgo=forgo,
            reasoning=reasoning,
            raw_response=raw_response
        )
    
    def _normalize_binary(self, value) -> int:
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, int):
            return 1 if value >= 1 else 0
        if isinstance(value, str):
            return 1 if value.lower().strip() in ["1", "yes", "true"] else 0
        return 0
    
    def _fallback_parse(self, response: str) -> tuple:
        response_lower = response.lower()
        delay = 0
        forgo = 0
        
        if any(w in response_lower for w in ["delay", "waited", "postponed"]):
            if "did not" not in response_lower and "didn't" not in response_lower:
                delay = 1
        
        if any(w in response_lower for w in ["forgo", "skip", "didn't get", "did not get"]):
            forgo = 1
        
        return delay, forgo, response[:200]
    
    def aggregate(
        self,
        profile: PatientProfile,
        response_a: AgentResponse,
        response_b: AgentResponse
    ) -> AgentResponse:
        prompt = self.build_prompt(profile, response_a, response_b)
        raw_response = self.call_llm(prompt)
        return self.parse_response(raw_response)


class DebateAgent:
    """
    Orchestrates debate between two agents + aggregator.
    
    Agent A: Lower temperature (0.5) - more conservative
    Agent B: Higher temperature (0.9) - more diverse
    Aggregator: Low temperature (0.3) - careful weighing
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temp_a: float = 0.5,
        temp_b: float = 0.9,
        temp_aggregator: float = 0.3,
    ):
        self.agent_a = IndividualAgent(model_name=model_name, temperature=temp_a, agent_id="A")
        self.agent_b = IndividualAgent(model_name=model_name, temperature=temp_b, agent_id="B")
        self.aggregator = AggregatorAgent(model_name=model_name, temperature=temp_aggregator)
        self.model_name = model_name
    
    def predict(self, profile: PatientProfile) -> DebateResult:
        """
        Run full debate: A predicts, B predicts, Aggregator decides.
        
        Args:
            profile: Patient profile
            
        Returns:
            DebateResult with all predictions
        """
        # Agent A prediction
        response_a = self.agent_a.predict(profile)
        
        # Agent B prediction
        response_b = self.agent_b.predict(profile)
        
        # Check agreement
        agreed_delay = response_a.delay == response_b.delay
        agreed_forgo = response_a.forgo == response_b.forgo
        
        # If both agree, skip aggregator (faster + avoids unnecessary call)
        if agreed_delay and agreed_forgo:
            final = AgentResponse(
                delay=response_a.delay,
                forgo=response_a.forgo,
                reasoning=f"Both agents agreed. Agent A: {response_a.reasoning}",
                raw_response="AGREED - aggregator skipped"
            )
        else:
            # Disagreement → aggregator decides
            final = self.aggregator.aggregate(profile, response_a, response_b)
        
        return DebateResult(
            agent_a_delay=response_a.delay,
            agent_a_forgo=response_a.forgo,
            agent_a_reasoning=response_a.reasoning,
            agent_b_delay=response_b.delay,
            agent_b_forgo=response_b.forgo,
            agent_b_reasoning=response_b.reasoning,
            final_delay=final.delay,
            final_forgo=final.forgo,
            final_reasoning=final.reasoning,
            agents_agreed_delay=agreed_delay,
            agents_agreed_forgo=agreed_forgo,
            raw_a=response_a.raw_response,
            raw_b=response_b.raw_response,
            raw_aggregator=final.raw_response,
        )
    
    def __repr__(self) -> str:
        return (
            f"DebateAgent(model={self.model_name}, "
            f"temp_a={self.agent_a.config.temperature}, "
            f"temp_b={self.agent_b.config.temperature}, "
            f"temp_agg={self.aggregator.config.temperature})"
        )


if __name__ == "__main__":
    import pandas as pd
    from configs.config import PROCESSED_DATA_DIR
    from src.profiles.patient_profile import create_patient_profile, extract_ground_truth
    from src.agents.base_agent import test_ollama_connection
    
    print("=" * 70)
    print("DEBATE AGENT TEST")
    print("=" * 70)
    
    # Test connection
    print(f"\nTesting Ollama ({DEFAULT_MODEL})...")
    if not test_ollama_connection(DEFAULT_MODEL):
        print("Error: Ollama not available")
        exit(1)
    print("✓ Connected\n")
    
    # Load data
    data_path = PROCESSED_DATA_DIR / "patient_profiles.parquet"
    df = pd.read_parquet(data_path)
    
    # Initialize debate agent
    debate = DebateAgent()
    print(f"Agent: {debate}\n")
    
    # Test: one with barrier, one without
    with_barrier = df[df["any_barrier"] == 1].iloc[0]
    without_barrier = df[df["any_barrier"] == 0].iloc[0]
    
    test_cases = [
        ("With barrier", with_barrier),
        ("Without barrier", without_barrier),
    ]
    
    for label, row in test_cases:
        profile = create_patient_profile(row)
        truth = extract_ground_truth(row)
        
        print(f"{'=' * 70}")
        print(f"TEST: {label}")
        print(f"{'=' * 70}")
        print(f"Age: {profile.age}, Income: ${profile.family_income:,.0f}, Insurance: {profile.insurance_status}")
        print(f"\nGround Truth — Delay: {truth.delayed_care}, Forgo: {truth.forgone_care}")
        
        result = debate.predict(profile)
        
        print(f"\n--- Agent A (temp=0.5) ---")
        print(f"Delay: {result.agent_a_delay}, Forgo: {result.agent_a_forgo}")
        print(f"Reasoning: {result.agent_a_reasoning[:300]}")
        
        print(f"\n--- Agent B (temp=0.9) ---")
        print(f"Delay: {result.agent_b_delay}, Forgo: {result.agent_b_forgo}")
        print(f"Reasoning: {result.agent_b_reasoning[:300]}")
        
        print(f"\n--- Agreement ---")
        print(f"Delay agreed: {result.agents_agreed_delay}")
        print(f"Forgo agreed: {result.agents_agreed_forgo}")
        
        print(f"\n--- Final Decision ---")
        print(f"Delay: {result.final_delay}, Forgo: {result.final_forgo}")
        print(f"Reasoning: {result.final_reasoning[:400]}")
        
        # Accuracy
        delay_correct = result.final_delay == truth.delayed_care
        forgo_correct = result.final_forgo == truth.forgone_care
        print(f"\n--- Accuracy ---")
        print(f"Delay: {'✓' if delay_correct else '✗'}")
        print(f"Forgo: {'✓' if forgo_correct else '✗'}")
        print()