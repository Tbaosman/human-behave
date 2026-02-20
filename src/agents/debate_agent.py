"""
True Debate Agent: Multi-round debate between two agents.

Architecture:
  Round 1: Agent A predicts, Agent B predicts (independently)
  Round 2: Agent A sees B's reasoning, can change mind
           Agent B sees A's reasoning, can change mind
  Round 3: If disagree → Aggregator decides
           If converged → use consensus

Tests whether debate improves accuracy over single agent.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.base_agent import BaseAgent, AgentConfig, AgentResponse, extract_json_from_response
from src.profiles.patient_profile import PatientProfile
from configs.config import DEFAULT_MODEL


@dataclass
class RoundResult:
    """Result from a single debate round."""
    round_num: int
    agent_id: str
    delay: int
    forgo: int
    reasoning: str
    raw_response: str
    changed_mind: bool = False


@dataclass
class DebateResult:
    """Full debate output across all rounds."""
    
    # Round 1 (independent)
    r1_agent_a: RoundResult
    r1_agent_b: RoundResult
    
    # Round 2 (after seeing other's reasoning)
    r2_agent_a: RoundResult
    r2_agent_b: RoundResult
    
    # Final decision
    final_delay: int
    final_forgo: int
    final_reasoning: str
    final_method: str  # "consensus_r1", "consensus_r2", "aggregator"
    
    # Aggregator (only if needed)
    aggregator_result: Optional[RoundResult] = None
    
    # Metadata
    rounds_needed: int = 0
    a_changed_mind: bool = False
    b_changed_mind: bool = False
    
    # Full history
    history: List[RoundResult] = field(default_factory=list)
    
    def agreement_at_round(self, round_num: int) -> dict:
        """Check agreement at a specific round."""
        if round_num == 1:
            a, b = self.r1_agent_a, self.r1_agent_b
        elif round_num == 2:
            a, b = self.r2_agent_a, self.r2_agent_b
        else:
            return {"delay": False, "forgo": False}
        
        return {
            "delay": a.delay == b.delay,
            "forgo": a.forgo == b.forgo,
            "full": a.delay == b.delay and a.forgo == b.forgo,
        }


class DebateParticipant(BaseAgent):
    """A single debate participant."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL, temperature: float = 0.7, agent_id: str = "A"):
        config = AgentConfig(
            model_name=model_name,
            temperature=temperature,
            max_tokens=512,
            system_prompt=""
        )
        super().__init__(config)
        self.agent_id = agent_id
    
    def build_round1_prompt(self, profile: PatientProfile) -> str:
        """Round 1: Independent prediction."""
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
    
    def build_round2_prompt(
        self,
        profile: PatientProfile,
        own_response: RoundResult,
        other_response: RoundResult
    ) -> str:
        """Round 2: Reconsider after seeing other's reasoning."""
        narrative = profile.to_narrative()
        
        prompt = f"""You are this person:

{narrative}

---

You previously answered a health survey about delaying or going without medical care due to cost.

Your previous answer:
- Delay: {"Yes" if own_response.delay == 1 else "No"}
- Forgo: {"Yes" if own_response.forgo == 1 else "No"}
- Your reasoning: {own_response.reasoning}

Another person with access to the same information answered differently:
- Delay: {"Yes" if other_response.delay == 1 else "No"}
- Forgo: {"Yes" if other_response.forgo == 1 else "No"}
- Their reasoning: {other_response.reasoning}

Reconsider your answer. You may keep your original answer or change it. Think carefully about both perspectives.

Respond in this JSON format:

{{
  "delay": <0 for No or 1 for Yes>,
  "forgo": <0 for No or 1 for Yes>,
  "reasoning": "<explain your final answer, noting if and why you changed your mind>"
}}"""
        
        return prompt
    
    def parse_response(self, raw_response: str) -> AgentResponse:
        """Parse response with nested JSON handling."""
        parsed = extract_json_from_response(raw_response)
        
        if parsed:
            delay = self._normalize_binary(parsed.get("delay", 0))
            forgo = self._normalize_binary(parsed.get("forgo", 0))
            reasoning = parsed.get("reasoning", "")
            
            if isinstance(reasoning, dict):
                delay = self._normalize_binary(reasoning.get("delay", delay))
                forgo = self._normalize_binary(reasoning.get("forgo", forgo))
                reasoning = reasoning.get("reasoning", str(reasoning))
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
        
        return delay, forgo, response[:300]
    
    def predict_round1(self, profile: PatientProfile) -> RoundResult:
        """Round 1 prediction."""
        prompt = self.build_round1_prompt(profile)
        raw = self.call_llm(prompt)
        response = self.parse_response(raw)
        
        return RoundResult(
            round_num=1,
            agent_id=self.agent_id,
            delay=response.delay,
            forgo=response.forgo,
            reasoning=response.reasoning,
            raw_response=raw,
        )
    
    def predict_round2(
        self,
        profile: PatientProfile,
        own_r1: RoundResult,
        other_r1: RoundResult
    ) -> RoundResult:
        """Round 2 prediction after seeing other's reasoning."""
        prompt = self.build_round2_prompt(profile, own_r1, other_r1)
        raw = self.call_llm(prompt)
        response = self.parse_response(raw)
        
        changed = (response.delay != own_r1.delay) or (response.forgo != own_r1.forgo)
        
        return RoundResult(
            round_num=2,
            agent_id=self.agent_id,
            delay=response.delay,
            forgo=response.forgo,
            reasoning=response.reasoning,
            raw_response=raw,
            changed_mind=changed,
        )


class Aggregator(BaseAgent):
    """Final decision maker when agents can't agree."""
    
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
        r2_a: RoundResult,
        r2_b: RoundResult
    ) -> str:
        """Build aggregator prompt from round 2 results."""
        narrative = profile.to_narrative()
        
        prompt = f"""You are this person:

{narrative}

---

Two survey responses were given on your behalf about whether you delayed or went without medical care due to cost. After discussion, they still disagree.

Response A (after reconsideration):
- Delay: {"Yes" if r2_a.delay == 1 else "No"}
- Forgo: {"Yes" if r2_a.forgo == 1 else "No"}
- Reasoning: {r2_a.reasoning}

Response B (after reconsideration):
- Delay: {"Yes" if r2_b.delay == 1 else "No"}
- Forgo: {"Yes" if r2_b.forgo == 1 else "No"}
- Reasoning: {r2_b.reasoning}

Based on your actual circumstances, give your final answer.

{{
  "delay": <0 for No or 1 for Yes>,
  "forgo": <0 for No or 1 for Yes>,
  "reasoning": "<explain which response better represents your situation and why>"
}}"""
        
        return prompt
    
    def parse_response(self, raw_response: str) -> AgentResponse:
        parsed = extract_json_from_response(raw_response)
        
        if parsed:
            delay = self._normalize_binary(parsed.get("delay", 0))
            forgo = self._normalize_binary(parsed.get("forgo", 0))
            reasoning = parsed.get("reasoning", "")
            
            if isinstance(reasoning, dict):
                delay = self._normalize_binary(reasoning.get("delay", delay))
                forgo = self._normalize_binary(reasoning.get("forgo", forgo))
                reasoning = reasoning.get("reasoning", str(reasoning))
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
        
        return delay, forgo, response[:300]
    
    def decide(
        self,
        profile: PatientProfile,
        r2_a: RoundResult,
        r2_b: RoundResult
    ) -> RoundResult:
        prompt = self.build_prompt(profile, r2_a, r2_b)
        raw = self.call_llm(prompt)
        response = self.parse_response(raw)
        
        return RoundResult(
            round_num=3,
            agent_id="Aggregator",
            delay=response.delay,
            forgo=response.forgo,
            reasoning=response.reasoning,
            raw_response=raw,
        )


class DebateAgent:
    """
    Orchestrates true multi-round debate.
    
    Round 1: Independent predictions
    Round 2: Each sees other's reasoning, can change mind
    Round 3: Aggregator (only if still disagree)
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temp_a: float = 0.5,
        temp_b: float = 0.9,
        temp_aggregator: float = 0.3,
    ):
        self.agent_a = DebateParticipant(model_name=model_name, temperature=temp_a, agent_id="A")
        self.agent_b = DebateParticipant(model_name=model_name, temperature=temp_b, agent_id="B")
        self.aggregator = Aggregator(model_name=model_name, temperature=temp_aggregator)
        self.model_name = model_name
    
    def predict(self, profile: PatientProfile) -> DebateResult:
        """
        Run full multi-round debate.
        
        Round 1: Independent predictions (2 LLM calls)
        Round 2: Reconsider after seeing other (2 LLM calls)
        Round 3: Aggregator if needed (0-1 LLM calls)
        
        Total: 4-5 LLM calls per patient
        """
        history = []
        
        # === ROUND 1: Independent predictions ===
        r1_a = self.agent_a.predict_round1(profile)
        r1_b = self.agent_b.predict_round1(profile)
        history.extend([r1_a, r1_b])
        
        # Check round 1 agreement
        r1_agreed = (r1_a.delay == r1_b.delay) and (r1_a.forgo == r1_b.forgo)
        
        if r1_agreed:
            # Early consensus
            return DebateResult(
                r1_agent_a=r1_a,
                r1_agent_b=r1_b,
                r2_agent_a=r1_a,  # No round 2 needed
                r2_agent_b=r1_b,
                final_delay=r1_a.delay,
                final_forgo=r1_a.forgo,
                final_reasoning=f"Round 1 consensus. A: {r1_a.reasoning}",
                final_method="consensus_r1",
                rounds_needed=1,
                a_changed_mind=False,
                b_changed_mind=False,
                history=history,
            )
        
        # === ROUND 2: Reconsider after seeing other's reasoning ===
        r2_a = self.agent_a.predict_round2(profile, r1_a, r1_b)
        r2_b = self.agent_b.predict_round2(profile, r1_b, r1_a)
        history.extend([r2_a, r2_b])
        
        # Check round 2 agreement
        r2_agreed = (r2_a.delay == r2_b.delay) and (r2_a.forgo == r2_b.forgo)
        
        if r2_agreed:
            return DebateResult(
                r1_agent_a=r1_a,
                r1_agent_b=r1_b,
                r2_agent_a=r2_a,
                r2_agent_b=r2_b,
                final_delay=r2_a.delay,
                final_forgo=r2_a.forgo,
                final_reasoning=f"Round 2 consensus. A: {r2_a.reasoning}",
                final_method="consensus_r2",
                rounds_needed=2,
                a_changed_mind=r2_a.changed_mind,
                b_changed_mind=r2_b.changed_mind,
                history=history,
            )
        
        # === ROUND 3: Aggregator decides ===
        agg = self._majority_vote(r1_a, r1_b, r2_a, r2_b)
        history.append(agg)
        
        return DebateResult(
            r1_agent_a=r1_a,
            r1_agent_b=r1_b,
            r2_agent_a=r2_a,
            r2_agent_b=r2_b,
            final_delay=agg.delay,
            final_forgo=agg.forgo,
            final_reasoning=agg.reasoning,
            final_method="aggregator",
            aggregator_result=agg,
            rounds_needed=3,
            a_changed_mind=r2_a.changed_mind,
            b_changed_mind=r2_b.changed_mind,
            history=history,
        )
    
    def __repr__(self) -> str:
        return (
            f"DebateAgent(model={self.model_name}, "
            f"temp_a={self.agent_a.config.temperature}, "
            f"temp_b={self.agent_b.config.temperature}, "
            f"temp_agg={self.aggregator.config.temperature})"
        )
    def _majority_vote(self, r1_a: RoundResult, r1_b: RoundResult, r2_a: RoundResult, r2_b: RoundResult) -> RoundResult:
        """
        Recency-weighted majority vote.
        
        Priority: R2 consensus > majority vote > R2_A (conservative tiebreak)
        """
        delay_votes = [r1_a.delay, r1_b.delay, r2_a.delay, r2_b.delay]
        forgo_votes = [r1_a.forgo, r1_b.forgo, r2_a.forgo, r2_b.forgo]
        
        delay_sum = sum(delay_votes)
        forgo_sum = sum(forgo_votes)
        
        # Majority (3+ out of 4)
        if delay_sum >= 3:
            final_delay = 1
        elif delay_sum <= 1:
            final_delay = 0
        else:
            # Tied 2-2: trust R2_A (more conservative)
            final_delay = r2_a.delay
        
        if forgo_sum >= 3:
            final_forgo = 1
        elif forgo_sum <= 1:
            final_forgo = 0
        else:
            final_forgo = r2_a.forgo
        
        reasoning = (
            f"Majority vote: delay={delay_sum}/4, forgo={forgo_sum}/4. "
            f"R2_A: delay={r2_a.delay}, forgo={r2_a.forgo}. "
            f"R2_B: delay={r2_b.delay}, forgo={r2_b.forgo}."
        )
        
        return RoundResult(
            round_num=3,
            agent_id="MajorityVote",
            delay=final_delay,
            forgo=final_forgo,
            reasoning=reasoning,
            raw_response="N/A - deterministic vote",
        )

if __name__ == "__main__":
    import pandas as pd
    from configs.config import PROCESSED_DATA_DIR
    from src.profiles.patient_profile import create_patient_profile, extract_ground_truth
    from src.agents.base_agent import test_ollama_connection
    
    print("=" * 70)
    print("TRUE DEBATE AGENT TEST (MULTI-ROUND)")
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
    
    # Initialize
    debate = DebateAgent()
    print(f"Agent: {debate}\n")
    
    # Test cases
    # with_barrier = df[df["any_barrier"] == 1].iloc[0]
    # without_barrier = df[df["any_barrier"] == 0].iloc[0]
    
    # test_cases = [
    #     ("With barrier", with_barrier),
    #     ("Without barrier", without_barrier),
    # ]
    
    
        # Test cases that are more likely to produce disagreement
    # 1. With barrier - find low income or uninsured
    barrier_cases = df[df["any_barrier"] == 1]
    hard_barrier = barrier_cases[
        (barrier_cases["insurance_coverage"] == 3) |  # Uninsured
        (barrier_cases["poverty_category"] <= 2)       # Poor/Near poor
    ]
    if len(hard_barrier) > 0:
        with_barrier = hard_barrier.iloc[0]
    else:
        with_barrier = barrier_cases.iloc[0]
    
    # 2. Without barrier - find someone who "should" have barriers but doesn't
    no_barrier_cases = df[df["any_barrier"] == 0]
    tricky_no_barrier = no_barrier_cases[
        (no_barrier_cases["insurance_coverage"] == 3) |  # Uninsured
        (no_barrier_cases["poverty_category"] <= 2)       # Poor/Near poor
    ]
    if len(tricky_no_barrier) > 0:
        without_barrier = tricky_no_barrier.iloc[0]
    else:
        without_barrier = no_barrier_cases.iloc[0]
    
    test_cases = [
        ("With barrier (low income/uninsured)", with_barrier),
        ("Without barrier (low income/uninsured)", without_barrier),
    ]
    
    for label, row in test_cases:
        profile = create_patient_profile(row)
        truth = extract_ground_truth(row)
        
        print(f"{'=' * 70}")
        print(f"TEST: {label}")
        print(f"{'=' * 70}")
        print(f"Age: {profile.age}, Income: ${profile.family_income:,.0f}, Insurance: {profile.insurance_status}")
        print(f"Ground Truth — Delay: {truth.delayed_care}, Forgo: {truth.forgone_care}")
        
        result = debate.predict(profile)
        
        # Round 1
        print(f"\n--- ROUND 1 (Independent) ---")
        print(f"Agent A: delay={result.r1_agent_a.delay}, forgo={result.r1_agent_a.forgo}")
        print(f"  Reasoning: {result.r1_agent_a.reasoning}")
        print(f"Agent B: delay={result.r1_agent_b.delay}, forgo={result.r1_agent_b.forgo}")
        print(f"  Reasoning: {result.r1_agent_b.reasoning}")
        print(f"R1 Agreement: {result.agreement_at_round(1)}")
        
        # Round 2 (only if needed)
        if result.rounds_needed >= 2:
            print(f"\n--- ROUND 2 (After seeing other's reasoning) ---")
            print(f"Agent A: delay={result.r2_agent_a.delay}, forgo={result.r2_agent_a.forgo} {'(CHANGED)' if result.a_changed_mind else '(unchanged)'}")
            print(f"  Reasoning: {result.r2_agent_a.reasoning}")
            print(f"Agent B: delay={result.r2_agent_b.delay}, forgo={result.r2_agent_b.forgo} {'(CHANGED)' if result.b_changed_mind else '(unchanged)'}")
            print(f"  Reasoning: {result.r2_agent_b.reasoning}")
            print(f"R2 Agreement: {result.agreement_at_round(2)}")
        
        # Aggregator (only if needed)
        if result.rounds_needed >= 3 and result.aggregator_result:
            print(f"\n--- ROUND 3 (Aggregator) ---")
            print(f"Decision: delay={result.aggregator_result.delay}, forgo={result.aggregator_result.forgo}")
            print(f"Reasoning: {result.aggregator_result.reasoning}")
        
        # Final
        print(f"\n--- FINAL RESULT ---")
        print(f"Method: {result.final_method}")
        print(f"Rounds needed: {result.rounds_needed}")
        print(f"Delay: {result.final_delay}, Forgo: {result.final_forgo}")
        print(f"Reasoning: {result.final_reasoning}")
        
        # Accuracy
        delay_correct = result.final_delay == truth.delayed_care
        forgo_correct = result.final_forgo == truth.forgone_care
        print(f"\n--- Accuracy ---")
        print(f"Delay: {'✓' if delay_correct else '✗'}")
        print(f"Forgo: {'✓' if forgo_correct else '✗'}")
        print()