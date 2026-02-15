"""
Base agent with Ollama integration.
"""


from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import sys
import json
import httpx
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from configs.config import OLLAMA_HOST, DEFAULT_MODEL, DEFAULT_TEMPERATURE


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    model_name: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = 256
    system_prompt: str = ""


@dataclass
class AgentResponse:
    """Response from an agent."""
    delay: int
    forgo: int
    reasoning: str
    raw_response: str


def test_ollama_connection(model: str = DEFAULT_MODEL) -> bool:
    """Test if Ollama is running and model is available."""
    try:
        response = httpx.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": model, "prompt": "Hi", "stream": False},
            timeout=30.0
        )
        return response.status_code == 200
    except Exception as e:
        print(f"Ollama connection error: {e}")
        return False


def extract_json_from_response(text: str) -> Optional[dict]:
    """Extract JSON from LLM response."""
    
    # Try direct parse
    try:
        return json.loads(text)
    except:
        pass
    
    # Try to find JSON in text
    import re
    patterns = [
        r'\{[^{}]*\}',
        r'```json\s*(\{[^```]*\})\s*```',
        r'```\s*(\{[^```]*\})\s*```',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str)
            except:
                continue
    
    return None


class BaseAgent:
    """Base class for agents."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
    
    def call_llm(self, prompt: str) -> str:
        """Call Ollama API."""
        
        messages = []
        
        if self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            response = httpx.post(
                f"{OLLAMA_HOST}/api/chat",
                json={
                    "model": self.config.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": self.config.max_tokens,
                    }
                },
                timeout=60.0
            )
            
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                return f"Error: {response.status_code}"
                
        except Exception as e:
            return f"Error: {e}"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model_name})"


if __name__ == "__main__":
    print("=" * 70)
    print("BASE AGENT TEST")
    print("=" * 70)
    
    model = DEFAULT_MODEL
    print(f"\nTesting Ollama connection ({model})...")
    
    if test_ollama_connection(model):
        print("✓ Connection successful")
        
        agent = BaseAgent()
        response = agent.call_llm("Say 'Hello, World!' in JSON format: {\"message\": \"...\"}")
        print(f"\nResponse: {response}")
    else:
        print("✗ Connection failed. Is Ollama running?")