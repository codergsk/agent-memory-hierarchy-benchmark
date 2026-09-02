"""LLM-as-judge for probe-turn quality and personalization scoring.

The judge receives the probe question, the agent's response, and the
memory facts the agent SHOULD have been able to use (scenario ground
truth). It scores two dimensions on 0-1:

    quality          - is the response correct, helpful, coherent?
    personalization  - does it correctly use the user-specific facts
                       (without the user re-stating them)?

Judged at temperature 0 with a strict JSON output contract.
"""
import json
from typing import Dict, List

JUDGE_PROMPT = """You are evaluating an AI assistant's response.

The user asked:
{question}

The assistant responded:
{response}

Ground truth - facts the assistant should have remembered about this user
from earlier conversations (the user did NOT restate them in the question):
{facts}

Score two dimensions from 0.0 to 1.0:
1. "quality": Is the response correct, helpful, specific, and coherent as
   an answer to the question?
2. "personalization": Does the response correctly incorporate the ground
   truth facts? Full credit only if the specific facts are reflected;
   0.0 if the response ignores them, asks the user to repeat them, or
   contradicts them.

Return ONLY a JSON object: {{"quality": <float>, "personalization": <float>,
"rationale": "<one sentence>"}}"""


class LLMJudge:
    """Scores probe responses using a Bedrock foundation model."""

    def __init__(self, bedrock_client, judge_model_id: str):
        self.bedrock = bedrock_client
        self.judge_model_id = judge_model_id
        self.total_cost = 0.0
        self.calls = 0

    def evaluate(self, question: str, response: str,
                 memory_facts: List[str]) -> Dict[str, float]:
        """Judge one probe response; returns quality/personalization dict."""
        prompt = JUDGE_PROMPT.format(
            question=question,
            response=response,
            facts="\n".join(f"- {f}" for f in memory_facts) or "- (none)")
        text, metrics = self.bedrock.generate_text(
            messages=[{"role": "user", "content": prompt}],
            model_id=self.judge_model_id, max_tokens=300, temperature=0.0)
        self.total_cost += self.bedrock.calculate_cost(
            metrics["input_tokens"], metrics["output_tokens"],
            self.judge_model_id)
        self.calls += 1
        # Tolerate code fences / prose around the JSON object.
        try:
            start, end = text.find("{"), text.rfind("}")
            payload = json.loads(text[start:end + 1])
            return {
                "quality": max(0.0, min(1.0, float(payload["quality"]))),
                "personalization": max(0.0, min(1.0,
                                       float(payload["personalization"]))),
            }
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # A malformed judgment scores conservatively rather than crashing.
            return {"quality": 0.0, "personalization": 0.0}
