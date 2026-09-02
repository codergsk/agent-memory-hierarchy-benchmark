"""Scenario registry: exports ALL_SCENARIOS for the benchmark harness.

Each scenario is a dict with:
    name           - display name used in reports
    max_tokens     - response budget (research scenario is long-form)
    sessions       - list of sessions; each session is a list of turns
Turn dicts contain:
    user               - the user message
    probe (optional)   - True if this turn is quality-evaluated
    expected_keywords  - keywords the reply should contain (hit-rate metric)
    memory_facts       - facts the agent should have remembered (judge input)
"""
from scenarios.customer_support import CUSTOMER_SUPPORT
from scenarios.personal_assistant import PERSONAL_ASSISTANT
from scenarios.multi_topic_research import MULTI_TOPIC_RESEARCH

ALL_SCENARIOS = [CUSTOMER_SUPPORT, MULTI_TOPIC_RESEARCH, PERSONAL_ASSISTANT]
