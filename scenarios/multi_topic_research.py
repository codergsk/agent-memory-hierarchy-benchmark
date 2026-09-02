"""Multi-Topic Research scenario: long-form work across three domains.

Tests retrieval precision under topic interference: a researcher works
with the agent across three unrelated threads (battery chemistry, EU AI
regulation, coffee-shop business plan). Long-form answers make this the
token- and latency-heavy scenario. Probes require pulling facts from the
RIGHT thread while ignoring the other two - where naive vector retrieval
degrades and MRDF-scored retrieval should hold precision.
"""

MULTI_TOPIC_RESEARCH = {
    "name": "Multi-Topic Research",
    "max_tokens": 1500,  # long-form research responses
    "sessions": [
        # Session 1: thread A (batteries) + thread B (regulation) seeded
        [
            {"user": "I'm researching solid-state battery chemistry. Key "
                     "constraint from my lab: we can only work with "
                     "sulfide-based electrolytes, and our target is 400 "
                     "Wh/kg by 2028. Give me an overview of the main "
                     "technical hurdles."},
            {"user": "Separate thread: I'm also mapping EU AI Act "
                     "obligations for my company. We ship a high-risk HR "
                     "screening product in Germany. Summarize what "
                     "compliance milestones apply."},
            {"user": "Back to batteries: what are the top 3 research groups "
                     "publishing on sulfide electrolyte interfaces?"},
        ],
        # Session 2: thread C seeded + cross-thread probes
        [
            {"user": "New thread: I'm writing a business plan for a "
                     "specialty coffee shop in Lisbon, budget 80k euros, "
                     "targeting digital nomads. Outline the plan."},
            {"user": "For my battery research: given my lab's constraint, "
                     "which electrolyte class should we NOT invest time "
                     "in, and why?",
             "probe": True,
             "expected_keywords": ["sulfide", "oxide"],
             "memory_facts": [
                 "The lab can only work with sulfide-based electrolytes",
                 "Target is 400 Wh/kg by 2028",
             ]},
            {"user": "On the regulation thread: what's my product's risk "
                     "classification and the single most urgent milestone?",
             "probe": True,
             "expected_keywords": ["high-risk", "HR"],
             "memory_facts": [
                 "The company ships a high-risk HR screening product",
                 "It operates in Germany under the EU AI Act",
             ]},
        ],
        # Session 3: precision probes across all three threads
        [
            {"user": "Give me a combined status brief: one paragraph per "
                     "research thread I have open with you, with my "
                     "specific constraints in each.",
             "probe": True,
             "expected_keywords": ["sulfide", "400 Wh/kg", "high-risk",
                                    "Lisbon", "80k"],
             "memory_facts": [
                 "Battery thread: sulfide-only lab, 400 Wh/kg by 2028",
                 "Regulation thread: high-risk HR product in Germany",
                 "Coffee thread: Lisbon shop, 80k euro budget, digital "
                 "nomad audience",
             ]},
            {"user": "For the coffee shop: adjust the plan for my budget "
                     "and audience, and keep it consistent with what we "
                     "discussed.",
             "probe": True,
             "expected_keywords": ["80k", "nomad"],
             "memory_facts": [
                 "Budget is 80k euros",
                 "Target audience is digital nomads in Lisbon",
             ]},
        ],
    ],
}
