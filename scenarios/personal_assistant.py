"""Personal Assistant scenario: preferences accumulate and evolve.

Tests preference memory and evolution across sessions: the user teaches
the assistant dietary constraints, schedule habits, and family details,
then CHANGES one preference (vegetarian -> pescatarian), testing whether
the agent honors the newest state instead of stale facts (contradiction
handling). Probes check personalized recommendations without re-telling.
"""

PERSONAL_ASSISTANT = {
    "name": "Personal Assistant",
    "max_tokens": 512,
    "sessions": [
        # Session 1: establish identity, diet, schedule, family facts
        [
            {"user": "Hey, I'm Marco. Quick facts for you: I'm vegetarian, "
                     "I do school pickup at 3pm on weekdays, and my wife "
                     "Elena is allergic to peanuts."},
            {"user": "I like to gym at 6am, and I hate morning meetings "
                     "before 10am for that reason."},
            {"user": "Book club is every second Thursday evening. Keep "
                     "those free."},
            {"user": "Great. Talk tomorrow."},
        ],
        # Session 2: probes + one preference EVOLUTION
        [
            {"user": "Plan a dinner menu for Friday with my in-laws.",
             "probe": True,
             "expected_keywords": ["vegetarian", "peanut"],
             "memory_facts": [
                 "Marco is vegetarian",
                 "Marco's wife Elena is allergic to peanuts",
             ]},
            {"user": "When can you schedule a 1-hour call with my "
                     "accountant next Tuesday?",
             "probe": True,
             "expected_keywords": ["10", "3"],
             "memory_facts": [
                 "Marco does school pickup at 3pm on weekdays",
                 "Marco dislikes meetings before 10am (6am gym)",
             ]},
            {"user": "Update: I've started eating fish, so I'm pescatarian "
                     "now, not strictly vegetarian."},
        ],
        # Session 3: probes must reflect the EVOLVED preference
        [
            {"user": "Suggest a restaurant for date night with Elena.",
             "probe": True,
             "expected_keywords": ["pescatarian", "peanut"],
             "memory_facts": [
                 "Marco is now pescatarian (updated from vegetarian)",
                 "Elena is allergic to peanuts",
             ]},
            {"user": "What standing commitments should I never book over?",
             "probe": True,
             "expected_keywords": ["3pm", "book club", "gym"],
             "memory_facts": [
                 "School pickup at 3pm on weekdays",
                 "Gym at 6am; no meetings before 10am",
                 "Book club every second Thursday evening",
             ]},
        ],
    ],
}
