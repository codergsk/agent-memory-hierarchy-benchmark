"""Customer Support scenario: multi-session support relationship.

Tests cross-session continuity: a customer contacts support three times
about an ongoing laptop issue. Later sessions probe whether the agent
remembers the device, the history, and the customer's constraints without
being re-told. Probe turns carry expected keywords (for deterministic
keyword-hit scoring) and memory facts (for LLM-as-judge personalization
scoring).
"""

CUSTOMER_SUPPORT = {
    "name": "Customer Support",
    "max_tokens": 512,
    "sessions": [
        # Session 1: establish facts (device, order, constraint, preference)
        [
            {"user": "Hi, I'm Dana. My Nimbus X15 laptop (order #88231) has "
                     "a screen that flickers whenever it runs on battery. "
                     "I bought it 3 weeks ago."},
            {"user": "I work as a video editor, so color accuracy matters "
                     "a lot to me. I can't be without a machine for more "
                     "than 2 business days."},
            {"user": "I already tried updating the display drivers and the "
                     "BIOS - neither fixed the flicker."},
            {"user": "OK, I'll run the panel self-test you suggested tonight "
                     "and get back to you."},
        ],
        # Session 2: continuation; probes for recall of device + history
        [
            {"user": "Hi, it's Dana again. The self-test showed no errors, "
                     "but the flicker is still happening on battery."},
            {"user": "What have we already ruled out so far for my issue?",
             "probe": True,
             "expected_keywords": ["driver", "BIOS", "self-test"],
             "memory_facts": [
                 "Dana owns a Nimbus X15 laptop, order #88231",
                 "The screen flickers on battery power",
                 "Driver update, BIOS update, and panel self-test were "
                 "already tried and did not fix it",
             ]},
            {"user": "If you need to repair it, remember my work constraint. "
                     "What repair option fits me best?",
             "probe": True,
             "expected_keywords": ["2 business days", "video editor"],
             "memory_facts": [
                 "Dana is a video editor who needs color accuracy",
                 "Dana cannot be without a machine for more than 2 "
                 "business days",
             ]},
        ],
        # Session 3: resolution; probe for full-relationship recall
        [
            {"user": "Hi, Dana here. The replacement panel arrived and was "
                     "installed, and the flicker is gone. Thanks!"},
            {"user": "Can you summarize my whole case for my records?",
             "probe": True,
             "expected_keywords": ["Nimbus X15", "flicker", "battery",
                                    "panel"],
             "memory_facts": [
                 "Dana's Nimbus X15 (order #88231) flickered on battery",
                 "Driver, BIOS, and self-test steps did not resolve it",
                 "A replacement panel resolved the issue",
             ]},
        ],
    ],
}
