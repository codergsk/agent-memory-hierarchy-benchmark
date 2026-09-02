"""Shared stale-memory noise preload.

Simulates an agent that has been in service for months: 24 old, resolved,
or irrelevant memory items seeded into every persistent memory store
(NaiveVectorAgent and AMHAgent receive the IDENTICAL list) before each
scenario begins. This operationalizes the whitepaper's scale-testing
requirement - the failure mode of append-only stores is retrieval noise
that grows with time, which a 12-turn conversation alone cannot exhibit.

Items are deliberately adjacent-but-wrong: same user, plausible past
threads, some superficially similar to scenario topics (old laptop, old
diets, old research), so similarity-only retrieval is genuinely tempted.
"""

NOISE_PRELOAD = [
    "User asked about upgrading RAM in their old Vertex Z3 desktop; resolved in March",
    "User's printer showed error E502; fixed by replacing the toner cartridge",
    "User once asked for a gluten-free pancake recipe for a weekend brunch",
    "User was comparing pet insurance quotes for their dog Biscuit last year",
    "User asked how to export contacts from an old Nokia phone; completed",
    "User's previous laptop, a Vertex B2, had a battery swelling issue; device was recycled",
    "User planned a camping trip to Yosemite two summers ago; trip completed",
    "User asked about the difference between index funds and ETFs",
    "User temporarily followed a keto diet for six weeks; stopped it afterwards",
    "User requested a summary of a documentary about deep-sea creatures",
    "User asked how to remove a red wine stain from a wool carpet",
    "User's old order #55102 for a monitor stand was delivered without issues",
    "User researched Spanish language classes but decided to postpone",
    "User asked whether to repair or replace a 9-year-old dishwasher; replaced it",
    "User wanted gift ideas for a colleague's retirement party last spring",
    "User asked about lithium-ion battery recycling drop-off points downtown",
    "User compared three noise-cancelling headphones and bought the mid-range pair",
    "User asked for the rules of pickleball before a company tournament",
    "User's old research thread on solar panel payback periods concluded in June",
    "User asked how to set an out-of-office reply in their mail client",
    "User inquired about a coffee subscription service and cancelled the trial",
    "User asked for stretches to relieve wrist strain from typing",
    "User's Wi-Fi dead-zone issue was solved with a mesh router in the hallway",
    "User once drafted a business idea for a dog-walking app; shelved it",
]
