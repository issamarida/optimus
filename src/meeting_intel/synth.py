"""Seeded synthetic corpus for a runnable, network-free demo.

Generates meetings with discussion-category structure, noisy client sentiment,
and DELIBERATELY PLANTED coachable moments (an unanswered client question; a
negative client turn nobody acknowledges) so the coaching detectors have real
signal to find. Numbers on this data validate the machinery, not the method.
"""
from __future__ import annotations

import random

from .schema import Meeting, Utterance, ROLE_CLIENT, ROLE_INTERNAL

CATEGORIES = ("status_update", "negotiation", "technical_review", "kickoff", "escalation")

_PHRASES = {
    "status_update": ["here's where we are this week", "the milestone slipped slightly",
                      "on track for the deadline", "progress since last sync", "open action items"],
    "negotiation": ["the pricing needs revisiting", "what discount can you offer",
                    "the contract terms", "we need a better rate", "scope versus budget"],
    "technical_review": ["the latency in the api", "we refactored the data layer",
                         "the integration test failed", "throughput under load", "the schema change"],
    "kickoff": ["welcome everyone to the project", "let's align on goals",
                "introductions around the room", "the roadmap for this quarter", "success criteria"],
    "escalation": ["this is a serious problem", "the outage affected production",
                  "we are very unhappy with this", "this needs leadership attention", "unacceptable delay"],
}
_FILLER = ["thanks everyone for joining", "let me share my screen", "let's move to the next item",
           "i'll follow up by email", "any questions on that", "let's circle back next week",
           "noted, thank you", "appreciate the time today", "let me pull up the document"]
_ACK = ["i understand your concern", "sorry about that, we'll fix it",
        "let me address that directly", "good question, here's the answer", "we'll sort that right away"]
_POS = ["that works for us", "this looks good", "happy with the progress so far",
        "okay that's fine i think", "reasonable, we can live with that"]
_NEG = ["this isn't really working for us", "we're not thrilled with the delays",
        "that's a bit of a problem", "honestly a little disappointed", "this can't happen again"]
_NEU = ["okay, understood", "can you clarify the timeline", "what's the next step",
        "send the details over", "i'll need to check internally"]


def generate_corpus(n_meetings: int = 240, seed: int = 7) -> list[Meeting]:
    rng = random.Random(seed)
    meetings: list[Meeting] = []
    for i in range(n_meetings):
        category = rng.choice(CATEGORIES)
        if category == "escalation":
            sentiment = rng.choices(["negative", "neutral", "positive"], weights=[6, 3, 1])[0]
        elif category == "kickoff":
            sentiment = rng.choices(["positive", "neutral", "negative"], weights=[5, 4, 1])[0]
        else:
            sentiment = rng.choices(["neutral", "positive", "negative"], weights=[5, 3, 2])[0]

        n_turns = rng.randint(16, 40)
        secondary = rng.choice([c for c in CATEGORIES if c != category]) if rng.random() < 0.3 else None
        plant_unanswered_q = rng.random() < 0.5
        plant_unack_negative = rng.random() < 0.5

        utts: list[Utterance] = []
        t = 0
        while t < n_turns:
            is_client = rng.random() < 0.4
            if is_client:
                if rng.random() < 0.25:
                    txt, role = rng.choice(_NEU), ROLE_CLIENT
                else:
                    bank = {"positive": _POS, "negative": _NEG, "neutral": _NEU}[sentiment]
                    txt, role = rng.choice(bank), ROLE_CLIENT
            else:
                roll = rng.random()
                if roll < 0.45:
                    txt = rng.choice(_PHRASES[category])
                elif secondary and roll < 0.6:
                    txt = rng.choice(_PHRASES[secondary])
                else:
                    txt = rng.choice(_FILLER)
                role = ROLE_INTERNAL
            utts.append(Utterance(speaker=("C" if role == ROLE_CLIENT else "I") + str(t % 3),
                                   text=txt, role=role, turn_index=t))
            t += 1

        # Plant an unanswered client question near the middle (client "?" followed
        # by internal FILLER, never an acknowledgement).
        if plant_unanswered_q and len(utts) > 6:
            j = len(utts) // 2
            utts.insert(j, Utterance("C9", "can you confirm the delivery date?", ROLE_CLIENT, j))
            utts.insert(j + 1, Utterance("I9", rng.choice(_FILLER), ROLE_INTERNAL, j + 1))

        # Plant an unacknowledged negative: negative client turn followed by filler.
        if plant_unack_negative and len(utts) > 8:
            k = max(2, len(utts) // 3)
            utts.insert(k, Utterance("C8", "we're really frustrated with this delay", ROLE_CLIENT, k))
            utts.insert(k + 1, Utterance("I8", rng.choice(_FILLER), ROLE_INTERNAL, k + 1))

        for idx, u in enumerate(utts):
            u.turn_index = idx

        if rng.random() < 0.15:  # noisy meeting-level gold
            sentiment = rng.choice(["positive", "neutral", "negative"])

        meetings.append(Meeting(meeting_id=f"M{i:04d}", utterances=utts,
                                category=category, client_sentiment=sentiment))
    return meetings
