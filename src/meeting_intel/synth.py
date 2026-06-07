from __future__ import annotations

import random

from . import config
from .schema import Meeting, Utterance, ROLE_CLIENT, ROLE_INTERNAL

CATEGORIES = ("status_update", "negotiation", "technical_review", "kickoff", "escalation")

PHRASES = {
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
FILLER = ["thanks everyone for joining", "let me share my screen", "let's move to the next item",
          "i'll follow up by email", "any questions on that", "let's circle back next week",
          "noted, thank you", "appreciate the time today", "let me pull up the document"]
POS = ["that works for us", "this looks good", "happy with the progress so far",
       "okay that's fine i think", "reasonable, we can live with that"]
NEG = ["this isn't really working for us", "we're not thrilled with the delays",
       "that's a bit of a problem", "honestly a little disappointed", "this can't happen again"]
NEU = ["okay, understood", "can you clarify the timeline", "what's the next step",
       "send the details over", "i'll need to check internally"]
SENTIMENT_BANK = {config.POSITIVE: POS, config.NEGATIVE: NEG, config.NEUTRAL: NEU}

CATEGORY_SENTIMENT_WEIGHTS = {
    "escalation": ([config.NEGATIVE, config.NEUTRAL, config.POSITIVE], [6, 3, 1]),
    "kickoff": ([config.POSITIVE, config.NEUTRAL, config.NEGATIVE], [5, 4, 1]),
}
DEFAULT_SENTIMENT_WEIGHTS = ([config.NEUTRAL, config.POSITIVE, config.NEGATIVE], [5, 3, 2])


def generate_corpus(n_meetings: int = config.CorpusConfig.n_meetings,
                    seed: int = config.CorpusConfig.seed) -> list[Meeting]:
    rng = random.Random(seed)
    meetings: list[Meeting] = []
    for i in range(n_meetings):
        category = rng.choice(CATEGORIES)
        labels, weights = CATEGORY_SENTIMENT_WEIGHTS.get(category, DEFAULT_SENTIMENT_WEIGHTS)
        sentiment = rng.choices(labels, weights=weights)[0]

        n_turns = rng.randint(16, 40)
        secondary = rng.choice([c for c in CATEGORIES if c != category]) if rng.random() < 0.3 else None
        utts: list[Utterance] = []
        for t in range(n_turns):
            if rng.random() < 0.4:
                bank = NEU if rng.random() < 0.25 else SENTIMENT_BANK[sentiment]
                utts.append(Utterance(f"C{t % 3}", rng.choice(bank), ROLE_CLIENT, t))
            else:
                roll = rng.random()
                if roll < 0.45:
                    text = rng.choice(PHRASES[category])
                elif secondary and roll < 0.6:
                    text = rng.choice(PHRASES[secondary])
                else:
                    text = rng.choice(FILLER)
                utts.append(Utterance(f"I{t % 3}", text, ROLE_INTERNAL, t))

        if rng.random() < 0.5 and len(utts) > 6:
            j = len(utts) // 2
            utts.insert(j, Utterance("C9", "can you confirm the delivery date?", ROLE_CLIENT, j))
            utts.insert(j + 1, Utterance("I9", rng.choice(FILLER), ROLE_INTERNAL, j + 1))
        if rng.random() < 0.5 and len(utts) > 8:
            k = max(2, len(utts) // 3)
            utts.insert(k, Utterance("C8", "we're really frustrated with this delay", ROLE_CLIENT, k))
            utts.insert(k + 1, Utterance("I8", rng.choice(FILLER), ROLE_INTERNAL, k + 1))
        for idx, u in enumerate(utts):
            u.turn_index = idx

        if rng.random() < 0.15:
            sentiment = rng.choice([config.POSITIVE, config.NEUTRAL, config.NEGATIVE])
        meetings.append(Meeting(f"M{i:04d}", utts, category=category, client_sentiment=sentiment))
    return meetings
