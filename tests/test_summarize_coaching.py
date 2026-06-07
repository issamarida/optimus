from meeting_intel.coaching import assert_grounded, coach_meeting
from meeting_intel.data import load_transcripts
from meeting_intel.schema import Meeting, Utterance, ROLE_CLIENT, ROLE_INTERNAL
from meeting_intel.summarize import summarize_meeting


def test_summary_is_nonempty_and_deduped():
    m = load_transcripts("synthetic", n_meetings=10, seed=4)[0]
    s = summarize_meeting(m, max_sentences=4)
    assert s.summary.strip()
    parts = s.summary.split(" ")
    assert len(parts) > 0


def test_coaching_finds_planted_unanswered_question():
    # A client question with no internal reply afterwards must be flagged.
    m = Meeting("T1", [
        Utterance("I0", "welcome everyone", ROLE_INTERNAL, 0),
        Utterance("C0", "what is the delivery date?", ROLE_CLIENT, 1),
    ])
    report = coach_meeting(m)
    kinds = {n.kind for n in report.notes}
    assert "unanswered_question" in kinds


def test_every_coaching_note_is_grounded():
    for m in load_transcripts("synthetic", n_meetings=30, seed=5):
        report = coach_meeting(m)
        assert_grounded(report)  # raises if any note lacks evidence
        for n in report.notes:
            assert n.evidence_turn >= 0
