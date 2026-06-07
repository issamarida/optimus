from meeting_intel import coach_meeting, load_transcripts, summarize_meeting
from meeting_intel.coaching import assert_grounded
from meeting_intel.schema import Meeting, Utterance, ROLE_CLIENT, ROLE_INTERNAL


def test_summary_is_nonempty():
    m = load_transcripts("synthetic", n_meetings=10, seed=4)[0]
    assert summarize_meeting(m).summary.strip()


def test_coaching_finds_planted_unanswered_question():
    m = Meeting("T1", [
        Utterance("I0", "welcome everyone", ROLE_INTERNAL, 0),
        Utterance("C0", "what is the delivery date?", ROLE_CLIENT, 1),
    ])
    assert "unanswered_question" in {n.kind for n in coach_meeting(m).notes}


def test_every_coaching_note_is_grounded():
    for m in load_transcripts("synthetic", n_meetings=30, seed=5):
        report = coach_meeting(m)
        assert_grounded(report)
        assert all(n.evidence_turn >= 0 for n in report.notes)
