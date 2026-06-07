"""Minimal programmatic example. Run: python examples/quickstart.py

Shows the intended usage: train the category model on a TRAIN split, then analyse
held-out meetings. Swap load_transcripts to real data and pass llm=get_llm("auto")
for abstractive summaries + LLM-phrased coaching.
"""
from meeting_intel import CategoryClassifier, analyze_meeting, load_transcripts
from meeting_intel.data import grouped_train_test_split, meetings_to_frame

meetings = load_transcripts("synthetic", n_meetings=200, seed=7)
df = meetings_to_frame(meetings)
train, test = grouped_train_test_split(df, test_size=0.25, seed=7)

model = CategoryClassifier().fit(train["full_text"], train["_category"])

analysis = analyze_meeting(test["_meeting_obj"].iloc[0], model)
print("category:", analysis.category)
print("negative interaction flagged:", analysis.flag_negative_interaction)
print("summary:", analysis.summary.summary)
print("action items:", analysis.summary.action_items)
for note in analysis.coaching.notes:
    print(f"  coaching@turn{note.evidence_turn}: {note.suggestion}")
