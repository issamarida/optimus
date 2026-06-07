from meeting_intel import analyze_meeting, build_feature_frame, fit_category, load_transcripts
from meeting_intel.config import MEETING_OBJ
from meeting_intel.data import grouped_train_test_split

frame = build_feature_frame(load_transcripts("synthetic", n_meetings=200, seed=7))
train, test = grouped_train_test_split(frame, test_size=0.25, seed=7)
model = fit_category(train)

analysis = analyze_meeting(test[MEETING_OBJ].iloc[0], model)
print("category:", analysis.category)
print("negative interaction flagged:", analysis.flag_negative_interaction)
print("summary:", analysis.summary.summary)
print("action items:", analysis.summary.action_items)
for note in analysis.coaching.notes:
    print(f"  coaching@turn{note.evidence_turn}: {note.suggestion}")
