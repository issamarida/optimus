from meeting_intel import analyze_meeting, load_transcripts
from meeting_intel.ami import load_dialogue_acts
from meeting_intel.config import SplitConfig, TARGET_DA
from meeting_intel.data import grouped_train_test_split
from meeting_intel.features import build_segment_features
from meeting_intel.training import fit_da

AMI_ROOT = "data/ami_public_auto_1.5.1"

df = build_segment_features(load_dialogue_acts(AMI_ROOT))
split = SplitConfig()
train, test = grouped_train_test_split(df, split.test_size, split.seed)
model = fit_da(train)
print("dialogue-act predictions:", model.predict(test.head())[:5])

meeting = load_transcripts("synthetic", n_meetings=10, seed=7)[0]
analysis = analyze_meeting(meeting)
print("summary:", analysis.summary.summary)
for note in analysis.coaching.notes:
    print(f"  coaching@turn{note.evidence_turn}: {note.suggestion}")
