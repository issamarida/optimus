from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

from . import config
from .schema import Meeting, Utterance

NITE = "{http://nite.sourceforge.net/}"
_ID_RE = re.compile(r"id\(([^)]+)\)")
_SERIES_RE = re.compile(r"^([A-Z]+\d+)")

DA_TYPES_REL = "ontologies/da-types.xml"
WORDS_REL = "words"
DA_REL = "dialogueActs"
ABS_REL = "abstractive"


def _series(meeting_id: str) -> str:
    m = _SERIES_RE.match(meeting_id)
    return m.group(1) if m else meeting_id


def _load_da_names(root: Path) -> dict[str, str]:
    tree = ET.parse(root / DA_TYPES_REL)
    return {node.get(f"{NITE}id"): node.get("name")
            for node in tree.iter("da-type")
            if node.get(f"{NITE}id", "").startswith("ami_da_") and node.get("name")}


def _load_words(path: Path):
    order, text, start = [], {}, {}
    for _event, elem in ET.iterparse(path):
        if elem.tag == "w" or elem.tag.endswith("}w"):
            wid = elem.get(f"{NITE}id")
            if wid is not None and elem.text:
                order.append(wid)
                text[wid] = elem.text.strip()
                st = elem.get("starttime")
                start[wid] = float(st) if st not in (None, "") else None
        elem.clear()
    pos = {wid: i for i, wid in enumerate(order)}
    return order, text, start, pos


def _resolve_child(href, order, text, start, pos):
    ids = _ID_RE.findall(href)
    if not ids:
        return "", None
    if len(ids) == 1:
        selected = [ids[0]] if ids[0] in pos else []
    else:
        a, b = ids[0], ids[-1]
        if a in pos and b in pos:
            i, j = sorted((pos[a], pos[b]))
            selected = order[i:j + 1]
        else:
            selected = []
    words = [text[w] for w in selected if w in text]
    starts = [start[w] for w in selected if start.get(w) is not None]
    return " ".join(words).strip(), (min(starts) if starts else None)


def _parse_da_file(path, da_names, order, text, start, pos) -> list[dict]:
    rows = []
    for dact in ET.parse(path).getroot().findall("dact"):
        pointer = dact.find(f"{NITE}pointer")
        label_id = None
        if pointer is not None:
            found = _ID_RE.search(pointer.get("href", ""))
            label_id = found.group(1) if found else None
        label = da_names.get(label_id)
        if label is None:
            continue
        spans, starts = [], []
        for child in dact.findall(f"{NITE}child"):
            span_text, span_start = _resolve_child(child.get("href", ""), order, text, start, pos)
            if span_text:
                spans.append(span_text)
                if span_start is not None:
                    starts.append(span_start)
        seg_text = " ".join(spans).strip()
        if seg_text:
            rows.append({config.SEG_TEXT: seg_text, config.TARGET_DA: label,
                         "starttime": min(starts) if starts else 0.0})
    return rows


def load_dialogue_acts(ami_root: str | Path = config.AMI_ROOT) -> pd.DataFrame:
    root = Path(ami_root)
    da_names = _load_da_names(root)
    rows: list[dict] = []
    for da_path in sorted((root / DA_REL).glob("*.dialog-act.xml")):
        stem = da_path.name.replace(".dialog-act.xml", "")
        meeting_id, _, speaker = stem.partition(".")
        words_path = root / WORDS_REL / f"{stem}.words.xml"
        if not words_path.exists():
            continue
        order, text, start, pos = _load_words(words_path)
        for row in _parse_da_file(da_path, da_names, order, text, start, pos):
            row.update({config.GROUP_COL: _series(meeting_id),
                        "meeting_id": meeting_id, "speaker": speaker})
            rows.append(row)
    return pd.DataFrame(rows)


def meetings_from_segments(df: pd.DataFrame) -> list[Meeting]:
    meetings = []
    for meeting_id, group in df.groupby("meeting_id"):
        ordered = group.sort_values("starttime")
        utterances = [
            Utterance(speaker=r.speaker, text=getattr(r, config.SEG_TEXT), turn_index=i)
            for i, r in enumerate(ordered.itertuples())
        ]
        meetings.append(Meeting(meeting_id=meeting_id, utterances=utterances))
    return meetings


def load_abstractive_references(ami_root: str | Path = config.AMI_ROOT) -> dict[str, str]:
    root = Path(ami_root) / ABS_REL
    refs: dict[str, str] = {}
    if not root.exists():
        return refs
    for path in sorted(root.glob("*.abssumm.xml")):
        meeting_id = path.name.split(".")[0]
        sentences = [s.text.strip() for s in ET.parse(path).iter("sentence")
                     if s.text and s.text.strip()]
        if sentences:
            refs[meeting_id] = " ".join(sentences)
    return refs
