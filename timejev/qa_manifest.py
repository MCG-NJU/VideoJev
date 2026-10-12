"""Convert multiple-choice video QA into TimeJev's one-question decision rows."""

import csv
import hashlib
import json
import math
import re
from pathlib import Path


_OPTION = re.compile(r"(?m)^[ \t]*(?:\(([A-H])\)|([A-H])[.)])[ \t]+")
_VIDEO_PREFIX = re.compile(r"^<video>[ \t]*\r?\n")


def parse_choice(text, answer):
    """Return a question stem, labelled options, and a zero-based answer."""
    if not isinstance(text, str) or not isinstance(answer, str):
        raise ValueError("Question and answer must be text")
    text = _VIDEO_PREFIX.sub("", text, count=1)
    matches = list(_OPTION.finditer(text))
    labels = [match.group(1) or match.group(2) for match in matches]
    if not 2 <= len(labels) <= 8 or labels != [chr(65 + i) for i in range(len(labels))]:
        raise ValueError(f"Expected consecutive A-H options, got {labels}")
    query = text[: matches[0].start()].strip()
    contents = [text[match.end() : matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
                for i, match in enumerate(matches)]
    if not query or any(not option for option in contents):
        raise ValueError("Question stem and every option must be nonempty")
    if any("<|" in value for value in [query, *contents]):
        raise ValueError("Model control-token syntax in QA text")
    letter = answer.strip()
    if letter not in labels:
        raise ValueError(f"Answer {letter!r} is not one of {labels}")
    return query, [f"{label}. {content}" for label, content in zip(labels, contents)], labels.index(letter)


def training_rows(source):
    """Yield every labelled turn, retaining the same video for each question."""
    video_keys = {}
    row_ids = set()
    with Path(source).open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            source_id = item["id"]
            if source_id in row_ids:
                raise ValueError(f"Duplicate source id on line {line_number}")
            row_ids.add(source_id)
            paths, info = item["video"], item["video_info"]
            if len(paths) != 1 or len(info) != 1:
                raise ValueError(f"Expected one video on line {line_number}")
            path = str(Path(paths[0]).resolve(strict=True))
            duration = float(info[0]["length"])
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"Invalid video duration on line {line_number}")
            key = f"qa_train/{Path(path).stem}"
            previous = video_keys.setdefault(key, (path, duration))
            if previous != (path, duration):
                raise ValueError(f"Video key {key} has inconsistent path or duration")
            turns = item["conversations"]
            if not turns or len(turns) % 2:
                raise ValueError(f"Unpaired QA turns on line {line_number}")
            for turn in range(0, len(turns), 2):
                human, assistant = turns[turn : turn + 2]
                if human["from"] != "human" or assistant["from"] != "gpt":
                    raise ValueError(f"Invalid QA role order on line {line_number}")
                query, options, answer_index = parse_choice(human["value"], assistant["value"])
                yield dict(id=f"qa/{source_id}/{turn // 2}", task="choice", video_key=key,
                           video_path=path, duration=duration, query=query, options=options,
                           answer_index=answer_index, source="LVBench_QA_MC_multiturn",
                           source_id=source_id, turn_index=turn // 2, split="train")


def lvbench_rows(tsv, metadata, root):
    """Use the official UID/answer inventory and the completed 1549-row TSV."""
    root = Path(root)
    entries = [json.loads(line) for line in Path(metadata).read_text().splitlines() if line.strip()]
    by_video = {item["key"]: item for item in entries}
    if len(by_video) != len(entries):
        raise ValueError("Duplicate official LVBench video ID")
    expected = {(item["key"], str(qa["uid"])): qa for item in entries for qa in item["qa"]}
    seen = set()
    with Path(tsv).open(newline="") as stream:
        for record in csv.DictReader(stream, delimiter="\t"):
            key, uid = record["video"], record["uid"]
            pair = (key, uid)
            if pair in seen or pair not in expected:
                raise ValueError(f"Unexpected or repeated LVBench item {pair}")
            seen.add(pair)
            qa = expected[pair]
            if record["question"] != qa["question"] or record["answer"] != qa["answer"]:
                raise ValueError(f"LVBench TSV differs from official annotation at {pair}")
            query, options, answer_index = parse_choice(record["question"], record["answer"])
            path = (root / record["video_path"]).resolve(strict=True)
            duration = float(by_video[key]["video_info"]["duration_minutes"]) * 60
            yield dict(id=f"lvbench/{uid}", uid=uid, task="choice", video_key=f"lvbench/{key}",
                       video_path=str(path), duration=duration, query=query, options=options,
                       answer_index=answer_index, source="LVBench", split="test",
                       question_type=qa["question_type"])
    if seen != set(expected):
        raise ValueError(f"LVBench TSV is missing {len(set(expected) - seen)} official questions")


def manifest_bytes(rows):
    ids = set()
    output = []
    for row in rows:
        if row["id"] in ids:
            raise ValueError(f"Duplicate QA id {row['id']}")
        ids.add(row["id"])
        output.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return "".join(output).encode(), len(ids)


def write_manifest(path, data):
    """Keep existing manifests immutable unless they are byte-identical."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Existing manifest differs: {path}")
        return hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()
