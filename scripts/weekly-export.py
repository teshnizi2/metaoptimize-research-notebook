#!/usr/bin/env python3
"""Turn one weekly-record data file (markdown with `## Q | ...` and `## T | ...` blocks) into the site's JSON.

    python3 scripts/weekly-export.py <data.md> <out-folder> [--open]

Writes <out-folder>/<week id>.json (week id = the file name's date), refreshes <out-folder>/index.json and prints a short report. Nothing here is
written into the repository: the output is uploaded to the private Blob store (see docs/MAINTENANCE.md, "Weekly record").

The data file format is the one used for the local weekly pages:
  header lines   title: range: oneline: updated:  and  summary:  followed by 1 to 4 "- " lines (where the project stands)
                 achievements:  followed by "- " lines, one per achievement of the week
                 other:  followed by "- " lines, one per other piece of work (optional; exported as the week's tasks, titles only)
  blocks         ## Q | <title>  (a direction question)   or   ## T | <title>  (a task a chat did; older files)
  block fields   said: <when> | <words>  (Q, may repeat; may be absent for a question from the work)
                 asked: options: chosen: because: next:  (the direction-question names, exported as
                 question, decisions, answer, why and outcome; the old names still work)
                 table: (caption line, then a markdown table; may repeat)   chart: <name> | <caption>  (may repeat)
                 details: dtable: dchart:  (the closed "Details" part: longer text, tables and charts; dtable and dchart may repeat)
                 sources: <paths>  (read, never copied: file paths stay out of the site)

Privacy filter: sentences and table rows about money, usage limits and budgets, file paths and a few internal
words are dropped or reworded; blocks listed in DROP_BLOCKS are left out; charts in DROP_CHARTS are left out.
The patterns below are the whole list; the report prints every dropped line so a human can read it.
"""
import json, os, re, sys

ALIAS = {"asked": "question", "options": "decisions", "chosen": "answer", "because": "why", "next": "outcome"}
KEYS = ("said", "question", "answer", "why", "decisions", "outcome", "table", "chart", "details", "dtable", "dchart",
        "sources") + tuple(ALIAS)
REPEAT = ("said", "table", "chart", "dtable", "dchart")
HEAD_KEYS = ("title", "range", "oneline", "updated", "summary", "achievements", "other")

# A sentence, bullet or table row matching this is dropped.
DROP_LINE = re.compile(
    r"usage limit|weekly limit|week's limit|week’s limit|of one week|of a week|a week's usage|\bbudget|\bspend\b|\bSpend\b"
    r"|\$\s?\d|\bprice|\bpaid\b|pay money|no need to pay|\bcredits\b|subscription|ChatGPT|account hit|hit the weekly"
    r"|PROJECT\.md|weekly/|\.\./|~/|/Users/|alice2|\bdisk\b|\bnpm\b|\buv cache|limitation|B1[78]\b|First 0[89]\b"
    r"|\bshrank\b|leftover build files|\bcaps?\b|GPU-hours? (?:used|left|of)|\.(?:py|sh|npz|json|csv|tsx?)\b",
    re.I)
# Whole blocks left out (by exact title).
DROP_BLOCKS = {
    "Is this using my disk? Clean it and finish", "My disk is full: what can go?", "t8: Clear the npm and uv caches",
    "A weekly record of what I did", "t56: These weekly pages", "Put the videos on the Desktop",
    "Put the videos in iCloud and keep the Mac's disk free", "t7: Move the videos to iCloud and free the project folder",
    "The voice is bad: can I hear samples, and what do paid voices cost?", "t5: Voice samples and paid prices",
}
# Charts left out (money, usage and compute caps, disk, labels of unpublished videos).
DROP_CHARTS = {"w1-spend", "w2-spend", "w3-spend", "w1-voice-prices", "w1-disk", "w2-disk", "w3-disk",
                "w3-gpu", "w3-t55-gpu", "w2-beginner-plan", "w2-v2-lengths"}
# Wording fixes for internal names (applied outside Reza's own quotes).
REWORD = [(r"\bThe mother's\b", "The lead chat's"), (r"\bThe mother\b", "The lead chat"), (r"\bMother's\b", "The lead chat's"),
          (r"\bthe mother's\b", "the lead chat's"), (r"\bmother's\b", "the lead chat's"), (r"\bthe mother\b", "the lead chat"),
          (r"\bmother, (\d)", r"lead chat, \1"),
          (r"\s*\(B01–B18\)", ""), (r"in videos B01–B18", "in the beginner videos"), (r"lists B01–B18", "lists the beginner videos"),
          (r"`?INDEX\.md`?", "the watch list"), (r"`?facts\.md`?", "the facts sheet"), (r"`?story\.md`?", "the story timeline"),
          (r"`?series\.md`?", "the series plan"), (r"`?Study list\.html`?", "the study list page"),
          (r"`?Idea tree\.html`?", "a local idea-tree page"), (r"tasks/(t\d+)\.md", r"\1's notes"),
          (r"your own ALICE account", "the university's cluster")]


class ExportError(Exception):
    pass


def parse(path):
    head, blocks, cur, field = {}, [], None, None
    for n, raw in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
        line = raw.rstrip()
        m = re.match(r"^##\s+([QT])\s*\|\s*(.+)$", line)
        if m:
            cur = {"kind": m.group(1), "title": m.group(2).strip()}
            blocks.append(cur)
            field = None
            continue
        target = head if cur is None else cur
        m = re.match(r"^([a-z]+):\s?(.*)$", line)
        if m and (m.group(1) in KEYS or (cur is None and m.group(1) in HEAD_KEYS)):
            key, val = m.group(1), m.group(2)
            if cur is not None:
                key = ALIAS.get(key, key)
            if key in REPEAT:
                target.setdefault(key, []).append(val)
            else:
                if key in target:
                    raise ExportError(f"{path}:{n}: field '{key}' appears twice in one block")
                target[key] = val
            field = key
            continue
        if field is None:
            if line.strip() and not line.startswith("#") and not line.startswith("<!--"):
                raise ExportError(f"{path}:{n}: text outside any field: {line[:60]}")
            continue
        if field in REPEAT:
            target[field][-1] += "\n" + line
        else:
            target[field] += "\n" + line
    for k in ("title", "range", "oneline", "summary", "achievements"):
        if k not in head:
            raise ExportError(f"{path}: header field '{k}:' missing")
    return head, blocks


def reword(text):
    for a, b in REWORD:
        text = re.sub(a, b, text)
    return text


def clean_text(text, dropped, where):
    """Prose (paragraphs and '- ' lists): drop matching sentences and bullets, keep the rest."""
    out = []
    for line in text.strip().splitlines():
        s = line.rstrip()
        if not s.strip():
            out.append("")
            continue
        bullet = s.lstrip().startswith("- ")
        prefix = "- " if bullet else ""
        body = s.lstrip()[2:] if bullet else s
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"“])", body)
        keep = []
        for sent in sentences:
            sent = reword(sent)
            if DROP_LINE.search(sent):
                dropped.append(f"{where}: {sent[:110]}")
            else:
                keep.append(sent)
        if keep:
            out.append(prefix + " ".join(keep))
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def parse_table(text, dropped, where):
    rows = [r.strip() for r in text.strip().splitlines() if r.strip()]
    caption = ""
    if rows and not rows[0].startswith("|"):
        caption = rows.pop(0)
    rows = [r for r in rows if r.startswith("|")]
    if len(rows) < 2 or not re.match(r"^\|[\s:|-]+\|$", rows[1]):
        raise ExportError(f"{where}: table needs a header row and a |---| row")
    cells = [[reword(c.strip()) for c in r.strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[2:]
    caption = reword(caption)
    if DROP_LINE.search(caption) or any(DROP_LINE.search(h) for h in head):
        dropped.append(f"{where}: whole table '{caption or head[0]}'")
        return None
    kept = []
    for r in body:
        if any(DROP_LINE.search(c) for c in r):
            dropped.append(f"{where}: row {' | '.join(r)[:100]}")
        else:
            kept.append(r)
    if not kept:
        return None
    return {"caption": caption, "columns": head, "rows": kept}


def convert_block(b, idx, dropped):
    where = f"{b['kind']} '{b['title'][:40]}'"
    out = {"id": f"{b['kind'].lower()}{idx}", "title": reword(b["title"])}
    if b["kind"] == "Q":
        said = []
        for s in b.get("said", []):
            when, sep, words = s.partition("|")
            if not sep:
                raise ExportError(f"{where}: 'said:' must be '<date time> | <words>'")
            words = words.strip()
            if DROP_LINE.search(words):
                dropped.append(f"{where}: quote {words[:100]}")
                continue
            said.append({"when": when.strip(), "text": words})
        out["said"] = said
    for key in ("question", "answer", "why", "decisions", "outcome", "details"):
        if key in b:
            t = clean_text(b[key], dropped, where)
            if t:
                out[key] = t
    for src, dst in (("table", "tables"), ("dtable", "detailTables")):
        tables = [t for t in (parse_table(x, dropped, where) for x in b.get(src, [])) if t]
        if tables:
            out[dst] = tables
    for src, dst in (("chart", "charts"), ("dchart", "detailCharts")):
        charts = []
        for c in b.get(src, []):
            name, _, cap = c.partition("|")
            name = name.strip()
            if name in DROP_CHARTS:
                dropped.append(f"{where}: chart {name}")
                continue
            charts.append({"name": name, "caption": clean_text(cap.strip(), dropped, where) if cap.strip() else ""})
        if charts:
            out[dst] = charts
    return out


def write_index(outdir):
    """index.json lists every week file in the folder (newest first on the site), with the numbers the index page shows."""
    weeks = []
    for name in sorted(os.listdir(outdir)):
        if re.match(r"^\d{4}-\d{2}-\d{2}\.json$", name):
            w = json.load(open(os.path.join(outdir, name), encoding="utf-8"))
            weeks.append({k: w[k] for k in ("id", "title", "range", "oneline", "updated", "open")}
                         | {"questions": len(w["questions"]), "tasks": len(w["tasks"]),
                            "achievements": len(w.get("achievements", []))})
    with open(os.path.join(outdir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"weeks": weeks}, f, ensure_ascii=False, indent=1)


def export(path, outdir, is_open):
    head, blocks = parse(path)
    week = os.path.splitext(os.path.basename(path))[0]
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", week):
        raise ExportError("the file name must be the week's Friday date, like 2026-11-13.md")
    summary = [l.strip()[2:] for l in head["summary"].splitlines() if l.strip().startswith("- ")]
    if not 1 <= len(summary) <= 4:
        raise ExportError(f"'summary:' needs 1 to 4 '- ' lines (found {len(summary)})")
    achievements = [l.strip()[2:] for l in head["achievements"].splitlines() if l.strip().startswith("- ")]
    if not achievements:
        raise ExportError("'achievements:' needs at least one '- ' line")
    dropped = []
    questions, tasks = [], []
    for line in head.get("other", "").splitlines():
        if line.strip().startswith("- "):
            t = clean_text(line.strip()[2:], dropped, "other")
            if t:
                tasks.append({"id": f"o{len(tasks) + 1}", "title": t})
    for b in blocks:
        if b["title"] in DROP_BLOCKS:
            dropped.append(f"whole block: {b['kind']} {b['title']}")
            continue
        bucket = questions if b["kind"] == "Q" else tasks
        bucket.append(convert_block(b, len(bucket) + 1, dropped))
    doc = {
        "id": week,
        "title": head["title"].strip(),
        "range": clean_text(head["range"], dropped, "range"),
        "oneline": clean_text(head["oneline"], dropped, "oneline"),
        "updated": head.get("updated", "").strip(),
        "open": bool(is_open),
        "summary": [clean_text(s, dropped, "summary") for s in summary],
        "achievements": [a for a in (clean_text(a, dropped, "achievements") for a in achievements) if a],
        "questions": questions,
        "tasks": tasks,
    }
    doc["summary"] = [s for s in doc["summary"] if s]
    os.makedirs(outdir, exist_ok=True)
    target = os.path.join(outdir, week + ".json")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    write_index(outdir)
    print(f"{target}: {len(questions)} questions, {len(doc['achievements'])} achievements, {len(tasks)} tasks, "
          f"{len(dropped)} dropped pieces")
    for d in dropped:
        print("  dropped:", d)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        sys.exit(2)
    try:
        export(args[0], args[1], "--open" in sys.argv)
    except ExportError as e:
        print("export failed:", e, file=sys.stderr)
        sys.exit(1)
