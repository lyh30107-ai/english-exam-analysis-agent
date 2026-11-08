"""Parse the supplied two-column English exam and OMR workbook locally.

No API calls. Answers and weights always come from the uploaded answer PDF.
Question images are retained; native text is NOT claimed to describe graphics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import re
import statistics
import unicodedata

import pdfplumber
import pypdfium2 as pdfium
from openpyxl import load_workbook


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def integer(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid answer")
    s = unicodedata.normalize("NFKC", str(value)).strip()
    if not re.fullmatch(r"\d+(?:\.0+)?", s):
        raise ValueError(f"Invalid integer: {value!r}")
    return int(float(s))


def parse_key(path):
    key = {}
    with pdfplumber.open(path) as doc:
        full_text = "\n".join(p.extract_text() or "" for p in doc.pages)
        for page_no, page in enumerate(doc.pages, 1):
            for table in page.find_tables():
                rows = table.extract()
                header = [re.sub(r"\s+", "", x or "") for x in rows[0]]
                starts = [i for i in range(len(header)-2)
                          if header[i:i+3] == ["문항번호", "정답", "배점"]]
                for row in rows[1:]:
                    for j in starts:
                        values = row[j:j+3]
                        if all(x is None or not str(x).strip() for x in values):
                            continue
                        if len(values) != 3 or any(x is None for x in values):
                            raise ValueError(f"Incomplete key row: {values}")
                        n, answer, points = map(integer, values)
                        if n in key or not 1 <= answer <= 5 or points <= 0:
                            raise ValueError(f"Invalid/duplicate answer key: {values}")
                        key[n] = {"question_id": n, "answer": answer, "points": points,
                                  "source_page": page_no, "raw_cells": values}
        if not key or sorted(key) != list(range(1, max(key)+1)):
            raise ValueError("Answer key has missing question numbers")
        form = re.search(r"(홀수|짝수)\s*\)?\s*형", full_text)
        return key, {"form": form.group(1) if form else None,
                     "question_count": len(key), "max_score": sum(x["points"] for x in key.values())}


def parse_students(path, key):
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        matches = []
        for sheet in workbook:
            first = next(sheet.values, ())
            header = [str(x).strip() if x is not None else "" for x in first]
            if "학생명" in header and any(re.fullmatch(r"Q\d+", x) for x in header):
                matches.append((sheet, header))
        if len(matches) != 1:
            raise ValueError("Expected exactly one OMR worksheet with 학생명 and Q columns")
        sheet, header = matches[0]
        columns = {}
        for j, label in enumerate(header):
            m = re.fullmatch(r"Q(\d+)", label)
            if m:
                q = int(m[1])
                if q in columns:
                    raise ValueError(f"Duplicate question column: {label}")
                columns[q] = j
        if set(columns) != set(key):
            raise ValueError("OMR columns do not match the answer PDF")
        students, seen, blank_rows = [], set(), []
        for row_no, row in enumerate(sheet.iter_rows(min_row=2, values_only=True), 2):
            if all(x is None or isinstance(x, str) and not x.strip() for x in row):
                blank_rows.append(row_no)
                continue
            name = row[header.index("학생명")]
            if name is None or not str(name).strip():
                raise ValueError(f"Student name missing at row {row_no}")
            name = str(name).strip()
            if name in seen:
                raise ValueError(f"Duplicate student: {name}")
            seen.add(name)
            answers = {}
            for q, j in columns.items():
                v = row[j]
                # Preserve unanswered items explicitly; scoring policy below is 0.
                a = None if v is None or isinstance(v, str) and not v.strip() else integer(v)
                if a is not None and a not in range(1, 6):
                    raise ValueError(f"Invalid answer at row {row_no}, Q{q}: {v}")
                answers[q] = a
            students.append({"student_id": name, "source_row": row_no, "answers": answers})
        if not students:
            raise ValueError("No students found")
        return students, {"sheet": sheet.title, "blank_rows_removed": blank_rows,
                          "student_count": len(students), "unanswered_policy": "0 points"}
    finally:
        workbook.close()


def grade(students, key):
    details, scores, wrong = [], [], Counter({q: 0 for q in key})
    for student in students:
        score = 0
        for q in sorted(key):
            selected = student["answers"][q]
            correct = selected == key[q]["answer"]
            earned = key[q]["points"] if correct else 0
            score += earned
            wrong[q] += int(not correct)
            details.append({"student_id": student["student_id"], "question_id": q,
                            "response": selected, "answer": key[q]["answer"],
                            "points": key[q]["points"], "correct": correct, "earned": earned})
        scores.append({"student_id": student["student_id"], "score": score})
    values = [s["score"] for s in scores]
    stats = {"student_count": len(students), "mean": statistics.mean(values),
             "median": statistics.median(values), "std_population": statistics.pstdev(values),
             "std_sample": statistics.stdev(values) if len(values) > 1 else None,
             "lowest_students": [s for s in scores if s["score"] == min(values)],
             "item_errors": [{"question_id": q, "wrong_count": n, "wrong_rate": n/len(students)}
                             for q, n in sorted(wrong.items(), key=lambda x: (-x[1], x[0]))]}
    return details, scores, stats


QSTART = re.compile(r"^(\d{1,2})\.\s*(.*)")
GROUP = re.compile(r"^\[(\d{1,2})[～~∼–-](\d{1,2})\]")


def question_type(text, listening):
    if listening:
        return "듣기"
    prompt = text[:250]
    for needle, label in [("낱말의 쓰임", "문맥상 어휘"), ("어법상", "어법"),
                          ("의미하는", "함축 의미 추론"), ("순서", "글의 순서"),
                          ("문장이 들어가기", "문장 삽입"), ("요약", "요약문 완성"),
                          ("빈칸", "빈칸 추론"), ("도표", "도표 해석"),
                          ("가리키는", "지칭 추론"), ("관계 없는", "무관한 문장"),
                          ("제목", "제목"), ("주제", "주제"), ("요지", "요지"),
                          ("목적", "목적"), ("심경", "심경"), ("주장", "주장")]:
        if needle in prompt:
            return label
    return "내용 일치/기타"


def parse_questions(path, key, output):
    """Layout-specific parser. Retain page crops for information not encoded as text."""
    records, groups, intro = {}, {}, {"text": "", "spans": []}
    current = intro
    pages_manifest = []
    rendered = pdfium.PdfDocument(path)
    image_cache = {}
    def add_region(target, page, page_no, column, x0, x1, top, bottom):
        if bottom <= top:
            return
        crop = page.crop((x0, top, x1, bottom))
        text = crop.extract_text(x_tolerance=2, y_tolerance=3) or ""
        if not text.strip():
            return
        if page_no not in image_cache:
            image_cache[page_no] = rendered[page_no-1].render(scale=1.5).to_pil()
        image = image_cache[page_no]
        filename = f"assets/p{page_no}_{column}_{round(top*10)}.png"
        image.crop(tuple(round(v*1.5) for v in (x0, top, x1, bottom))).save(output/filename)
        target["text"] += ("\n" if target["text"] else "") + text
        # Keep vector/character geometry so underlines and blank positions remain recoverable.
        geometry = {"chars": [{k: c[k] for k in ("text", "x0", "x1", "top", "bottom", "fontname", "size")}
                              for c in crop.chars],
                    "lines": [{k: c[k] for k in ("x0", "x1", "top", "bottom")} for c in crop.lines],
                    "rects": [{k: c[k] for k in ("x0", "x1", "top", "bottom")} for c in crop.rects]}
        geometry_file = filename.replace(".png", ".json")
        save_json(output/geometry_file, geometry)
        target["spans"].append({"page": page_no, "column": column,
                                "bbox": [x0, top, x1, bottom], "image": filename,
                                "geometry": geometry_file, "image_objects": len(crop.images)})
        # Candidate table cells retain their row/column relationships, including boxed text.
        target.setdefault("table_candidates", []).extend(
            {"page": page_no, "bbox": list(t.bbox), "cells": t.extract()}
            for t in crop.find_tables())
        annotations = target.setdefault("horizontal_marks", [])
        seen_marks = set()
        for line in crop.lines:
            width = line["x1"]-line["x0"]
            sig = tuple(round(line[k], 1) for k in ("x0", "x1", "top", "bottom"))
            if sig in seen_marks or abs(line['top']-line['bottom'])>.3 or not 10<width<(x1-x0)*.7:
                continue
            seen_marks.add(sig)
            baseline_chars = sorted([c for c in crop.chars if -1 < line['top']-c['bottom'] < 6],key=lambda c:c['x0'])
            above = ''.join(c['text'] for c in baseline_chars if line['x0']-1 <= (c['x0']+c['x1'])/2 <= line['x1']+1)
            left = ''.join(c['text'] for c in baseline_chars if c['x1']<=line['x0'])
            right = ''.join(c['text'] for c in baseline_chars if c['x0']>=line['x1'])
            annotations.append({"page": page_no, "bbox": list(sig), "text_above": above,
                                "left_context": left, "right_context": right,
                                "kind": "underline_candidate" if above.strip() else "blank_candidate"})
    with pdfplumber.open(path) as doc:
        all_text = "\n".join(p.extract_text() or "" for p in doc.pages)
        listen = re.search(r"(\d+)번부터\s*(\d+)번까지는\s*듣고", all_text)
        listening_ids = set(range(int(listen[1]), int(listen[2])+1)) if listen else set()
        for page_no, page in enumerate(doc.pages, 1):
            # Detect the long rule below the page header; no question-specific coordinates.
            rules = [l["top"] for l in page.lines if l["x1"]-l["x0"] > page.width*.65
                     and abs(l["bottom"]-l["top"]) < 1 and page.height*.08 < l["top"] < page.height*.22]
            if not rules:
                raise ValueError(f"Unrecognized two-column exam layout on page {page_no}")
            body_top, body_bottom = max(rules)+2, page.height*.9
            pages_manifest.append({"page": page_no, "body_top": body_top, "body_bottom": body_bottom})
            for col, (x0, x1) in enumerate([(0, page.width/2), (page.width/2, page.width)], 1):
                crop = page.crop((x0, body_top, x1, body_bottom))
                lines = crop.extract_text_lines(x_tolerance=2, y_tolerance=3)
                events = []
                for line in lines:
                    txt = line["text"].strip()
                    qm, gm = QSTART.match(txt), GROUP.match(txt)
                    if txt.startswith("* 확인 사항") or txt.startswith("이제 듣기 문제가 끝났습니다"):
                        events.append((line["top"]-1, "end", None))
                    elif gm:
                        events.append((line["top"]-1, "group", (int(gm[1]), int(gm[2]))))
                    elif qm and int(qm[1]) in key:
                        events.append((line["top"]-1, "question", int(qm[1])))
                cursor = body_top
                for top, kind, value in events:
                    if current is not None:
                        add_region(current, page, page_no, col, x0, x1, cursor, top)
                    if kind == "question":
                        if value in records:
                            raise ValueError(f"Duplicate question anchor: {value}")
                        current = records[value] = {"question_id": value, "text": "", "spans": []}
                    elif kind == "group":
                        label = f"{value[0]}-{value[1]}"
                        current = groups[label] = {"question_ids": list(range(value[0], value[1]+1)),
                                                   "text": "", "spans": []}
                    else:
                        current = None
                    cursor = top
                if current is not None:
                    add_region(current, page, page_no, col, x0, x1, cursor, body_bottom)
        if set(records) != set(key):
            raise ValueError(f"Missing question anchors: {set(key)-set(records)}")
        for q, record in records.items():
            group_ids = [g for g, data in groups.items() if q in data["question_ids"]]
            record["shared_context_ids"] = group_ids
            context = "\n".join(groups[g]["text"] for g in group_ids)
            record["complete_context"] = context + "\n" + record["text"]
            # The actual question takes precedence over a long shared passage.
            group_instructions = "\n".join(groups[g]["text"].split("\n")[0] for g in group_ids)
            record["type"] = question_type(record["text"][:180] + "\n" + group_instructions, q in listening_ids)
            record["listening_script_available"] = False if q in listening_ids else None
            record["answer"] = key[q]["answer"]
            record["points"] = key[q]["points"]
            # Preserve inline markers as markers, never blindly label them standalone options.
            record["choice_markers"] = sorted(set(re.findall("[①②③④⑤]", record["text"])))
            markers = list(re.finditer("[①②③④⑤]", record["text"]))
            inline = record['type'] in ['어법','무관한 문장','문장 삽입','도표 해석'] or '밑줄 친 부분' in record['text'][:100]
            record['choice_mode'] = 'inline_positions' if inline else 'standalone_options'
            if len(markers)==5 and [m[0] for m in markers]==list('①②③④⑤'):
                record['stem_text'] = record['text'][:markers[0].start()] if not inline else record['text']
                record['options'] = [] if inline else [
                    {'number':i+1,'text':record['text'][m.end():markers[i+1].start() if i<4 else len(record['text'])].strip()}
                    for i,m in enumerate(markers)]
            else:
                record['stem_text'] = record['text']
                record['options'] = []
                record['choice_mode'] = 'source_image_required'
            record["representation"] = "native_text_and_source_crops"
            record["requires_source_images_for_generation"] = True
            record["warnings"] = (["Original listening script/audio not supplied"] if q in listening_ids else [])
            if any(s["image_objects"] for s in record["spans"]):
                record["warnings"].append("Embedded visual retained in crop; semantic extraction not verified")
        form = re.search(r"(홀수|짝수)\s*형", all_text)
    rendered.close()
    return {"questions": [records[n] for n in sorted(records)], "shared_contexts": groups,
            "introduction": intro, "pages": pages_manifest,
            "form": form[1] if form else None}


def review_html(parsed, stats):
    body = ["<h1>문항별 파싱 대조</h1><p>왼쪽은 원본 영역, 오른쪽은 추출 텍스트입니다. "
            "그림·밑줄·빈칸의 의미를 텍스트가 모두 표현한다고 보장하지 않습니다.</p>"]
    for q in parsed["questions"]:
        spans = [s for g in q["shared_context_ids"] for s in parsed["shared_contexts"][g]["spans"]] + q["spans"]
        images = "".join(f'<figure><img loading="lazy" src="{s["image"]}"><figcaption>'
                         f'PDF {s["page"]}쪽, {s["column"]}단</figcaption></figure>' for s in spans)
        body.append(f'<section id="q{q["question_id"]}"><h2>{q["question_id"]}번 · {q["type"]} '
                    f'· 정답 {q["answer"]} · {q["points"]}점</h2><p>{html.escape(" / ".join(q["warnings"]))}</p>'
                    f'<div class="row"><div>{images}</div><pre>{html.escape(q["complete_context"])}</pre></div></section>')
    return '<!doctype html><meta charset="utf-8"><title>문항 파싱 검토</title><style>' \
           'body{font:16px sans-serif;margin:28px;background:#f5f6f8}section{background:white;padding:20px;margin:24px 0}' \
           '.row{display:grid;grid-template-columns:1fr 1fr;gap:24px}img{width:100%}pre{white-space:pre-wrap;line-height:1.6}' \
           'figure{margin:0 0 20px}h2{color:#174a71}@media(max-width:800px){.row{grid-template-columns:1fr}}</style>' + "\n".join(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--students", type=Path, required=True)
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("parsed"))
    args = ap.parse_args()
    out = args.output
    (out/"assets").mkdir(parents=True, exist_ok=True)
    key, key_meta = parse_key(args.answers)
    students, student_meta = parse_students(args.students, key)
    details, scores, stats = grade(students, key)
    questions = parse_questions(args.questions, key, out)
    if not key_meta["form"] or key_meta["form"] != questions["form"]:
        raise ValueError("Exam/answer sheet form missing or mismatched")
    for filename, data in [("answer_key.json", list(key.values())), ("students.json", students),
                           ("questions.json", questions), ("scores.json", scores), ("statistics.json", stats)]:
        save_json(out/filename, data)
    with (out/"grading_details.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=details[0].keys());writer.writeheader();writer.writerows(details)
    manifest = {"sources": [{"filename": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                            for p in [args.students, args.answers, args.questions]],
                "answer_key": key_meta, "students": student_meta,
                "question_count": len(questions["questions"]), "shared_groups": list(questions["shared_contexts"]),
                "limitations": ["Layout profile: supplied two-column English exam", "No listening audio/transcript supplied",
                                "Graphics are preserved as crops, not automatically semantically decoded",
                                "Question types are rule-based candidates; verify against source"]}
    save_json(out/"manifest.json", manifest)
    (out/"review.html").write_text(review_html(questions, stats), encoding="utf-8")
    print(json.dumps({"students": len(students), "questions": len(key), "max_score": key_meta["max_score"],
                      "mean": stats["mean"], "median": stats["median"], "top_errors": stats["item_errors"][:3]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
