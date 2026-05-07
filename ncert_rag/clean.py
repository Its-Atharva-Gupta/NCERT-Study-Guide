from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


_WS_RE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class CleanOptions:
    max_header_footer_lines: int = 3
    header_footer_min_frac: float = 0.55
    drop_pages_below_chars: int = 150


_DROP_LINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*\d+\s*$"),  # standalone page number
    re.compile(r"^\s*Unit\s+\d+\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Reprint\b.*$", re.IGNORECASE),
    re.compile(r".*\.indd\s+\d+.*$", re.IGNORECASE),
    re.compile(r"^\s*Poorvi\s+—\s*Grade\s+\d+.*$", re.IGNORECASE),
    re.compile(r"^\s*FabLes\s+and\s+Folk\s+Tales\s*$", re.IGNORECASE),
)

_SECTION_HEADING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*Let\s+us\s+do\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Let\s+us\s+think\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Let\s+us\s+talk\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Let\s+us\s+write\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Let\s+us\s+read\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Let\s+us\s+practice\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Let\s+us\s+learn\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Let\s+us\s+listen\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Let\s+us\s+sing\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Let\s+us\s+play\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Let\s+us\s+think\s+and\s+reflect\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Word\s+meanings?\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Glossary\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Exercises?\b.*$", re.IGNORECASE),
    re.compile(r"^\s*Comprehension\b.*$", re.IGNORECASE),
)

_QUESTIONISH_RE = re.compile(r"^\s*(?:[IVX]+\.|\d+\.|\(\s*[ivx]+\s*\)|\(\s*\d+\s*\))\s+")
_INSTRUCTION_VERBS_RE = re.compile(
    r"^\s*(?:Read|Answer|Write|Complete|Fill|Match|Choose|Tick|Circle|Discuss|Think|Explain)\b",
    re.IGNORECASE,
)

_GLOSSARY_LINE_RE = re.compile(
    r"^\s*[A-Za-z][A-Za-z '\u2010\u2011\u2012\u2013\u2014-]{1,35}:\s+.+$"
)

_AUTHOR_LINE_RE = re.compile(
    r"^\s*(?:—|-)\s*([A-Z][A-Za-z .'-]{2,})\s*$"
)


def _norm_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    return _WS_RE.sub(" ", line).strip()


def _candidate_header_footer(lines: list[str], n: int) -> list[str]:
    if not lines:
        return []
    top = [l for l in lines[:n] if l]
    bottom = [l for l in lines[-n:] if l]
    return top + bottom


def _boilerplate_lines(pages_lines: list[list[str]], options: CleanOptions) -> set[str]:
    counts: Counter[str] = Counter()
    for lines in pages_lines:
        for l in _candidate_header_footer(lines, options.max_header_footer_lines):
            counts[l] += 1
    if not pages_lines:
        return set()
    threshold = int(len(pages_lines) * options.header_footer_min_frac)
    return {l for l, c in counts.items() if c >= threshold}


def _should_drop_line(line: str, boilerplate: set[str]) -> bool:
    if not line:
        return True
    if line in boilerplate:
        return True
    for pat in _DROP_LINE_PATTERNS:
        if pat.match(line):
            return True
    return False


def _is_section_heading(line: str) -> bool:
    for pat in _SECTION_HEADING_PATTERNS:
        if pat.match(line):
            return True
    return False


def _is_exercise_line(line: str) -> bool:
    if _is_section_heading(line):
        return True
    if _GLOSSARY_LINE_RE.match(line):
        return True
    if _QUESTIONISH_RE.match(line) and ("?" in line or _INSTRUCTION_VERBS_RE.match(line)):
        return True
    if _INSTRUCTION_VERBS_RE.match(line):
        return True
    return False


def extract_clean_text(pdf_path: Path, *, options: CleanOptions = CleanOptions()) -> tuple[str, str | None]:
    reader = PdfReader(str(pdf_path))
    raw_pages: list[list[str]] = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        lines = [_norm_line(l) for l in txt.splitlines()]
        lines = [l for l in lines if l]
        raw_pages.append(lines)

    boilerplate = _boilerplate_lines(raw_pages, options)

    kept_lines: list[str] = []
    author: str | None = None
    for page_lines in raw_pages:
        for line in page_lines:
            if _should_drop_line(line, boilerplate):
                continue

            m = _AUTHOR_LINE_RE.match(line)
            if m:
                author = m.group(1).strip()
                continue

            if _is_exercise_line(line):
                continue

            kept_lines.append(line)

    text = "\n".join(kept_lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    if len(text) < options.drop_pages_below_chars:
        return "", author
    return text, author


def write_clean_corpus(
    pdf_paths: Iterable[Path],
    *,
    out_dir: Path,
    options: CleanOptions = CleanOptions(),
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for pdf_path in sorted(pdf_paths):
        clean_text, author = extract_clean_text(pdf_path, options=options)
        if not clean_text:
            continue
        title = pdf_path.stem
        header = [f"Title: {title}", f"Source: {pdf_path.name}"]
        if author:
            header.append(f"Author: {author}")
        header.append("")  # blank line
        out_text = "\n".join(header) + clean_text + "\n"

        out_path = out_dir / f"{pdf_path.stem}.txt"
        out_path.write_text(out_text, encoding="utf-8")
        written.append(out_path)

    return written
