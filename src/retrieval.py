"""Read small Markdown policies and retrieve candidate source sections."""

import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge_base"


@dataclass(frozen=True)
class Chunk:
    """One complete policy section, with its source reference."""

    chunk_id: str
    source_path: str
    section: str
    text: str
    line_start: int
    line_end: int


def read_sections(path: Path) -> list[Chunk]:
    """Split the supported Markdown format at level-two headings."""
    resolved_path = path.resolve()
    try:
        source_path = resolved_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        source_path = resolved_path.as_posix()
    lines = path.read_text(encoding="utf-8").splitlines()
    chunks = []
    section = ""
    body = []
    start_line = 0

    # A final heading flushes the last section without duplicating the code.
    for line_number, line in enumerate([*lines, "## "], start=1):
        if line.startswith("## "):
            if line_number <= len(lines) and not line[3:].strip():
                raise ValueError(f"Empty section heading in {path.name}:{line_number}")
            if section:
                text = "\n".join(body).strip()
                if not text:
                    raise ValueError(f"Empty section in {path.name}: {section}")
                slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
                if not slug:
                    raise ValueError(f"Invalid section heading in {path.name}")
                chunks.append(
                    Chunk(
                        chunk_id=f"{path.stem}::{slug}",
                        source_path=source_path,
                        section=section,
                        text=text,
                        line_start=start_line,
                        line_end=line_number - 1,
                    )
                )
            section = line[3:].strip()
            start_line = line_number
            body = []
        elif section:
            body.append(line)

    if not chunks:
        raise ValueError(f"No nonempty level-two sections in {path.name}")
    return chunks


def load_chunks(knowledge_base: Path = DEFAULT_KNOWLEDGE_BASE) -> list[Chunk]:
    """Load policy files only; evaluation examples are never indexed."""
    if not knowledge_base.is_dir():
        raise ValueError(f"Knowledge-base directory does not exist: {knowledge_base}")
    chunks = []
    for path in sorted(knowledge_base.glob("*.md")):
        if path.name.lower() != "readme.md":
            chunks.extend(read_sections(path))
    if not chunks:
        raise ValueError("The knowledge base contains no policy sections.")
    identifiers = [chunk.chunk_id for chunk in chunks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate chunk identifiers in the knowledge base.")
    return chunks
