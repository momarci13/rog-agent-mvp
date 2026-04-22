"""Ingest PDFs / markdown / text / BibTeX into the RAG store.

Usage:
  python -m rag.ingest path/to/file_or_dir [--collection main]
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from .hybrid import LiteHybridRAG


# -------------- chunking --------------

def chunk_text(text: str, target_tokens: int = 256, overlap: int = 32) -> list[str]:
    """Simple token-approximate chunker. 1 token ≈ 4 chars for English."""
    target = target_tokens * 4
    over = overlap * 4
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if not p.strip():
            continue
        if len(buf) + len(p) + 2 <= target:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) > target:
                # hard-wrap long paragraphs
                for i in range(0, len(p), target - over):
                    chunks.append(p[i : i + target])
                buf = ""
            else:
                # retain overlap tail
                tail = buf[-over:] if buf else ""
                buf = (tail + "\n\n" + p).strip()
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


# -------------- loaders --------------

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf required. Install: pip install pypdf")
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)


def parse_bibtex(path: Path) -> list[dict]:
    """Return one entry per BibTeX record with {id,text,meta}.

    Keeps it stdlib only; handles @article, @book, @inproceedings, etc.
    """
    raw = load_text(path)
    entries = []
    # split on @type{key, ... }
    pattern = re.compile(r"@(\w+)\s*\{\s*([^,]+)\s*,(.*?)\n\}", re.DOTALL)
    for m in pattern.finditer(raw):
        kind, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        fields = {}
        for fm in re.finditer(r"(\w+)\s*=\s*[{\"]([^}\"]+)[}\"]\s*,?", body):
            fields[fm.group(1).lower()] = fm.group(2).strip()
        title = fields.get("title", "")
        authors = fields.get("author", "")
        year = fields.get("year", "")
        abstract = fields.get("abstract", "")
        text = (
            f"{kind.upper()} {key}\n"
            f"Title: {title}\nAuthors: {authors}\nYear: {year}\n"
            f"Abstract: {abstract}"
        )
        entries.append({
            "id": f"bib::{key}",
            "text": text,
            "meta": {"kind": "bib", "key": key, "year": year, "title": title},
        })
    return entries


# -------------- top-level --------------

def ingest_file(path: Path, rag: LiteHybridRAG, chunk_tokens: int = 256) -> int:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".pdf":
        text = load_pdf(path)
    elif suf in {".md", ".txt", ".tex"}:
        text = load_text(path)
    elif suf == ".bib":
        entries = parse_bibtex(path)
        rag.add(entries)
        return len(entries)
    else:
        return 0

    chunks = chunk_text(text, target_tokens=chunk_tokens)
    docs = []
    for i, ch in enumerate(chunks):
        h = hashlib.md5(f"{path.name}::{i}".encode()).hexdigest()[:12]
        docs.append({
            "id": f"{path.stem}::{h}",
            "text": ch,
            "meta": {"source": path.name, "chunk": i, "kind": "doc"},
        })
    rag.add(docs)
    return len(docs)


def ingest_path(path: str | Path, rag: LiteHybridRAG, chunk_tokens: int = 256) -> int:
    p = Path(path)
    total = 0
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in {".pdf", ".md", ".txt", ".tex", ".bib"}:
                n = ingest_file(f, rag, chunk_tokens)
                total += n
                print(f"  ingested {n:4d} chunks from {f.name}")
    else:
        total = ingest_file(p, rag, chunk_tokens)
        print(f"  ingested {total} chunks from {p.name}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="file or directory to ingest")
    ap.add_argument("--db", default="./kb/chroma")
    ap.add_argument("--collection", default="main")
    ap.add_argument("--chunk-tokens", type=int, default=256)
    args = ap.parse_args()

    rag = LiteHybridRAG(db_path=args.db, collection=args.collection)
    n = ingest_path(args.path, rag, args.chunk_tokens)
    print(f"\nTotal chunks in collection '{args.collection}': {len(rag)} (+{n} new)")


if __name__ == "__main__":
    main()
