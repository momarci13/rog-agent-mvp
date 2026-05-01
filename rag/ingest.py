"""Ingest PDFs / markdown / text / BibTeX into the RAG store.

Usage:
  python -m rag.ingest path/to/file_or_dir [--collection main]
"""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

try:
    import tiktoken
    _TIKTOKEN_ENC = None  # Lazy initialization
except ImportError:
    _TIKTOKEN_ENC = False  # Marker that tiktoken is not available

try:
    import nltk
    from nltk.tokenize import sent_tokenize
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

from .hybrid import LiteHybridRAG

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".tex", ".bib"}


def _get_tiktoken_enc():
    """Lazily initialize tiktoken encoder."""
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENC = False
    return _TIKTOKEN_ENC if _TIKTOKEN_ENC else None


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (accurate)."""
    enc = _get_tiktoken_enc()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback to ÷4 heuristic if encoding fails
    return max(1, len(text) // 4)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK or fallback."""
    if _NLTK_AVAILABLE:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    # Fallback: simple regex split
    return re.split(r'(?<=[.!?])\s+', text.strip())


# -------------- chunking --------------

def chunk_text(text: str, target_tokens: int = 256, overlap: int = 32) -> list[str]:
    """Token-aware chunker using tiktoken for accurate token counting."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: list[str] = []
    buf = ""
    overlap_budget = overlap  # tokens to preserve for overlap

    for p in paragraphs:
        if not p.strip():
            continue

        # Check if adding this paragraph to buffer would exceed target
        if buf:
            candidate = buf + "\n\n" + p
        else:
            candidate = p
        
        tokens_candidate = _count_tokens(candidate)

        if tokens_candidate <= target_tokens:
            # Fits within budget, add to buffer
            buf = candidate
        else:
            # Would exceed budget
            if buf:
                chunks.append(buf)
            
            # Check if paragraph alone fits
            tokens_p = _count_tokens(p)
            if tokens_p > target_tokens:
                # Paragraph too long; split on sentences semantically
                sentences = _split_sentences(p)
                sub_buf = ""
                for sent in sentences:
                    cand = (sub_buf + " " + sent).strip() if sub_buf else sent
                    if _count_tokens(cand) <= target_tokens:
                        sub_buf = cand
                    else:
                        if sub_buf:
                            chunks.append(sub_buf)
                        sub_buf = sent
                if sub_buf:
                    chunks.append(sub_buf)
                buf = ""
            else:
                # Paragraph fits alone; prepare overlap
                # Take tail of previous buffer for overlap context
                words = buf.split()
                overlap_words = []
                overlap_tokens_current = 0
                for word in reversed(words):
                    test_seq = " ".join([word] + overlap_words)
                    if _count_tokens(test_seq) <= overlap_budget:
                        overlap_words.insert(0, word)
                        overlap_tokens_current = _count_tokens(test_seq)
                    else:
                        break
                
                if overlap_words:
                    buf = " ".join(overlap_words) + "\n\n" + p
                else:
                    buf = p

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

def ingest_file(
    path: Path,
    rag: LiteHybridRAG,
    chunk_tokens: int = 256,
    overlap: int = 32,
    source_tag: str | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> tuple[int, int, int]:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".pdf":
        text = load_pdf(path)
    elif suf in {".md", ".txt", ".tex"}:
        text = load_text(path)
    elif suf == ".bib":
        entries = parse_bibtex(path)
        if source_tag:
            for entry in entries:
                entry["meta"]["source_tag"] = source_tag
        existing = set(rag._ids) if skip_existing else set()
        docs_to_add = [e for e in entries if e["id"] not in existing]
        skipped = len(entries) - len(docs_to_add)
        if docs_to_add and not dry_run:
            rag.add(docs_to_add)
        return len(docs_to_add), skipped, len(entries)
    else:
        return 0, 0, 0

    chunks = chunk_text(text, target_tokens=chunk_tokens, overlap=overlap)
    docs = []
    topic = None
    if path.stem in {
        "stochastics_probability",
        "bayesian_methods",
        "markov_chains",
        "stochastic_processes_finance",
    }:
        topic = path.stem

    for i, ch in enumerate(chunks):
        h = hashlib.md5(f"{path.name}::{i}".encode()).hexdigest()[:12]
        meta = {"source": path.name, "chunk": i, "kind": "doc"}
        if topic:
            meta["topic"] = topic
        docs.append({
            "id": f"{path.stem}::{h}",
            "text": ch,
            "meta": meta,
        })

    existing = set(rag._ids) if skip_existing else set()
    docs_to_add = [d for d in docs if d["id"] not in existing]
    skipped = len(docs) - len(docs_to_add)
    if docs_to_add and not dry_run:
        rag.add(docs_to_add)
    return len(docs_to_add), skipped, len(docs)


def ingest_path(
    path: str | Path,
    rag: LiteHybridRAG,
    chunk_tokens: int = 256,
    overlap: int = 32,
    source_tag: str | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
) -> dict[str, int]:
    p = Path(path)
    counts = {"added": 0, "skipped": 0, "total": 0}
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                added, skipped, total = ingest_file(
                    f,
                    rag,
                    chunk_tokens=chunk_tokens,
                    overlap=overlap,
                    source_tag=source_tag,
                    skip_existing=skip_existing,
                    dry_run=dry_run,
                )
                print(
                    f"  {'would ingest' if dry_run else 'ingested'} {added:4d} chunks",
                    f"from {f.name} ({skipped} skipped, {total} total)"
                )
                counts["added"] += added
                counts["skipped"] += skipped
                counts["total"] += total
    else:
        added, skipped, total = ingest_file(
            p,
            rag,
            chunk_tokens=chunk_tokens,
            overlap=overlap,
            source_tag=source_tag,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )
        print(
            f"  {'would ingest' if dry_run else 'ingested'} {added:4d} chunks",
            f"from {p.name} ({skipped} skipped, {total} total)"
        )
        counts["added"] += added
        counts["skipped"] += skipped
        counts["total"] += total
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="file or directory to ingest")
    ap.add_argument("--db", default="./kb/chroma")
    ap.add_argument("--collection", default="main")
    ap.add_argument("--chunk-tokens", type=int, default=256)
    ap.add_argument("--overlap-tokens", type=int, default=32)
    ap.add_argument("--source-tag", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rag = LiteHybridRAG(db_path=args.db, collection=args.collection)
    counts = ingest_path(
        args.path,
        rag,
        chunk_tokens=args.chunk_tokens,
        overlap=args.overlap_tokens,
        source_tag=args.source_tag,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"\nDry run complete. {counts['added']} chunks would be added, {counts['skipped']} skipped, {counts['total']} seen.")
    else:
        print(f"\nTotal chunks in collection '{args.collection}': {len(rag)} (+{counts['added']} new; {counts['skipped']} duplicates skipped)")


if __name__ == "__main__":
    main()
