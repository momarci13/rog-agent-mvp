"""Lite hybrid RAG: dense (bge-small) + BM25, weighted fusion.

Deliberately no cross-encoder reranker — it costs another 300 MB VRAM
we don't have. For <100k chunks, fusion alone is ~95% as good.

Math:
  s_dense(q,d) = cos(phi(q), phi(d))
  s_bm25(q,d)  = sum_t IDF(t) * tf(t,d)(k1+1) / (tf(t,d) + k1(1-b+b|d|/avgL))
  s(q,d)       = alpha * s_dense + (1-alpha) * s_bm25_norm
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:  # pragma: no cover
    raise ImportError(
        "chromadb is required. Install with: pip install chromadb"
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    raise ImportError(
        "sentence-transformers is required. Install: pip install sentence-transformers"
    ) from e

try:
    from rank_bm25 import BM25Okapi
except Exception as e:  # pragma: no cover
    raise ImportError("rank-bm25 is required. Install: pip install rank-bm25") from e

try:
    import tiktoken
except Exception as e:  # pragma: no cover
    raise ImportError("tiktoken is required. Install: pip install tiktoken") from e

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
except Exception as e:  # pragma: no cover
    raise ImportError("nltk is required. Install: pip install nltk") from e


# -------------- tokenization & stemming ----------------

_stemmer = PorterStemmer()
_stopwords = set(stopwords.words('english'))
_tiktoken_enc = tiktoken.get_encoding("cl100k_base")


def _tokenize_stem(text: str) -> list[str]:
    """Tokenize, stem, and remove stopwords."""
    tokens = text.lower().split()
    stemmed = [_stemmer.stem(t) for t in tokens]
    return [t for t in stemmed if t not in _stopwords and len(t) > 1]


def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken (more accurate than ÷4 heuristic)."""
    try:
        return len(_tiktoken_enc.encode(text))
    except Exception:
        # Fallback to rough estimate if encoding fails
        return max(1, len(text) // 4)


class LiteHybridRAG:
    def __init__(
        self,
        db_path: str = "./kb/chroma",
        collection: str = "main",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        alpha_dense: float = 0.6,
    ):
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.collection_name = collection
        self.alpha = alpha_dense

        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.col = self.client.get_or_create_collection(collection)

        self.emb = SentenceTransformer(embedding_model)

        # BM25 is rebuilt from chroma on load, cached to disk.
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._bm25_path = Path(db_path) / f"{collection}_bm25.pkl"
        self._load_bm25_cache()

    # ---------------- ingestion ----------------

    def add(self, docs: list[dict]) -> None:
        """docs: [{"id": str, "text": str, "meta": dict}]"""
        if not docs:
            return
        texts = [d["text"] for d in docs]
        vecs = self.emb.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        ids = [d["id"] for d in docs]
        metas = [d.get("meta", {}) or {} for d in docs]

        self.col.add(ids=ids, embeddings=vecs, documents=texts, metadatas=metas)

        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metas.extend(metas)
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        tokens = [_tokenize_stem(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokens) if tokens else None
        self._save_bm25_cache()

    def _save_bm25_cache(self) -> None:
        with open(self._bm25_path, "wb") as f:
            pickle.dump(
                {"ids": self._ids, "texts": self._texts, "metas": self._metas}, f,
            )

    def _load_bm25_cache(self) -> None:
        if self._bm25_path.exists():
            with open(self._bm25_path, "rb") as f:
                cache = pickle.load(f)
            self._ids = cache.get("ids", [])
            self._texts = cache.get("texts", [])
            self._metas = cache.get("metas", [])
            if self._texts:
                self._bm25 = BM25Okapi([_tokenize_stem(t) for t in self._texts])

    # ---------------- retrieval ----------------

    def retrieve(
        self,
        query: str,
        k: int = 6,
        m: int = 30,
        token_budget: int = 3500,
        metadata_filters: dict | None = None,
    ) -> list[dict]:
        """Retrieve top-k documents using hybrid dense+BM25 retrieval.
        
        Args:
            query: search query
            k: max documents to return
            m: candidates before fusion
            token_budget: max tokens for context packing
            metadata_filters: dict with optional keys:
                - kind: "doc" or "bib"
                - year_min, year_max: year range (as strings)
                - source: filename filter
        
        Returns:
            list of {"id", "text", "meta", "score"} dicts
        """
        if not self._ids:
            return []

        m = min(m, len(self._ids))

        # Apply metadata filtering
        valid_indices = set(range(len(self._ids)))
        if metadata_filters:
            if metadata_filters.get("kind"):
                kind = metadata_filters["kind"]
                valid_indices &= {
                    i for i in valid_indices
                    if i < len(self._metas) and self._metas[i].get("kind") == kind
                }
            if "year_min" in metadata_filters or "year_max" in metadata_filters:
                year_min = metadata_filters.get("year_min")
                year_max = metadata_filters.get("year_max")
                def in_year_range(y):
                    if not y:
                        return False
                    try:
                        y_int = int(y)
                        if year_min and y_int < int(year_min):
                            return False
                        if year_max and y_int > int(year_max):
                            return False
                        return True
                    except (ValueError, TypeError):
                        return False
                valid_indices &= {
                    i for i in valid_indices
                    if i < len(self._metas) and in_year_range(self._metas[i].get("year"))
                }
            if metadata_filters.get("source"):
                source = metadata_filters["source"]
                valid_indices &= {
                    i for i in valid_indices
                    if i < len(self._metas) and self._metas[i].get("source") == source
                }

        if not valid_indices:
            return []

        # Dense
        qv = self.emb.encode([query], normalize_embeddings=True).tolist()
        dres = self.col.query(query_embeddings=qv, n_results=len(self._ids))
        # chroma returns distances in cosine distance space (1 - cos)
        d_ids = dres["ids"][0]
        d_dist = dres["distances"][0]
        dense_scores = {
            i: max(0.0, 1.0 - float(dist)) for i, dist in zip(d_ids, d_dist)
            if self._ids.index(i) in valid_indices
        }

        # BM25
        bm25_scores: dict[str, float] = {}
        if self._bm25 is not None:
            query_tokens = _tokenize_stem(query)
            if query_tokens:
                raw = np.asarray(self._bm25.get_scores(query_tokens), dtype=float)
                if raw.size:
                    mn, mx = float(raw.min()), float(raw.max())
                    norm = (raw - mn) / (mx - mn + 1e-9)
                    for idx in valid_indices:
                        if idx < len(self._ids):
                            bm25_scores[self._ids[idx]] = float(norm[idx])

        # Fuse
        keys = set(dense_scores) | set(bm25_scores)
        fused = {
            i: self.alpha * dense_scores.get(i, 0.0)
            + (1 - self.alpha) * bm25_scores.get(i, 0.0)
            for i in keys
        }
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        # Greedy knapsack pack under token budget
        id_to_idx = {i: j for j, i in enumerate(self._ids)}
        selected: list[dict] = []
        used = 0
        for doc_id, score in ordered:
            if len(selected) >= k:
                break
            idx = id_to_idx.get(doc_id)
            if idx is None:
                continue
            text = self._texts[idx]
            approx_tokens = _count_tokens(text)  # Use tiktoken instead of ÷4
            if used + approx_tokens > token_budget and selected:
                continue
            selected.append({
                "id": doc_id,
                "text": text,
                "meta": self._metas[idx] if idx < len(self._metas) else {},
                "score": score,
            })
            used += approx_tokens
        return selected

    # ---------------- introspection ----------------

    def iter_metadata(self) -> Iterator[dict]:
        yield from self._metas

    def __len__(self) -> int:
        return len(self._ids)

    def reset(self) -> None:
        """Nuke the collection and cache. Useful for tests."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.col = self.client.get_or_create_collection(self.collection_name)
        self._ids.clear()
        self._texts.clear()
        self._metas.clear()
        self._bm25 = None
        if self._bm25_path.exists():
            os.remove(self._bm25_path)
