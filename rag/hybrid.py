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
        tokens = [t.lower().split() for t in self._texts]
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
                self._bm25 = BM25Okapi([t.lower().split() for t in self._texts])

    # ---------------- retrieval ----------------

    def retrieve(
        self,
        query: str,
        k: int = 6,
        m: int = 30,
        token_budget: int = 3500,
    ) -> list[dict]:
        if not self._ids:
            return []

        m = min(m, len(self._ids))

        # Dense
        qv = self.emb.encode([query], normalize_embeddings=True).tolist()
        dres = self.col.query(query_embeddings=qv, n_results=m)
        # chroma returns distances in cosine distance space (1 - cos)
        d_ids = dres["ids"][0]
        d_dist = dres["distances"][0]
        dense_scores = {
            i: max(0.0, 1.0 - float(dist)) for i, dist in zip(d_ids, d_dist)
        }

        # BM25
        bm25_scores: dict[str, float] = {}
        if self._bm25 is not None:
            raw = np.asarray(self._bm25.get_scores(query.lower().split()), dtype=float)
            if raw.size:
                mn, mx = float(raw.min()), float(raw.max())
                norm = (raw - mn) / (mx - mn + 1e-9)
                top_bm = np.argsort(-raw)[:m]
                for idx in top_bm:
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
            approx_tokens = max(1, len(text) // 4)  # rough tok estimate
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
