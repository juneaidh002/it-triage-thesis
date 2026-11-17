"""Search the runbook index.

  from retriever import Retriever
  r = Retriever("data/index")
  r.search("vpn wont connect from home", k=5)                    # top chunks, for the agent's context
  r.rank_runbooks("vpn wont connect from home", k=3)             # top runbooks, for Recall@k
  r.search(q, exclude_runbooks=["RB-VPN-01"])                    # evidence-removal test

Methods:
  bm25    keyword search
  tfidf   keyword search with word pairs (a second lexical baseline, no model download)
  dense   sentence-embedding search (needs an index built with --embedder)
  hybrid  rank fusion (RRF) of bm25 and dense; if there is no dense index, of bm25 and tfidf

Try it:  python src/retrieval/retriever.py "printer queue stuck" --k 3
"""
import argparse
import json
import os
import re

import numpy as np

STOP = set("a an and are as at be but by can for from has have i if in is it its me my of on or our so that "
           "the this to was we with you your".split())
RRF_K = 60


def tokens(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]


def make_query(subject, body, mode="plain", min_body_words=4):
    """Build the search text from a ticket.
    plain       subject + body
    body_first  the body alone when it has enough words (a poor subject then cannot mislead the search),
                otherwise subject + body (the subject carries the problem when the body is thin or empty)
    """
    subject, body = (subject or "").strip(), (body or "").strip()
    if mode == "body_first" and len(body.split()) >= min_body_words:
        return body
    return f"{subject} {body}".strip()


class Embedder:
    """Wraps the embedding backends. `hash` needs no download and is for testing only."""

    def __init__(self, backend, model):
        self.backend, self.model = backend, model
        if backend == "fastembed":
            from fastembed import TextEmbedding
            self._m = TextEmbedding(model_name=model)
        elif backend == "st":
            from sentence_transformers import SentenceTransformer
            self._m = SentenceTransformer(model)
        elif backend == "hash":
            from sklearn.feature_extraction.text import HashingVectorizer
            self._m = HashingVectorizer(n_features=1024, ngram_range=(1, 2), alternate_sign=False, norm=None)
        else:
            raise ValueError(f"unknown embedder backend: {backend}")

    @staticmethod
    def _norm(x):
        x = np.asarray(x, dtype="float32")
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.where(n == 0, 1, n)

    def passages(self, texts):
        if self.backend == "fastembed":
            return self._norm(list(self._m.passage_embed(texts)))
        if self.backend == "st":
            return self._norm(self._m.encode(texts, normalize_embeddings=True))
        return self._norm(self._m.transform(texts).toarray())

    def queries(self, texts):
        if self.backend == "fastembed":
            return self._norm(list(self._m.query_embed(texts)))
        if self.backend == "st":
            pre = "Represent this sentence for searching relevant passages: " if "bge" in self.model.lower() else ""
            return self._norm(self._m.encode([pre + t for t in texts], normalize_embeddings=True))
        return self._norm(self._m.transform(texts).toarray())


class Retriever:
    def __init__(self, index_dir="data/index"):
        self.dir = index_dir
        with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]
        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        from rank_bm25 import BM25Okapi
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [c["text"] for c in self.chunks]
        self._bm25 = BM25Okapi([tokens(t) for t in texts])
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._tfidf_matrix = self._tfidf.fit_transform(texts)
        dense_path = os.path.join(index_dir, "dense.npy")
        self._dense = np.load(dense_path) if os.path.exists(dense_path) else None
        self._embedder = None
        self.has_dense = self._dense is not None
        self.methods = ["bm25", "tfidf"] + (["dense"] if self.has_dense else []) + ["hybrid"]
        self.hybrid_parts = ["bm25", "dense"] if self.has_dense else ["bm25", "tfidf"]
        self._rid = np.array([c["runbook_id"] for c in self.chunks])
        self._sec = np.array([c["section"] for c in self.chunks])

    # ---- scoring -------------------------------------------------------------------------------
    def _query_embedding(self, query):
        if self._embedder is None:
            self._embedder = Embedder(self.meta["embedder"], self.meta["model"])
        return self._embedder.queries([query])[0]

    def raw_scores(self, method, query):
        """Score of every chunk for one method (not fused)."""
        if method == "bm25":
            return np.asarray(self._bm25.get_scores(tokens(query)), dtype=float)
        if method == "tfidf":
            return (self._tfidf_matrix @ self._tfidf.transform([query]).T).toarray().ravel()
        if method == "dense":
            if not self.has_dense:
                raise RuntimeError("this index has no dense vectors; rebuild with --embedder")
            return self._dense @ self._query_embedding(query)
        raise ValueError(f"unknown method {method}")

    def _allowed(self, exclude_runbooks, sections):
        ok = ~np.isin(self._rid, list(exclude_runbooks or ()))
        if sections:
            ok &= np.isin(self._sec, list(sections))
        return ok

    def scores(self, method, query, exclude_runbooks=(), sections=None):
        """Scores for ranking. Not allowed chunks get -inf. Hybrid uses rank fusion, so its scores are rank based."""
        ok = self._allowed(exclude_runbooks, sections)
        if method != "hybrid":
            s = self.raw_scores(method, query)
            s = np.where(ok, s, -np.inf)
            if method != "dense":
                s = np.where(s > 0, s, -np.inf)       # a keyword score of 0 means "no match", not a weak match
            return s
        fused = np.zeros(len(self.chunks))
        hit = np.zeros(len(self.chunks), dtype=bool)
        for m in self.hybrid_parts:
            s = self.scores(m, query, exclude_runbooks, sections)
            order = [i for i in np.argsort(-s) if np.isfinite(s[i])]
            for rank, i in enumerate(order, 1):
                fused[i] += 1.0 / (RRF_K + rank)
                hit[i] = True
        return np.where(hit & ok, fused, -np.inf)

    # ---- public search -------------------------------------------------------------------------
    def search(self, query, k=5, method="hybrid", exclude_runbooks=(), sections=None):
        s = self.scores(method, query, exclude_runbooks, sections)
        order = [i for i in np.argsort(-s) if np.isfinite(s[i])][:k]
        return [dict(score=float(s[i]), **self.chunks[i]) for i in order]

    def rank_runbooks(self, query, k=5, method="hybrid", exclude_runbooks=()):
        """Top runbooks: each runbook is ranked by its best chunk."""
        s = self.scores(method, query, exclude_runbooks)
        seen, out = set(), []
        for i in np.argsort(-s):
            if not np.isfinite(s[i]):
                break
            rid = self.chunks[i]["runbook_id"]
            if rid not in seen:
                seen.add(rid)
                out.append((rid, float(s[i])))
                if len(out) == k:
                    break
        return out

    def top_raw_score(self, method, query):
        """Best raw score of one method; used to see whether a ticket matches ANY runbook well."""
        return float(np.max(self.raw_scores(method, query)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--index", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "index"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--method", default="hybrid")
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()
    r = Retriever(a.index)
    print(f"methods available: {r.methods}\n")
    for h in r.search(a.query, k=a.k, method=a.method, exclude_runbooks=a.exclude):
        print(f"{h['score']:.4f}  {h['chunk_id']}  [{h['category']}]\n    {h['text'].splitlines()[-1][:110]}")
