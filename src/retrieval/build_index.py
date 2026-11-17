"""Build the runbook index.

  python src/retrieval/build_index.py                                  # keyword indexes only (no model download)
  python src/retrieval/build_index.py --embedder fastembed             # adds dense vectors, small ONNX model, no torch
  python src/retrieval/build_index.py --embedder st                    # adds dense vectors with sentence-transformers
  python src/retrieval/build_index.py --embedder fastembed --chroma    # also writes a Chroma vector database

Writes data/index/chunks.jsonl, meta.json and (with an embedder) dense.npy.
BM25 and TF-IDF are rebuilt in memory when the index is loaded, which takes well under a second for about 120 chunks.
The meta file records the runbook fingerprint, embedder and model, so an old index is noticed (see evaluate_retrieval.py).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from chunking import build_chunks, runbook_hash
from retriever import Embedder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def export_chroma(index_dir, chroma_dir):
    """Optional: copy the chunks and vectors into a persistent Chroma collection (pip install chromadb)."""
    import chromadb
    with open(os.path.join(index_dir, "chunks.jsonl"), encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    vecs = np.load(os.path.join(index_dir, "dense.npy"))
    client = chromadb.PersistentClient(path=chroma_dir)
    try:
        client.delete_collection("runbooks")
    except Exception:
        pass
    col = client.create_collection("runbooks", metadata={"hnsw:space": "cosine"})
    metas = [dict(runbook_id=c["runbook_id"], section=c["section"], category=c["category"], owner=c["owner"],
                  title=c["title"], auto_resolvable=c["auto_resolvable"],
                  allowed_actions=",".join(c["allowed_actions"])) for c in chunks]     # Chroma metadata must be flat
    col.add(ids=[c["chunk_id"] for c in chunks], embeddings=vecs.tolist(),
            documents=[c["text"] for c in chunks], metadatas=metas)
    return col.count()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runbooks", default=f"{ROOT}/data/runbooks")
    ap.add_argument("--out", default=f"{ROOT}/data/index")
    ap.add_argument("--embedder", choices=["none", "fastembed", "st", "hash"], default="none",
                    help="hash is a no-download stand-in for testing only")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--chroma", action="store_true", help="also write data/index/chroma (needs --embedder)")
    a = ap.parse_args()

    chunks = build_chunks(a.runbooks)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), n_chunks=len(chunks),
                n_runbooks=len({c["runbook_id"] for c in chunks}), runbook_hash=runbook_hash(a.runbooks),
                embedder=a.embedder if a.embedder != "none" else None,
                model=a.model if a.embedder in ("fastembed", "st") else (a.embedder if a.embedder == "hash" else None))
    dense_path = os.path.join(a.out, "dense.npy")
    if a.embedder == "none":
        if os.path.exists(dense_path):
            os.remove(dense_path)          # do not leave vectors from an older build next to a keyword-only meta file
    else:
        emb = Embedder(a.embedder, a.model)
        t0 = time.time()
        vecs = emb.passages([c["text"] for c in chunks])
        np.save(dense_path, vecs)
        meta.update(dim=int(vecs.shape[1]), embed_seconds=round(time.time() - t0, 2))
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"indexed {meta['n_chunks']} chunks from {meta['n_runbooks']} runbooks into {a.out}")
    print(f"embedder: {meta['embedder'] or 'none (keyword search only)'}  runbook fingerprint: {meta['runbook_hash']}")
    if a.chroma:
        if a.embedder == "none":
            sys.exit("--chroma needs --embedder, because Chroma stores the vectors")
        n = export_chroma(a.out, os.path.join(a.out, "chroma"))
        print(f"chroma collection 'runbooks' written with {n} items to {os.path.join(a.out, 'chroma')}")


if __name__ == "__main__":
    main()
