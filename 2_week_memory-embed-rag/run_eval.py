import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import (DATA, RESULTS, MODELS, Ledger, LLMClient, EmbeddingCache, VectorStore, chunk_corpus, chunk_lines, emb_text,
                 load_jsonl, make_settings)
from rag.evaluation import evaluate
from rag.rag import RAG_SYSTEM_V2, AgentAnswerer, PlainAnswerer, RagAnswerer, Retriever, make_registry

CONFIGS = {
    "plain_sonnet": ("strong", "без инструментов"),
    "plain_cheap": ("cheap", "без инструментов"),
    "rag_cheap": ("cheap", "RAG всегда, k=5"),
    "agent_cheap": ("cheap", "агент с knowledge_base"),
    "rag_mid": ("mid", "RAG всегда, k=5"),
    "rag_cheap_v2": ("cheap", "RAG всегда, k=5, промпт v2"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(CONFIGS))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--collection", default="lines")
    args = ap.parse_args()

    settings = make_settings()
    ledger = Ledger()
    client = LLMClient(settings, ledger)
    tasks, none = load_jsonl("questions.jsonl"), load_jsonl("unanswerable.jsonl")
    if args.limit:
        tasks = tasks[:args.limit]
    cache = EmbeddingCache(client, DATA / "embeddings_cache.npz", seeds=(DATA / "embeddings_512.npz",))
    cache.load()
    retriever = None
    if any(name.startswith(("rag", "agent")) for name in args.only):
        store = VectorStore()
        if not store.has(args.collection):
            corpus = load_jsonl("corpus.jsonl")
            lines = chunk_corpus(corpus, chunk_lines)
            store.index(args.collection, lines, cache.embed([emb_text(c) for c in lines]))
            cache.save()
        retriever = Retriever(store, cache, args.collection, args.k)

    for name in args.only:
        tier, label = CONFIGS[name]
        model = MODELS[tier]
        if name.startswith("plain"):
            fn = PlainAnswerer(client)
        elif name.startswith("rag"):
            fn = RagAnswerer(client, retriever, k=args.k, **({"system": RAG_SYSTEM_V2} if name.endswith("_v2") else {}))
        else:
            fn = AgentAnswerer(client, make_registry(retriever, args.k))
        traces = []
        suffix = f"_limit{args.limit}" if args.limit else ""
        df = evaluate(fn, tasks, label, model, workers=args.workers, cache=RESULTS / f"{name}{suffix}.csv", traces=traces)
        dn = evaluate(fn, none, label, model, workers=args.workers, cache=RESULTS / f"{name}_unanswerable{suffix}.csv")
        if traces:
            (RESULTS / f"{name}_traces.jsonl").write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in traces), encoding="utf-8")
        print(f"{label:24s} {model:32s} верно {df['correct'].mean():5.0%}  отказов зря {int(df['refused'].sum()):3d}  "
              f"отклонено без ответа {int(dn['refused'].sum())}/{len(dn)}  цена ${df['cost'].sum():.4f}  потрачено ${ledger.total:.4f}", flush=True)
    cache.save()
    print(ledger.table())


if __name__ == "__main__":
    main()
