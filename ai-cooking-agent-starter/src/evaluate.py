import json
from pathlib import Path
from typing import List, Dict
from .data_processing import load_recipes, build_corpus
from .search import TfidfSearch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_eval_queries(path: Path = DATA_DIR / "eval_queries.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def precision_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    pred_k = pred_ids[:k]
    if not pred_k:
        return 0.0
    hit = sum(1 for pid in pred_k if pid in gold_ids)
    return hit / len(pred_k)

def recall_at_k(pred_ids: List[str], gold_ids: List[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    pred_k = set(pred_ids[:k])
    hit = sum(1 for gid in gold_ids if gid in pred_k)
    return hit / len(gold_ids)

def main(top_k: int = 5):
    recipes = load_recipes()
    docs, meta = build_corpus(recipes)

    idx = TfidfSearch()
    idx.fit(docs, meta)

    eval_items = load_eval_queries()

    p_scores = []
    r_scores = []

    for item in eval_items:
        query = item["query"]
        gold = item["relevant_recipe_ids"]
        results = idx.search(query, top_k=top_k)
        pred_ids = [m["id"] for m, _, _ in results]

        p_scores.append(precision_at_k(pred_ids, gold, k=top_k))
        r_scores.append(recall_at_k(pred_ids, gold, k=top_k))

    print(f"Precision@{top_k}: {sum(p_scores)/len(p_scores):.3f}")
    print(f"Recall@{top_k}:    {sum(r_scores)/len(r_scores):.3f}")

if __name__ == "__main__":
    main()
