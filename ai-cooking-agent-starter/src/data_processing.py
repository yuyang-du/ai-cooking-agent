import json
from pathlib import Path
from typing import List, Dict, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_recipes(path: Path = DATA_DIR / "recipes.json") -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def recipe_to_document(recipe: Dict) -> str:
    title = recipe.get("title", "")
    ingredients = ", ".join(recipe.get("ingredients", []))
    tags = ", ".join(recipe.get("tags", []))
    dietary = ", ".join(recipe.get("dietary", []))
    time_m = recipe.get("time_minutes", "unknown")
    cost = recipe.get("cost_level", "unknown")

    return (
        f"{title}. "
        f"Ingredients: {ingredients}. "
        f"Tags: {tags}. "
        f"Time: {time_m} minutes. "
        f"Cost: {cost}. "
        f"Dietary: {dietary}."
    )

def build_corpus(recipes: List[Dict]) -> Tuple[List[str], List[Dict]]:
    docs = [recipe_to_document(r) for r in recipes]
    meta = [{"id": r.get("id"), "title": r.get("title")} for r in recipes]
    return docs, meta
