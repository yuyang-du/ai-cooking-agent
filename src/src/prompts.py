from typing import List, Tuple

def build_search_action_prompt(user_request: str) -> str:
    return (
        "You are a helpful cooking assistant.\n"
        "Given the user's request, produce a short search query string.\n"
        "Only output the search query.\n\n"
        f"User request: {user_request}\n"
        "Search query:"
    )

def build_final_answer_prompt(user_request: str, retrieved_docs: List[Tuple[dict, float, str]]) -> str:
    context = "\n".join(
        [f"- {meta['title']} (score={score:.3f}): {doc}"
         for meta, score, doc in retrieved_docs]
    )

    return (
        "You are an AI cooking agent for college students.\n"
        "You must recommend 1-2 recipes that best match the user's constraints.\n"
        "Use ONLY the recipes in the context below.\n"
        "Explain briefly why each choice fits budget/time/dietary constraints.\n\n"
        f"User request: {user_request}\n\n"
        f"Context recipes:\n{context}\n\n"
        "Answer:"
    )
