from typing import Dict
from .data_processing import load_recipes, build_corpus
from .search import TfidfSearch
from .prompts import build_search_action_prompt, build_final_answer_prompt
from .llm_utils import LLM, LLMConfig

class CookingAgent:
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.recipes = load_recipes()
        docs, meta = build_corpus(self.recipes)

        self.searcher = TfidfSearch()
        self.searcher.fit(docs, meta)

        self.llm = LLM(LLMConfig(model_name=model_name))

    def _make_user_request(self, query: str, constraints: Dict) -> str:
        parts = [query.strip()]
        if constraints:
            parts.append("Constraints:")
            for k, v in constraints.items():
                parts.append(f"{k}={v}")
        return " ".join(parts)

    def recommend(self, query: str, constraints: Dict = None, top_k: int = 5) -> str:
        constraints = constraints or {}
        user_request = self._make_user_request(query, constraints)

        action_prompt = build_search_action_prompt(user_request)
        search_query = self.llm.generate(action_prompt)

        retrieved = self.searcher.search(search_query, top_k=top_k)

        final_prompt = build_final_answer_prompt(user_request, retrieved)
        answer = self.llm.generate(final_prompt)
        return answer

def main():
    agent = CookingAgent()
    query = "cheap vegan dinner under 20 minutes"
    constraints = {"budget": "low", "diet": "vegan", "time_minutes_max": 20}
    _ = agent.recommend(query, constraints=constraints)

if __name__ == "__main__":
    main()
