# AI Cooking Agent for College Students

## 1. Abstract
College students often struggle to decide what to cook based on limited time, budget, and dietary needs. 
We build an AI cooking agent that recommends recipes and explains why each recommendation fits a user’s constraints.
The agent uses TF-IDF retrieval over a curated recipe collection and a small Hugging Face model to generate search actions 
and final responses. We evaluate retrieval accuracy and end-to-end usefulness using simple quantitative and human-judged metrics.
Our results show that constraint-aware prompting improves the relevance of recommended recipes compared to a retrieval-only baseline.

## 2. Overview

### 2.1 What is the problem?
Many students want meals that are cheap, fast, and compatible with dietary restrictions. 
Searching the internet or scrolling through recipes is time-consuming and often returns options that ignore constraints.

### 2.2 Why is this problem interesting?
A reliable cooking agent can reduce decision fatigue, help students eat healthier, and support budget-conscious planning.
This is also a good real-world testbed for agentic information retrieval: the system must interpret user intent, retrieve 
relevant documents, and produce grounded recommendations.

### 2.3 What approach do we propose?
We propose a lightweight retrieval-augmented agent:
1) Convert recipes into short “document” entries.
2) Use TF-IDF to retrieve the most relevant recipes given a query.
3) Use an open-source language model to:
   - generate a structured search action
   - produce a final recommendation grounded in the retrieved recipes.

### 2.4 Rationale and related work
Retrieval-augmented generation is a common approach for grounding language model outputs in a limited knowledge base.
Instead of relying on large external datasets, we focus on a small curated recipe set to emphasize controllability and reproducibility.
Our design prioritizes simplicity and interpretability for the course project setting.

### 2.5 Key components and limitations
**Key components**
- Recipe document collection
- TF-IDF retrieval
- Prompting templates for search-action generation
- A Hugging Face LLM wrapper
- An agent class that iteratively retrieves and answers

**Limitations**
- A small recipe database limits coverage.
- Cost estimates are simplified into rough categories.
- The small LLM may produce imperfect search actions without careful prompting.

## 3. Approach

### 3.1 Overall methodology
Our agent follows this workflow:
1. Read a user request (e.g., “cheap vegan dinner under 20 minutes”).
2. Use the LLM to generate a short structured search query.
3. Retrieve top-k recipes using TF-IDF.
4. Use the LLM again to generate a final recommendation and justification that references retrieved recipes.

### 3.2 Algorithm/model/method
- **Retrieval**: TF-IDF + cosine similarity.
- **Generation**: a small open-source Hugging Face model (default set in code).
- **Agent loop**: single-step retrieval is used by default.

### 3.3 Assumptions/design choices
- Each recipe is represented as a short text document containing title, ingredients, tags, time, cost level, and dietary labels.
- We label cost coarsely (low/medium/high) to keep the dataset lightweight.
- We favor small LLMs for accessibility and reproducibility.

### 3.4 Limitations of setup
- Our dataset is curated and not exhaustive.
- The evaluation queries are small and manually labeled.

## 4. Experiments

### 4.1 Dataset
We create a small curated recipe collection aimed at common college constraints:
- quick meals
- budget-friendly options
- vegan/vegetarian/gluten-free variants
- simple ingredient lists

We also create a small evaluation set of user-like queries with ground-truth relevant recipe IDs.

### 4.2 Implementation details
- Python
- NumPy/Pandas
- scikit-learn for TF-IDF
- Hugging Face Transformers for model loading and generation

### 4.3 Model architecture
We use pretrained transformer models via Hugging Face without training new weights.

## 5. Results

### 5.1 Main Results

We evaluate our system at two levels:

1) **Retrieval performance** using TF-IDF over the curated recipe collection.  
2) **End-to-end agent quality** using a small human evaluation rubric focused on constraint adherence and usefulness.

#### Retrieval Metrics

| Method | Precision@5 | Recall@5 |
|--------|-------------|----------|
| TF-IDF baseline | 0.522 | 0.623 |
| Agent search-action prompting |  |  |

**Notes on interpretation**
- The TF-IDF baseline measures how well keyword-based retrieval matches user intent.
- The agent variant tests whether structured search-action prompting improves the relevance of retrieved recipes.

#### End-to-End Human Evaluation (Small-Scale)

We evaluate final responses on a small set of user-like queries.

| Criterion | Description | Score Scale |
|----------|-------------|------------|
| Constraint adherence | Respects budget, time, and dietary requirements | 1–5 |
| Grounding | Recommendation matches retrieved recipes | 1–5 |
| Helpfulness | Clear, practical, student-friendly suggestion | 1–5 |

| System Version | Constraint Adherence | Grounding | Helpfulness |
|---------------|----------------------|-----------|-------------|
| Retrieval-only baseline |  |  |  |
| Full agent (retrieval + LLM prompting) |  |  |  |

### 5.2 Supplementary Results

We also examine:
- Performance differences across constraint types (budget vs. dietary vs. time).
- Sensitivity to `top_k` retrieval settings.
- A small comparison of prompt formats for search-action generation.

| Setting | Observation |
|--------|-------------|
| top_k = 3 |  |
| top_k = 5 |  |
| top_k = 10 |  |


## 6. Discussion
We discuss why the agent succeeds or fails given the current recipe coverage and prompting formats.
Future improvements could include expanding the recipe dataset and adding multi-step meal planning.

## 7. Conclusion
We built a lightweight AI cooking agent for college students using TF-IDF retrieval and an open-source LLM.
Although limited by dataset size, the system demonstrates a reproducible pipeline for agentic information retrieval.

## 8. How to Run

### 8.1 Install
```bash
pip install -r requirements.txt
```

### 8.2 Run a quick demo
```bash
python -m src.agent
```

### 8.3 Evaluate retrieval
```bash
python -m src.evaluate
```

## 9. Repository Structure
- `src/` for source code
- `data/` for lightweight datasets
- `results/` for figures/logs
- `notebooks/` for optional demos
- `requirements.txt` for environment setup

## 10. References
- scikit-learn documentation (TF-IDF)
- Hugging Face Transformers documentation
- Any recipe sources used to curate the dataset (if applicable)
