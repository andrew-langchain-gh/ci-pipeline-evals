# CI Pipeline Evals

Demonstrates how to integrate **LangSmith offline evaluations** with a **GitHub Actions CI/CD pipeline**, using a quality gate that fails the build when an accuracy metric drops below a configurable threshold.

## How It Works

```
PR opened
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  evaluate job                                       │
│                                                     │
│  1. Run qa_assistant against a LangSmith dataset    │
│  2. LLM-as-judge scores each answer (1–10)          │
│  3. Compare average score against threshold         │
│  4. Exit non-zero if below threshold  ← GATE        │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  report job (runs even if evaluate fails)           │
│                                                     │
│  1. Query LangSmith for experiment scores           │
│  2. Generate markdown report                        │
│  3. Post as PR comment                              │
└─────────────────────────────────────────────────────┘
```

## Project Structure

```
├── main.py                          # Target application (Q&A assistant)
├── evals/
│   ├── create_dataset.py            # Seed the LangSmith dataset
│   ├── run_eval.py                  # Run eval + threshold gate
│   └── report_eval.py              # Generate markdown report
├── .github/
│   └── workflows/
│       └── evaluate.yml             # GitHub Actions workflow
└── pyproject.toml
```

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Set environment variables

```bash
export LANGSMITH_API_KEY="your-langsmith-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

### 3. Create the evaluation dataset (one-time)

```bash
uv run python evals/create_dataset.py
```

### 4. Run the evaluation locally

```bash
# Default threshold: 7/10
uv run python evals/run_eval.py

# Custom threshold
uv run python evals/run_eval.py --threshold 8

# Via environment variable
ACCURACY_THRESHOLD=8 uv run python evals/run_eval.py
```

### 5. Generate a report (optional)

```bash
uv run python evals/report_eval.py
```

## GitHub Actions Setup

Add these secrets to your repository (`Settings > Secrets and variables > Actions`):

| Secret | Description |
|--------|-------------|
| `LANGSMITH_API_KEY` | Your LangSmith API key |
| `OPENAI_API_KEY` | OpenAI API key (used by the target app and the LLM judge) |

The workflow runs on every PR to `main`. You can also trigger it manually via `workflow_dispatch` and pass a custom threshold.

## The Accuracy Evaluator

The evaluator uses an **LLM-as-judge** pattern (via [`openevals`](https://github.com/langchain-ai/openevals)):

- An LLM (GPT-4o-mini) reads the question, context, assistant answer, and reference answer
- It scores accuracy on a **1–10 scale** using a rubric:
  - **10**: Perfect — factually identical to reference
  - **7–9**: Mostly correct — minor omissions
  - **4–6**: Partially correct — missing key facts
  - **1–3**: Mostly wrong — significant errors
- The average score across all dataset examples is compared against the threshold
- If the average falls below the threshold, the CI pipeline **fails**

## Customization

- **Threshold**: Set `ACCURACY_THRESHOLD` env var or use `--threshold` CLI flag
- **Dataset**: Change `EVAL_DATASET_NAME` env var or edit `evals/create_dataset.py`
- **Model**: Swap `gpt-4o-mini` in `main.py` or the evaluator prompt in `run_eval.py`
- **Evaluators**: Add more evaluators to the `evaluators` list in `run_eval.py` and corresponding criteria in the config artifact
