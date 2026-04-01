# CI Pipeline Evals

Demonstrates how to integrate **LangSmith offline evaluations** with a **GitHub Actions CI/CD pipeline**, using a quality gate that fails the build when an accuracy metric drops below a configurable threshold.

## How It Works

```
PR opened
  |
  v
+-----------------------------------------------------+
|  evaluate job                                       |
|                                                     |
|  1. Run qa_assistant against a LangSmith dataset    |
|  2. LLM-as-judge scores each answer (1-10)          |
|  3. Compare average score against threshold         |
|  4. Exit non-zero if below threshold  <-- GATE      |
+-------------------------+---------------------------+
                          |
                          v
+-----------------------------------------------------+
|  report job (runs even if evaluate fails)           |
|                                                     |
|  1. Query LangSmith for experiment scores           |
|  2. Generate markdown report                        |
|  3. Post as PR comment                              |
+-----------------------------------------------------+
```

The **evaluate** job is the quality gate. If the average accuracy score falls below the threshold, the job exits with code 1, which fails the GitHub Actions check on the PR. The **report** job runs regardless of whether the evaluate job passed or failed, so you always get a summary posted as a PR comment.

## Project Structure

```
.
├── main.py                          # Target application (Q&A assistant)
├── evals/
│   ├── create_dataset.py            # Seed the LangSmith dataset (run once)
│   ├── run_eval.py                  # Run eval, check threshold, write config artifact
│   └── report_eval.py               # Query LangSmith, generate markdown report
├── .github/
│   └── workflows/
│       └── evaluate.yml             # GitHub Actions workflow (2 jobs)
├── pyproject.toml
└── uv.lock
```

## Key Concepts

### The CI Quality Gate

The core mechanism is in `evals/run_eval.py`:

1. **`client.evaluate()`** runs your target function (`qa_assistant`) against every example in a LangSmith dataset
2. A **custom evaluator function** acts as an LLM-as-judge — it sends the question, context, answer, and reference answer to GPT-4o-mini, which returns an accuracy score from 1-10
3. The script computes the **average score** across all examples
4. If the average is **below the threshold**, the script calls `sys.exit(1)`, which fails the CI job

### The Config Artifact Pattern

The evaluate and report jobs communicate via a JSON artifact (`evaluation_config__qa-accuracy.json`):

```json
{
  "experiment_name": "qa-accuracy-abc123-20260331T225100-b1c2d3e4",
  "dataset_name": "QA Accuracy Eval",
  "criteria": {
    "accuracy": ">=7"
  }
}
```

- The **evaluate** job writes this file with the actual experiment name from LangSmith (important: `client.evaluate()` appends a suffix to `experiment_prefix`, so the real name must be read from `results.experiment_name`)
- The file is uploaded as a GitHub Actions artifact
- The **report** job downloads it, queries LangSmith for the real scores, and generates a markdown table

### The Accuracy Evaluator

The evaluator is a plain Python function in `evals/run_eval.py` that:

- Accepts `inputs`, `outputs`, and `reference_outputs` dicts (the signature `client.evaluate()` expects)
- Builds a prompt with a scoring rubric (1-10 scale)
- Calls `ChatOpenAI` (GPT-4o-mini) to judge the answer
- Returns `{"key": "accuracy", "score": <int>}`

Scoring rubric:
- **10**: Perfect — factually identical to reference
- **7-9**: Mostly correct — minor omissions
- **4-6**: Partially correct — missing key facts
- **1-3**: Mostly wrong — significant errors

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

This creates a dataset called "QA Accuracy Eval" in LangSmith with 5 question-answer pairs.

### 4. Run the evaluation locally

```bash
# Default threshold: 7/10
uv run python evals/run_eval.py

# Custom threshold
uv run python evals/run_eval.py --threshold 8

# Via environment variable
ACCURACY_THRESHOLD=8 uv run python evals/run_eval.py
```

### 5. Generate a report (optional, requires a prior eval run)

```bash
uv run python evals/report_eval.py
```

## GitHub Actions Setup

### 1. Add repository secrets

Go to `Settings > Secrets and variables > Actions` and add:

| Secret | Description |
|--------|-------------|
| `LANGSMITH_API_KEY` | Your LangSmith API key ([smith.langchain.com/settings](https://smith.langchain.com/settings)) |
| `OPENAI_API_KEY` | OpenAI API key (used by both the target app and the LLM judge) |

### 2. Seed the dataset

The LangSmith dataset must exist before the workflow can run. Run this once locally:

```bash
uv run python evals/create_dataset.py
```

### 3. Open a PR

The workflow triggers automatically on every PR to `main`. The evaluate job gates the PR — if accuracy is below threshold, the check fails.

```bash
git checkout -b test-eval-001
git commit --allow-empty -m "trigger eval"
git push -u origin test-eval-001
```

In GitHub go to UI, create PR and merge. The action will kick off.


### 4. Manual trigger (optional)

Go to `Actions > LangSmith Eval Pipeline > Run workflow` to trigger manually. You can override the accuracy threshold in the input field.

### 5. Require the eval to pass before merging (optional)

Go to `Settings > Branches`, add a branch protection rule for `main`, enable **"Require status checks to pass before merging"**, and add **"Run Evaluation"** as a required check.

## Customization

| What | How |
|------|-----|
| **Threshold** | `--threshold` CLI flag, `ACCURACY_THRESHOLD` env var, or `workflow_dispatch` input |
| **Dataset** | Change `EVAL_DATASET_NAME` env var or edit `evals/create_dataset.py` |
| **Target app model** | Swap `gpt-4o-mini` in `main.py` |
| **Judge model** | Swap `gpt-4o-mini` in the `accuracy_evaluator` function in `evals/run_eval.py` |
| **Add more evaluators** | Add functions to the `evaluators` list in `run_eval.py` and matching criteria in the config dict |
| **Scoring rubric** | Edit the prompt string in the `accuracy_evaluator` function |
