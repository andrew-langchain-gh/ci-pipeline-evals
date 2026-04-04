"""Run offline evaluation and gate CI on accuracy threshold.

Usage:
    uv run python evals/run_eval.py                     # default threshold: 7
    uv run python evals/run_eval.py --threshold 8       # custom threshold
    ACCURACY_THRESHOLD=8 uv run python evals/run_eval.py  # via env var

Exit code 0 = pass, 1 = fail (accuracy below threshold).
"""

import json
import os
import sys
import time
import uuid

# Ensure the project root is on the Python path so `main` is importable
# regardless of which directory the script is invoked from.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI
from langsmith import Client

from main import qa_assistant

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_NAME = os.environ.get("EVAL_DATASET_NAME", "QA Accuracy Eval")
EXPERIMENT_PREFIX = os.environ.get("EVAL_EXPERIMENT_PREFIX", "qa-accuracy")
ACCURACY_THRESHOLD = float(os.environ.get("ACCURACY_THRESHOLD", "7"))
ANNOTATION_QUEUE_NAME = os.environ.get(
    "ANNOTATION_QUEUE_NAME", "Low Accuracy Reviews"
)


def accuracy_evaluator(
    inputs: dict, outputs: dict, reference_outputs: dict
) -> dict:
    """LLM-as-judge evaluator that scores accuracy on a 1-10 scale."""
    prompt = f"""\
You are an expert evaluator. Given a QUESTION, the CONTEXT provided to the \
assistant, the assistant's ANSWER, and the REFERENCE answer, score the \
assistant's ANSWER for accuracy on a scale of 1 to 10.

Scoring rubric:
- 10: Perfect — factually identical to the reference.
- 7-9: Mostly correct — captures the key facts, minor omissions or extra detail.
- 4-6: Partially correct — some relevant information but missing key facts or \
contains inaccuracies.
- 1-3: Mostly wrong — significant factual errors or largely irrelevant.

<question>
{inputs["question"]}
</question>

<context>
{inputs.get("context", "")}
</context>

<answer>
{outputs["answer"]}
</answer>

<reference_answer>
{reference_outputs["answer"]}
</reference_answer>

Return ONLY a JSON object with a single key "score" whose value is an integer \
from 1 to 10. Example: {{"score": 8}}"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([{"role": "user", "content": prompt}])
    score = json.loads(response.content)["score"]
    return {"key": "accuracy", "score": int(score)}


def _parse_threshold() -> float:
    """CLI --threshold flag takes precedence over env var."""
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg == "--threshold" and i + 1 < len(args):
            return float(args[i + 1])
    return ACCURACY_THRESHOLD


def _wait_for_runs(
    client: Client, run_ids: list, timeout: int = 30, poll_interval: float = 2
):
    """Poll until all runs are available server-side.

    After client.evaluate(), runs may still be ingesting asynchronously.
    The annotation queue API returns 404 if the runs aren't persisted yet.
    """
    pending = set(str(r) for r in run_ids)
    deadline = time.time() + timeout
    while pending and time.time() < deadline:
        still_pending = set()
        for rid in pending:
            try:
                client.read_run(rid)
            except Exception:
                still_pending.add(rid)
        pending = still_pending
        if pending:
            time.sleep(poll_interval)
    if pending:
        print(f"WARNING: {len(pending)} run(s) not available after {timeout}s")


def _get_or_create_annotation_queue(client: Client, name: str):
    """Return an existing annotation queue by name, or create a new one."""
    queues = list(client.list_annotation_queues(name=name, limit=1))
    if queues:
        return queues[0]
    return client.create_annotation_queue(
        name=name,
        description="Automatically populated with eval runs that scored below threshold.",
    )


def main():
    threshold = _parse_threshold()
    client = Client()

    experiment_name = f"{EXPERIMENT_PREFIX}-{uuid.uuid4().hex[:8]}"
    print(f"Dataset:    {DATASET_NAME}")
    print(f"Experiment: {experiment_name}")
    print(f"Threshold:  {threshold}/10")
    print()

    # -- Run evaluation ------------------------------------------------------
    results = client.evaluate(
        qa_assistant,
        data=DATASET_NAME,
        evaluators=[accuracy_evaluator],
        experiment_prefix=experiment_name,
        max_concurrency=4,
        blocking=True,
    )

    # -- Capture the real experiment name (LangSmith appends a suffix) --------
    actual_experiment_name = results.experiment_name
    print(f"LangSmith experiment: {actual_experiment_name}")

    # -- Check each result against threshold ----------------------------------
    failing_run_ids = []
    all_scores = []

    for result in results:
        run_id = result["run"].id
        for eval_result in result["evaluation_results"]["results"]:
            if eval_result.key == "accuracy" and eval_result.score is not None:
                score = eval_result.score
                all_scores.append(score)
                if score < threshold:
                    failing_run_ids.append(run_id)
                    print(f"  BELOW THRESHOLD: run {run_id} scored {score}/10")

    if not all_scores:
        print("ERROR: No accuracy scores were produced.")
        sys.exit(1)

    avg_score = sum(all_scores) / len(all_scores)
    print(f"\nResults: {len(all_scores)} examples evaluated")
    print(f"Average accuracy: {avg_score:.2f}/10")
    print(f"Below threshold:  {len(failing_run_ids)}/{len(all_scores)}")
    print()

    # -- Add failing runs to annotation queue --------------------------------
    if failing_run_ids:
        client.flush()
        _wait_for_runs(client, failing_run_ids)
        queue = _get_or_create_annotation_queue(client, ANNOTATION_QUEUE_NAME)
        client.add_runs_to_annotation_queue(
            queue_id=queue.id,
            run_ids=failing_run_ids,
        )
        print(
            f"Added {len(failing_run_ids)} failing run(s) to annotation queue "
            f"'{ANNOTATION_QUEUE_NAME}'"
        )

    # -- Write config artifact for the report job ----------------------------
    config = {
        "experiment_name": actual_experiment_name,
        "dataset_name": DATASET_NAME,
        "criteria": {
            "accuracy": f">={threshold}",
        },
    }
    config_path = f"evaluation_config__{EXPERIMENT_PREFIX}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Wrote config artifact: {config_path}")

    # -- Gate ----------------------------------------------------------------
    if failing_run_ids:
        print(
            f"\nFAILED: {len(failing_run_ids)} result(s) scored below "
            f"threshold {threshold}/10"
        )
        sys.exit(1)

    print(f"\nPASSED: All results meet threshold {threshold}/10")


if __name__ == "__main__":
    main()
