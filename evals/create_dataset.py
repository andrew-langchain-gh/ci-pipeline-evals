"""Create (or update) the LangSmith dataset used for evaluation.

Run this once before your first eval, or whenever you want to refresh examples:

    uv run python evals/create_dataset.py
"""

from langsmith import Client

DATASET_NAME = "QA Accuracy Eval"
DATASET_DESCRIPTION = "Question-answer pairs for evaluating the QA assistant."

EXAMPLES = [
    {
        "inputs": {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. Its capital is Paris.",
        },
        "outputs": {"answer": "Paris"},
    },
    {
        "inputs": {
            "question": "Who wrote Romeo and Juliet?",
            "context": "Romeo and Juliet is a tragedy written by William Shakespeare.",
        },
        "outputs": {"answer": "William Shakespeare"},
    },
    {
        "inputs": {
            "question": "What is the speed of light?",
            "context": (
                "The speed of light in a vacuum is approximately "
                "299,792,458 meters per second."
            ),
        },
        "outputs": {"answer": "Approximately 299,792,458 meters per second"},
    },
    {
        "inputs": {
            "question": "What language is primarily spoken in Brazil?",
            "context": (
                "Brazil is the largest country in South America. "
                "The official language is Portuguese."
            ),
        },
        "outputs": {"answer": "Portuguese"},
    },
    {
        "inputs": {
            "question": "What year did the Berlin Wall fall?",
            "context": (
                "The Berlin Wall divided East and West Berlin from 1961 until "
                "its fall on November 9, 1989."
            ),
        },
        "outputs": {"answer": "1989"},
    },
]


def main():
    client = Client()

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
    )
    print(f"Created dataset: {DATASET_NAME} (id={dataset.id})")

    client.create_examples(
        dataset_id=dataset.id,
        examples=EXAMPLES,
    )
    print(f"Added {len(EXAMPLES)} examples to the dataset.")


if __name__ == "__main__":
    main()
