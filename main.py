"""Sample application: a simple Q&A assistant.

This is the "target" function that LangSmith evaluates.
Replace this with your real application logic.
"""

from langchain_openai import ChatOpenAI


def qa_assistant(inputs: dict) -> dict:
    """Answer a question given optional context.

    This function signature (dict -> dict) is what langsmith.evaluate() expects.
    The keys in `inputs` match the columns of your LangSmith dataset.
    """
    question = inputs["question"]
    context = inputs.get("context", "")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = []
    if context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the provided context. "
                    "Be concise and accurate.\n\n"
                    f"Context:\n{context}"
                ),
            }
        )
    else:
        messages.append(
            {
                "role": "system",
                "content": "Answer the user's question concisely and accurately.",
            }
        )
    messages.append({"role": "user", "content": question})

    response = llm.invoke(messages)
    return {"answer": response.content}
