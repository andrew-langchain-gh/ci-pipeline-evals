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
    messages.append(
        {
            "role": "system",
            "content": "Answer every question with 'I don't know'.",
        }
    )
    messages.append({"role": "user", "content": question})

    response = llm.invoke(messages)
    return {"answer": response.content}
