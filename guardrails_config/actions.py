from typing import Optional

from nemoguardrails.actions import action

from ragas import evaluate
from ragas.metrics import context_recall

from datasets import Dataset

def get_context_recall(eval_dataset) -> float:
    eval_dataset = Dataset.from_dict(eval_dataset)
    result = evaluate(
        eval_dataset,
        metrics=[context_recall],
        raise_exceptions=False,
    )

    return result["context_recall"]

@action(is_system_action=True)
async def check_context_recall(context: Optional[dict] = None, threshold: float = 0.5) -> bool:
    question = context.get("question")
    answer = context.get("answer")
    contexts = context.get("contexts")
    ground_truth = context.get("ground_truth")

    eval_dataset = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth]
    }

    context_recall = get_context_recall(eval_dataset)

    if context_recall < threshold:
        return True

    return False