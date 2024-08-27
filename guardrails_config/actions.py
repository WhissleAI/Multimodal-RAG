from typing import Optional

from nemoguardrails.actions import action

from ragas import evaluate

def get_context_recall(eval_dataset) -> float:
    result = evaluate(
        eval_dataset,
        metrics=["context_recall"],
        raise_exceptions=False,
    )
    return result["context_recall"]

@action(is_system_action=True)
async def check_context_recall(context: Optional[dict] = None, threshold: float = 0.5) -> bool:
    eval_dataset = context.get("eval_dataset")

    context_recall = get_context_recall(eval_dataset)

    if context_recall < threshold:
        return True

    return False