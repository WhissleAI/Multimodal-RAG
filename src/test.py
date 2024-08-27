from nemoguardrails import RailsConfig, LLMRails
from datasets import Dataset

config = RailsConfig.from_path("../guardrails_config")
rails = LLMRails(config)

dataset = Dataset.from_dict({"questions": ["How many vacation days do I have per year?"],
                             "answers": ["You have 25 vacation days per year."],
                             "contexts": ["You have 25 vacation days per year."],
                             "ground_truths": ["You have 25 vacation days per year."]})


response = rails.generate(messages=[{
    "role": "context",
    "content": {
        "eval_dataset": dataset
    }},
    {"role": "user",
    "content": "How many vacation days do I have per year?"
    }
])
print(response["content"])