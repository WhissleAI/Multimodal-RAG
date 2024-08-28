from nemoguardrails import RailsConfig, LLMRails


config = RailsConfig.from_path("/home/yfg2/Multimodal-RAG/guardrails_config")
rails = LLMRails(config)




response = rails.generate(messages=[{
    "role": "context",
    "content": {
        "question": "How many vacation days do I have per year?",
        "answer": "You have 25 vacation days per year.",
        "contexts": ["You have 25 vacation days per year."],
        "ground_truth": "You have 25 vacation days per year."
    }},
    {"role": "user",
     "content": "How many vacation days do I have per year?"
    }
])
import pdb; pdb.set_trace()
print(response["content"])