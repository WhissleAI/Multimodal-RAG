# import os
# import json
# from utils import experiment_init
# import torch
# import yaml
# from torch.utils.data import DataLoader
# from datasets import Dataset
# import huggingface_hub
# import os
# from dotenv import load_dotenv
# import asyncio
# from tqdm import tqdm

# from RagPipeline import RagPipeline
# from QAdataset import QuestionsDataset
# from eval import aeval, get_avg_result


# if __name__ == "__main__":
#     # os.chdir("Multimodal-RAG-opensource")
#     load_dotenv()
    
#     outpath = experiment_init()

#     with open(f"{outpath}/configs.yml", 'r') as f:
#         config = yaml.safe_load(f)    

#     conversational_chain = RagPipeline(config)

#     question = 'What happened to the pedestrians who were hit by a car in the Fairfax district?'

#     import pdb; pdb.set_trace()

#     res = conversational_chain.conversation_chain.invoke(question)

#     # res2 = conversational_chain.chain_with_guardrails.invoke(question)

from ragas import evaluate
import importlib
import os
# from ragas.metrics import (
#     answer_relevancy,
#     faithfulness,
#     context_recall,
#     context_precision,
# )
metrics = {}
module = importlib.import_module("ragas.metrics")
metric = getattr(module, "context_recall")
metrics["context_recall"] = metric

from datasets import load_dataset

# loading the V2 dataset
amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2")
import pdb; pdb.set_trace()
result = evaluate(amnesty_qa["eval"], metrics = [metric for metric in metrics.values()], raise_exceptions=False)

print(result)