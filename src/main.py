import os
import json
from utils import experiment_init
import torch
import yaml
from torch.utils.data import DataLoader
from datasets import Dataset
import huggingface_hub
import os
from dotenv import load_dotenv
import asyncio

from RagPipeline import RagPipeline
from QAdataset import QuestionsDataset
from eval import aeval, get_avg_result


def rag_and_eval():
    collected_data = {
        'question': [],
        'ground_truth': [],
        'answer': [],
        'contexts': []
    }
    for i, batch in enumerate(data_loader):
        try:
            batch = zip(batch[0], batch[1])
            for question, ground_truth in batch:
                result = conversational_chain.conversation_chain.invoke(question)
                if config['use_rag']:
                    context = [doc.page_content for doc in result['context']]        
                else:
                    context = [""]
                collected_data['contexts'].append(context)
                response = result['answer']
                collected_data['answer'].append(response)
                collected_data['question'].append(question)
                collected_data['ground_truth'].append(ground_truth)

                data = {
                'question': [question],
                'ground_truth': [ground_truth],
                'answer': [response],
                'contexts': context        
                }

                # print(f"    question: {question}\n\n    ground_truth: {ground_truth}\n\n    answer: {response}\n\n    context: {context}\n\n ---end--- \n\n")
                print(data)

                eval_dataset = Dataset.from_dict(data)
                asyncio.run(aeval(eval_dataset, config, outpath))
                
            output_filename = f"{outpath}/result.json"
            with open(output_filename, 'w') as f:
                json.dump(collected_data, f)
            print(f"batch {i} finished.")
        except huggingface_hub.utils._errors.HfHubHTTPError:
            print(f"batch {i} failed.")

if __name__ == "__main__":
    # os.chdir("Multimodal-RAG-opensource")
    load_dotenv()

    torch.cuda.empty_cache()
    
    outpath = experiment_init()

    with open(f"{outpath}/configs.yml", 'r') as f:
        config = yaml.safe_load(f)    

    conversational_chain = RagPipeline(config)

    with open(config['dataset']['file']) as f:
        json_data = json.load(f)

    dataset = QuestionsDataset(json_data)
    data_loader = DataLoader(dataset, batch_size=config['dataloader']['batch_size'], shuffle=config['dataloader']['shuffle'])


    rag_and_eval()

    get_avg_result(f"{outpath}/eval.csv")

