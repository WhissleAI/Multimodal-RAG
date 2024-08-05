import importlib
import aiofiles
from ragas import evaluate
from datasets import Dataset
import json
import numpy as np
import json
import os
import csv



async def aeval(eval_dataset, config, outpath):
    if not os.path.exists(f"{outpath}/eval.csv"):
        with open(f"{outpath}/eval.csv", 'w') as f:
            f.write(",".join(config['evaluation']['metrics']))        
    metric_names = config['evaluation']['metrics']
    
    metrics = []
    for metric_name in metric_names:
        if (metric_name=="context_recall" or metric_name=="context_precision") and config['use_rag']:
            continue
        module = importlib.import_module('ragas.metrics')
        metric = getattr(module, metric_name)
        metrics.append(metric)

    result = evaluate(
        eval_dataset,
        metrics=metrics,
        raise_exceptions=False,
    )

    async with aiofiles.open(f"{outpath}/eval.csv", 'a+') as f:
        await f.write(",".join([str(item) for key, item in result.items()]))
    print("eval_result: " + result)

def get_avg_result(filepath):
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    means = np.mean(data, axis=0)
    print("Column means:", means)
    with open(filepath) as f:
        writer = csv.writer(f)
        writer.writerow("mean: ")
        writer.writerow(means)