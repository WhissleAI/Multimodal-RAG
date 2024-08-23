import pandas as pd
import json
import re
for i in range(10, 13):
    csvname = f"2016_{i}_english.csv"
    csvpath = f"/home/featurize/work/Multimodal-RAG/data/context_metadata/{csvname}"
    df = pd.read_csv(csvpath)

    metadata = ['lan', 'src', 'dur', 'col', 'path']

    column_content = df['metadata']

    for key in metadata:
        values = [json.loads(content.replace("\'", "\""))[key] for content in column_content]
        df[key] = values

    df.drop(columns=['metadata'], inplace=True)

    df.to_csv(f"/home/featurize/work/Multimodal-RAG/data/context_metadata/processed_{csvname}", index=False)