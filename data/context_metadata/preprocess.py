import pandas as pd
import json
import re
csvname = "2016_01_english_with_metadata.csv"
csvpath = f"./data/context_metadata/{csvname}"
df = pd.read_csv(csvpath)

metadata = ['lan', 'src', 'dur', 'col', 'path']

column_content = df['metadata']

for key in metadata:
    values = [json.loads(content.replace("\'", "\""))[key] for content in column_content]
    df[key] = values

df.drop(columns=['metadata'], inplace=True)

df.to_csv(f"./data/context_metadata/processed_{csvname}", index=False)