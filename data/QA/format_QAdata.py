import pandas
import json

QA_data = pandas.read_csv("data/QA/Q_and_A_pairs_data.csv") # question answer

json_data = {'question': [],
             'ground_truth': [],
             'answer': [],
             'contexts': []
             }


json_data['question'] = QA_data['question'].tolist()[30:80]
json_data['ground_truth'] = QA_data['answer'].tolist()[30:80]


with open('data/QA/format_QAdata_small.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=4, ensure_ascii=False)

