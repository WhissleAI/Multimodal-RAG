import pandas
import json

QA_data = pandas.read_csv("./ground_truth/Q_and_A_pairs_data.csv") # question answer

json_data = {'question': [],
             'ground_truth': [],
             'answer': [],
             'contexts': []
             }


json_data['question'] = QA_data['question'].tolist()
json_data['ground_truth'] = QA_data['answer'].tolist()


with open('dataset/format_QAdata.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=4, ensure_ascii=False)

