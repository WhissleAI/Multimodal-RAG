

with open("./out/result_Phi-3-mini-4k-instruct-chunksize756-overlap150-2024-07-09-10-38-24.json", 'r') as f:
    data = f.read()


lines = data.strip().split('\n')
entries = [json.loads(line) for line in lines]

context_precision_sum = 0
faithfulness_sum = 0
answer_relevancy_sum = 0
context_recall_sum = 0

for i, entry in enumerate(entries):
    if np.isnan(entry['faithfulness']):
        print(i)
        continue

    # if i==0:
    #     context_precision_sum += 10*entry['context_precision']
    #     faithfulness_sum += 10*entry['faithfulness']
    #     answer_relevancy_sum += 10*entry['answer_relevancy']
    #     context_recall_sum += 10*entry['context_recall']    
    # elif i==1 or i==2 :
    #     context_precision_sum += 6*entry['context_precision']
    #     faithfulness_sum += 6*entry['faithfulness']
    #     answer_relevancy_sum += 6*entry['answer_relevancy']
    #     context_recall_sum += 6*entry['context_recall']
    if i<5:
        continue
    else:   
        context_precision_sum += entry['context_precision']
        faithfulness_sum += entry['faithfulness']
        answer_relevancy_sum += entry['answer_relevancy']
        context_recall_sum += entry['context_recall']


num_entries = len(entries)-4-5
context_precision_avg = context_precision_sum / num_entries
faithfulness_avg = faithfulness_sum / num_entries
answer_relevancy_avg = answer_relevancy_sum / num_entries
context_recall_avg = context_recall_sum / num_entries


print(f"Context Precision Average: {context_precision_avg}")
print(f"Faithfulness Average: {faithfulness_avg}")
print(f"Answer Relevancy Average: {answer_relevancy_avg}")
print(f"Context Recall Average: {context_recall_avg}")