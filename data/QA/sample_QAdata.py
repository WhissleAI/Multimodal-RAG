import json
import random

# 步骤2: 读取JSON文件
with open('data/QA/format_QAdata_small.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 步骤3: 计算5%的样本大小
sample_size = int(len(data) * 0.05)

# 步骤4: 从数据中随机采样5%的QA对
sampled_data = random.sample(data, sample_size)

# 步骤5: 将采样结果保存到一个新的JSON文件
with open('sampled_QAdata.json', 'w', encoding='utf-8') as file:
    json.dump(sampled_data, file, ensure_ascii=False, indent=4)