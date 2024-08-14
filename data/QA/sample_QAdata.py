import pandas as pd
import random

# 步骤2: 读取CSV文件
df = pd.read_csv('data/QA/Q_and_A_pairs_data.csv')

# 步骤3: 计算5%的样本大小
sample_size = int(len(df) * 0.1)
print(f"Sample size: {sample_size}")

# 步骤4: 从数据中随机采样5%的QA对
sampled_df = df.sample(n=sample_size, random_state=1)

# 步骤5: 将采样结果保存到一个新的CSV文件
sampled_df.to_csv('data/QA/sampled_Q_and_A_pairs_data.csv', index=False)