import pandas as pd
import random


df = pd.read_csv('data/QA/Q_and_A_pairs_data.csv')


sample_size = int(len(df) * 0.1)
print(f"Sample size: {sample_size}")


sampled_df = df.sample(n=sample_size, random_state=1)


sampled_df.to_csv('data/QA/sampled_Q_and_A_pairs_data.csv', index=False)