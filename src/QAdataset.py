from torch.utils.data import Dataset

class QuestionsDataset(Dataset):
    def __init__(self, json_data):
        self.questions = json_data['question']
        self.ground_truth = json_data['ground_truth']
        self.question_ground_truth = [pair for pair in zip(self.questions, self.ground_truth)]

    def __len__(self):
        assert len(self.ground_truth) == len(self.questions)
        return len(self.questions)

    def __getitem__(self, idx):
        return self.question_ground_truth[idx]  