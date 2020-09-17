from config import COMMENT_COL, CLASS_COLS

from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class ToxicDataset(Dataset):

    def __init__(self, tokenizer, df):
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_token_id
        self.features = []
        self.targets = []
        for i, (row) in tqdm(df.iterrows()):
            feature, target = self.to_tensor(row, self.tokenizer)
            self.features.append(feature)
            self.targets.append(target)

    @staticmethod
    def to_tensor(row, tokenizer):
        tokens = tokenizer.encode(row[COMMENT_COL], add_special_tokens=True, max_length=512)
        if len(tokens) > 120:
            tokens = tokens[:119] + [tokens[-1]]
        feature = torch.LongTensor(tokens)
        target = torch.FloatTensor(row[CLASS_COLS])
        return feature, target

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


def collate_function(batch, device):
    feature, target = list(zip(*batch))
    feature = pad_sequence(feature, batch_first=True, padding_value=0)
    target = torch.stack(target)

    return feature.to(device), target.to(device)


