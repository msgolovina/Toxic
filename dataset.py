import config

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class ToxicDataset(Dataset):

    def __init__(self, tokenizer, df):
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.pad_token_id
        self.X = []
        self.Y = []
        for i, (row) in tqdm(df.iterrows()):
            x, y = self.to_tensor(row, self.tokenizer)
            self.X.append(x)
            self.Y.append(y)

    @staticmethod
    def to_tensor(row, tokenizer):
        tokens = tokenizer.encode(
            row[config.COMMENT_COL],
            add_special_tokens=True,
            max_length=config.MAX_LENGTH,
            truncation=True
        )
        if len(tokens) > 120:
            tokens = tokens[:119] + [tokens[-1]]
        x = torch.LongTensor(tokens)
        y = torch.FloatTensor(row[config.CLASS_COLS])
        return x, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def collate_function(batch, device):
    x, y = list(zip(*batch))
    x = pad_sequence(x, batch_first=True, padding_value=0)
    y = torch.stack(y)

    return x.to(device), y.to(device)
