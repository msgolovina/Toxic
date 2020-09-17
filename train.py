from dataset import ToxicDataset, collate_function
from model import BertClassifier

from functools import partial
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch.nn import BCELoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

CLASS_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
BATCH_SIZE=4
BERT_NAME='bert-base-cased'


def train(model, data_iterator, criterion, optimizer, scheduler):
    model.train()
    total_loss = 0
    for x, y in tqdm(data_iterator):
        optimizer.zero_grad()
        attention_mask = (x != 0).float()
        outputs = model(x, attention_mask=attention_mask)
        loss = criterion(outputs, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f'Train loss: {total_loss / len(data_iterator)}')

def evaluate(model, data_iterator, criterion):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        total_loss = 0
        for x, y in tqdm(data_iterator):
            attention_mask = (x != 0).float()
            outputs = model(x, attention_mask=attention_mask)
            loss = criterion(outputs, y)
            total_loss += loss
            y_true += y.cpu().numpy().tolist()
            y_pred += outputs.cpu().numpy().tolist()
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        for i, class_col in enumerate(CLASS_COLS):
            class_roc_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            print(f'{class_col} ROC AUC: {class_roc_auc}')
        print(f'Evaluation loss: {total_loss / len(data_iterator)}')


if __name__ == '__main__':

    if torch.cuda.is_available:
        device = torch.device('cuda:6')
    else:
        device = torch.device('cpu')

    model = BertClassifier(BertModel.from_pretrained(BERT_NAME), 6).to(device)

    train_data = pd.read_csv('data/train.csv')
    train_data, valid_data = train_test_split(train_data, test_size=0.05)

    tokenizer = BertTokenizer.from_pretrained(BERT_NAME)

    train_dataset = ToxicDataset(df=train_data, tokenizer=tokenizer)
    valid_dataset = ToxicDataset(df=valid_data, tokenizer=tokenizer)

    collate = partial(collate_function, device=device)

    train_sampler = RandomSampler(train_data)
    valid_sampler = RandomSampler(valid_data)

    train_data_iterator = DataLoader(train_data, batch_size=BATCH_SIZE,
                                     sampler=train_sampler, collate_fn=collate)
    valid_data_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE,
                                     sampler = valid_sampler, collate_fn=collate)

    criterion = BCELoss()

    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_params = [
        {
            'weight_decay': 0.0,
            'params': [
                param for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ]
        },
        {
            'weight_decay': 0.01,
            'params': [
                param for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ]
        }
    ]

    NUM_EPOCHS = 2
    NUM_WARMUP_STEPS = 10 ** 3
    total_steps = len(train_data_iterator) * NUM_EPOCHS - NUM_WARMUP_STEPS

    optimizer = AdamW(optimizer_params, lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, NUM_WARMUP_STEPS, total_steps)

    train_iterator = trange(
        0,
        NUM_EPOCHS,
        desc='Epoch',
    )

    for iteration in train_iterator:
        epoch_iterator = tqdm(
            train_data_iterator,
            desc='Iteration',
        )
        train(model, epoch_iterator, criterion, optimizer, scheduler)
        evaluate(model, valid_data_iterator, criterion)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained('trained_model')
    tokenizer.save_pretrained('trained_model')
