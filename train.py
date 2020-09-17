import config
from dataset import ToxicDataset, collate_function
from model import BertClassifier

from functools import partial
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import torch
from torch.nn import BCELoss
from torch.utils.data import Dataset, DataLoader, RandomSampler
from tqdm import tqdm, trange
from transformers import BertConfig, BertTokenizer, AdamW, \
    get_linear_schedule_with_warmup


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
        for i, class_col in enumerate(config.CLASS_COLS):
            class_roc_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            print(f'{class_col} ROC AUC: {class_roc_auc}')
        print(f'Evaluation loss: {total_loss / len(data_iterator)}')


if __name__ == '__main__':

    if torch.cuda.is_available:
        device = torch.device('cuda:6')
    else:
        device = torch.device('cpu')

    bert_config = BertConfig.from_pretrained(config.BERT_NAME)
    bert_config.num_labels = config.NUM_CLASSES
    model = BertClassifier.from_pretrained(config.BERT_NAME, bert_config)
    model.to(device)

    train_data = pd.read_csv('data/train.csv')
    train_data, valid_data = train_test_split(train_data, test_size=0.05)
    tokenizer = BertTokenizer.from_pretrained(config.BERT_NAME)
    train_dataset = ToxicDataset(df=train_data, tokenizer=tokenizer)
    valid_dataset = ToxicDataset(df=valid_data, tokenizer=tokenizer)
    collate = partial(collate_function, device=device)
    train_sampler = RandomSampler(train_dataset)
    valid_sampler = RandomSampler(valid_dataset)
    train_data_iterator = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate
    )
    valid_data_iterator = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=valid_sampler,
        collate_fn=collate,
    )

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

    total_steps = len(train_data_iterator) * config.NUM_EPOCHS - \
                  config.NUM_WARMUP_STEPS

    optimizer = AdamW(optimizer_params, lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, config.NUM_WARMUP_STEPS, total_steps
    )

    # training

    for i in range(config.NUM_EPOCHS):
        print('='*50, f'EPOCH {i}', '='*50)
        train(model, train_data_iterator, criterion, optimizer, scheduler)
        evaluate(model, valid_data_iterator, criterion)

    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained('trained_model')
    tokenizer.save_pretrained('trained_model')
