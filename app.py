import config
from model import BertClassifier
from transformers import BertTokenizer, BertConfig
from torch.nn.utils.rnn import pad_sequence

import flask
from flask import Flask
from flask import request
import time
import torch


app = Flask(__name__)


def to_tensor(comment, tokenizer):
    tokens = tokenizer.encode(
        comment,
        add_special_tokens=True,
        max_length=config.MAX_LENGTH,
        truncation=True
    )
    if len(tokens) > 120:
        tokens = tokens[:119] + [tokens[-1]]
    x = torch.LongTensor(tokens)
    return x

def comment_prediction(comment):
    comment = str(comment)
    tokenizer = BertTokenizer.from_pretrained(config.BERT_NAME)
    x = to_tensor(comment, tokenizer).view(1, -1)
    x = pad_sequence(x, batch_first=True, padding_value=0)
    x = x.to(config.DEVICE).long()
    attention_mask = (x != 0).float().to(config.DEVICE).long()
    outputs = MODEL(x, attention_mask=attention_mask)
    return outputs.cpu().detach().numpy()


@app.route('/predict')
def predict():
    comment = request.args.get('comment')
    start_time = time.time()
    prediction = comment_prediction(comment)
    response = {'response': {
        label: str(prob) for label, prob in
        zip(config.CLASS_COLS, prediction[0])
    }}
    response['response']['comment'] = comment
    response['response']['time_taken'] = str(time.time() - start_time)

    return flask.jsonify(response)


if __name__ == '__main__':
    bert_config = BertConfig.from_pretrained(config.BERT_NAME)
    bert_config.num_labels = config.NUM_CLASSES
    MODEL = BertClassifier(bert_config)
    MODEL.load_state_dict(torch.load(config.TRAINED_MODEL_PATH))
    MODEL.to(config.DEVICE)
    MODEL.eval()
    app.run(host='0.0.0.0', port='6006')
