from attrdict import AttrDict
import os
import copy
import json
import numpy as np
import torch
from torch import nn
from sklearn.metrics import hamming_loss, f1_score
from sklearn.metrics import accuracy_score, classification_report

from data_toxic_preprocess import GoToxicProcessor
import sys
import warnings
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def toTokens(tokenizer, batch_text, max_seq_len):
    sentences = [s.text_a for s in batch_text]
    batch = tokenizer.batch_encode_plus(
        sentences,
        add_special_tokens=True,
        max_length=max_seq_len,
        truncation=True,
        padding='max_length',

        return_tensors='pt',
        return_attention_mask=True)

    return batch


class InputsTokenizer(object):
    """A single set of inputs of data."""

    def __init__(self, inputs, label):
        self.inputs = inputs
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_inputs(
        args,
        examples,
        max_length,
):
    labels = [example.label.astype(int) for example in examples]
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name_or_path)

    n_samples = len(examples)
    inputs = []
    for ii in range(0, n_samples, args.train_batch_size):
        batch = examples[ii:ii + args.train_batch_size]
        tokens = toTokens(tokenizer, batch, max_length)
        batch_labels = torch.tensor(np.array(labels[ii:ii + args.train_batch_size]))
        feature = InputsTokenizer(tokens, label=batch_labels)
        inputs.append(feature)
    return inputs


class BertClassifier(nn.Module):

    def __init__(self, bert: BertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,

                labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            output_hidden_states=True,
                            head_mask=head_mask)
        cls_output = outputs[1]  # batch, hidden
        cls_output = self.classifier(cls_output)  # batch, 6
        cls_output = torch.sigmoid(cls_output)
        criterion = nn.BCELoss()
        loss = 0
        if labels is not None:
            loss += criterion(cls_output, labels)

        return loss, cls_output


def train(model, iterator, optimizer, scheduler, args):
    model.train()
    total_loss = 0
    for step, batch in enumerate(iterator):
        optimizer.zero_grad()
        # mask = (x != 0).float()
        inputs = batch.inputs.to(args.device)
        label_y = batch.label.to(torch.float).to(args.device)
        loss, outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=label_y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f"Train loss {total_loss / len(iterator)}")


def do_train(train_iterator, args):
    model = BertClassifier(BertModel.from_pretrained('bert-base-uncased'), 7).to(args.device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    EPOCH_NUM = 8
    warmup_steps = 10 ** 3
    total_steps = len(train_iterator) * EPOCH_NUM - warmup_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for i in range(EPOCH_NUM):
        print('=' * 50, f"EPOCH {i}", '=' * 50)
        train(model, train_iterator, optimizer, scheduler, args)
        # evaluate(model, dev_iterator)
    return model


def do_test(model, test_inputs, args):
    model.eval()
    y_pred = []
    y_test = []

    for step, batch in enumerate(test_inputs):
        inputs = batch.inputs.to(args.device)
        label_y = batch.label
        with torch.no_grad():
            _, outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        outputs = outputs.cpu().numpy()
        y_pred.append(outputs)
        y_test.append(label_y)

    y_pred = np.vstack(np.asarray(y_pred))
    y_pred = np.where(y_pred > 0.5, 1, 0)
    y_test = np.vstack(np.asarray(y_test))

    arr = {'model_name': 'BERT', 'micro_avg_f1_score': f1_score(y_test, y_pred, average="micro"),
           'macro_avg_f1_score': f1_score(y_test, y_pred, average="macro"),
           'hamming_loss': hamming_loss(y_test, y_pred), 'accuracy': accuracy_score(y_test, y_pred)}
    for eval_name in arr.keys():
        print(eval_name, ":", arr[eval_name])
    print(classification_report(y_pred, y_test, zero_division=0))


def main():
    config_filename = "{}.json".format('original')
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))
    args.device = "cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    processor = GoToxicProcessor()
    data_sentences, data_targets = processor.read_data()
    X_train, X_test, y_train, y_test = processor.data_split(X=data_sentences, y=data_targets)
    train_examples, test_examples = processor.get_examples(X_train, X_test, y_train, y_test)

    train_inputs = convert_examples_to_inputs(args, train_examples, args.max_seq_len)
    test_inputs = convert_examples_to_inputs(args, test_examples, args.max_seq_len)

    model = do_train(train_inputs, args)
    do_test(model, test_inputs, args)


if __name__ == '__main__':
    main()
