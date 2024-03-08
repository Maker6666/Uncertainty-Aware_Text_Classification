import os
import copy
import logging
import json
from tqdm import trange
import numpy as np
from attrdict import AttrDict
import torch.utils.data.distributed
from transformers import AdamW, get_linear_schedule_with_warmup
from model import TextCnn, MultiLabelClassification
from data_toxic_preprocess import convert_examples_to_features, GoToxicProcessor
from sklearn.metrics import hamming_loss, f1_score, classification_report, accuracy_score
import warnings
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def create_data(args):
    processor = GoToxicProcessor()
    data_sentences, data_targets = processor.read_data()
    X_train, X_test, y_train, y_test = processor.data_split(X=data_sentences, y=data_targets)
    train_examples, test_examples = processor.get_examples(X_train, X_test, y_train, y_test)

    for layer_idx in range(1, args.bert_layers + 1):
        print("bert_layer {} generate embedding...".format(layer_idx))

        train_features = convert_examples_to_features(args, train_examples, args.max_seq_len, layer_idx)
        cached_features_file = os.path.join(
            args.data_dir,
            "{}_train_layer{}_feature_bert".format(
                args.data_name,
                layer_idx
            )
        )
        torch.save(train_features, cached_features_file)
        test_features = convert_examples_to_features(args, test_examples, args.max_seq_len, layer_idx)
        cached_features_file = os.path.join(
            args.data_dir,
            "{}_test_layer{}_feature_bert".format(
                args.data_name,
                layer_idx
            )
        )
        torch.save(np.array(test_features), cached_features_file)
        print("bert_layer {} generate done".format(layer_idx))


def run_train(args):

    for label_idx in range(0, 1):
        global best_model
        best_layer_idx = 1
        min_layer_eval_loss = 2

        for layer_idx in range(1, args.bert_layers + 1):
            cached_train_features_file = os.path.join(
                args.data_dir,
                "{}_train_layer{}_feature_bert".format(
                    args.data_name,
                    layer_idx
                )
            )
            train_dataset = torch.load(cached_train_features_file)

            model, layer_eval_loss = layer_var_train(train_dataset, label_idx, layer_idx, args)
            # print("layer {} eval_loss {}".format(layer_idx, layer_eval_loss))

            if layer_eval_loss < min_layer_eval_loss:
                min_layer_eval_loss = layer_eval_loss
                best_layer_idx = layer_idx
                best_model = copy.deepcopy(model)

        cached_model_save_file = os.path.join(
            args.data_dir,
            "{}_label_{}_best_model_bert".format(
                args.data_name,
                label_idx
            )
        )
        torch.save({'best_layer': best_layer_idx,
                    'model_state_dict': best_model.state_dict()}, cached_model_save_file)
        print("Label {} Best layer {} min layer eval loss {}".format(label_idx, best_layer_idx, min_layer_eval_loss))


def layer_var_train(train_dataset, label_idx, layer_idx, args):
    warmup_steps = 10 ** 3
    total_steps = len(train_dataset) * args.num_train_epochs - warmup_steps
    model = MultiLabelClassification(TextCnn(args.embedding_dim, 768, [3, 4, 5]), args.embedding_dim + label_idx, 1).to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    layer_eval_loss = 0
    x, los, acc_list = [], [], []
    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_loss = 0
        x.append(epoch + 1)

        for step, batch in enumerate(train_dataset):
            inputs = batch.embeddings.to(args.device)
            label_y = batch.label.to(args.device)

            loss, outputs = model(inputs, labels=label_y, label_idx=label_idx, istrain=True)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        print("Label {} Layer {} Epoch {} Train loss {}".format(label_idx, layer_idx, epoch, epoch_loss / len(train_dataset)))
        layer_eval_loss += epoch_loss / len(train_dataset)
        los.append(epoch_loss / len(train_dataset))

    return model, layer_eval_loss / args.num_train_epochs


def run_test(args):

    y_pred, y_true = [], []
    for label_idx in trange(0, args.label_num):
        cached_model_save_file = os.path.join(
            args.data_dir,
            "{}_label_{}_best_model_bert".format(
                args.data_name,
                label_idx
            )
        )
        label_model = MultiLabelClassification(TextCnn(args.embedding_dim, 768, [3, 4, 5]), args.embedding_dim + label_idx, 1)
        checkpoint = torch.load(cached_model_save_file)
        best_layer = checkpoint['best_layer']
        label_model.load_state_dict(checkpoint['model_state_dict'])
        label_model.to(args.device)
        label_model.eval()

        cached_test_features_file = os.path.join(
            args.data_dir,
            "{}_test_layer{}_feature_bert".format(
                args.data_name,
                best_layer
            )
        )
        test_dataset = torch.load(cached_test_features_file)

        with torch.no_grad():
            label_y_true = []
            label_y_pred = []

            for idx, batch in enumerate(test_dataset):
                inputs = batch.embeddings.to(args.device)
                label_y = batch.label.to(args.device)
                _, outputs = label_model(inputs, labels=label_y, label_idx=label_idx)
                outputs = outputs.cpu().numpy()
                predicted = np.where(outputs > 0.5, 1, 0).squeeze()
                label_y_pred.extend(predicted)
                label_y = label_y.cpu().numpy()
                label_y = label_y[:, label_idx]
                label_y_true.extend(label_y)

        y_pred.append(label_y_pred)
        y_true.append(label_y_true)
    pred = [[sublist[i] for sublist in y_pred] for i in range(len(y_pred[0]))]
    true = [[sublist[i] for sublist in y_true] for i in range(len(y_true[0]))]
    arr = {'model_name': 'bert_uncertainty', 'micro_avg_f1_score': f1_score(true, pred, average="micro"),
           'macro_avg_f1_score': f1_score(true, pred, average="macro"),
           'hamming_loss': hamming_loss(true, pred), 'accuracy': accuracy_score(true, pred)}
    for eval_name in arr.keys():
        print(eval_name, ":", arr[eval_name])
    print(classification_report(pred, true))


def main():
    # Read from config file and make args
    config_filename = "{}.json".format('original')
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    # GPU or CPU
    args.device = "cuda:1" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    # create and save text embeddings
    if args.do_data:
        create_data(args)
    # train
    if args.do_train:
        run_train(args)
    # test
    if args.do_test:
        run_test(args)


if __name__ == '__main__':
    main()
