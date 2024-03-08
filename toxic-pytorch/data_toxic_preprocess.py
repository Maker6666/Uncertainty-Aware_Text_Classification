import re
import copy
import json
import pandas as pd
import torch
import numpy as np
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel


class InputExample(object):
    """ A single training/test example for simple sequence classification. """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
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


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, embeddings, label):
        self.embeddings = embeddings
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


class predict_label(object):
    def __init__(self, label_pred, label_true):
        self.label_pred = label_pred
        self.label_true = label_true


def toEmbeddings(
        tokenizer,
        bert_model,
        device,
        batch_text,
        max_seq_len,
        bert_layer
):

    sentences = [s.text_a for s in batch_text]
    batch = tokenizer.batch_encode_plus(
        sentences,
        add_special_tokens=True,
        max_length=max_seq_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        return_attention_mask=True)
    input_ids = batch['input_ids'].to(device)
    att_mask = batch['attention_mask'].to(device)
    with torch.no_grad():
        hidden_states = bert_model(input_ids, attention_mask=att_mask, output_hidden_states=True)[2]  # (B, L, H)
        features = hidden_states[bert_layer].cpu()
    return features


def convert_examples_to_features(
        args,
        examples,
        max_length,
        layer_idx
):

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    labels = [example.label.astype(int) for example in examples]
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name_or_path)
    bert_model = BertModel.from_pretrained(args.model_name_or_path).to(device)

    n_samples = len(examples)
    features = []
    for ii in range(0, n_samples, args.train_batch_size):
        batch = examples[ii:ii + args.train_batch_size]
        embeddings = toEmbeddings(tokenizer, bert_model, device, batch, max_length, layer_idx)
        batch_labels = torch.tensor(np.array(labels[ii:ii + args.train_batch_size]))
        feature = InputFeatures(embeddings, label=batch_labels)
        features.append(feature)
    return features


class GoToxicProcessor(object):
    def __init__(self):
        pass

    def data_split(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200, stratify=y[:, -1])
        return X_train, X_test, y_train, y_test

    def get_examples(self, X_train, X_test, y_train, y_test):
        train_examples, test_examples = [], []
        for i, (X, y) in enumerate(zip(X_train, y_train)):
            guid = "%s-%s" % ('train', i)
            text_a = X
            label = y
            train_examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        for i, (X, y) in enumerate(zip(X_test, y_test)):
            guid = "%s-%s" % ('test', i)
            text_a = X
            label = y
            test_examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

        return train_examples, test_examples

    def sample_dataset(self, train_df):
        df = train_df[train_df['clean'] == 0]
        df = pd.concat([df, train_df[train_df['clean'] == 1].sample(frac=1, random_state=200).iloc[:df.shape[0], :]])
        df = df.sample(frac=1, random_state=200)
        return df

    def read_data(self):

        targets, sentences = [], []
        data = pd.read_csv("dataset/toxic_comment.csv")
        # Adding new column of Clean Text
        arr = []
        for i in range(data.shape[0]):
            if (data.iloc[i, 2:] == 0).all():
                arr.append(1)
            else:
                arr.append(0)
        data['clean'] = pd.Series(np.asarray(arr))
        data = self.sample_dataset(data)
        for row in data.values:
            target = row[2:]
            sentence = str(row[1])
            sentence = self.cleanHtml(sentence)
            sentence = self.cleanPunc(sentence)
            sentence = self.keepAlpha(sentence)
            sentence = self.lemmatize(sentence)
            sentence = self.removeStopWords(sentence)
            if sentence:
                targets.append(target)
                sentences.append(sentence)
        return np.array(sentences, dtype=np.object), np.array(targets, dtype=np.int)

    def get_labels(self):
        return ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "clean"]

    def cleanHtml(self, sentence):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', str(sentence))
        return cleantext

    def cleanPunc(self, sentence):  # function to clean the word of any punctuation or special characters
        cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n", " ")
        return cleaned

    def keepAlpha(self, sentence):
        alpha_sent = ""
        for word in sentence.split():
            alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
            alpha_sent += alpha_word
            alpha_sent += " "
        alpha_sent = alpha_sent.strip()
        return alpha_sent

    def removeStopWords(self, sentence):
        stop_words = set(stopwords.words('english'))
        re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
        # global re_stop_words
        return re_stop_words.sub(" ", sentence)

    def lemmatize(self, sentence):
        lemmatizer = WordNetLemmatizer()
        lemSentence = ""
        for word in sentence.split():
            lem = lemmatizer.lemmatize(word)
            lemSentence += lem
            lemSentence += " "
        lemSentence = lemSentence.strip()
        return lemSentence
