from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from data_toxic_preprocess import GoToxicProcessor
import torch
from torch import nn
from sklearn.metrics import hamming_loss, f1_score
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.metrics import accuracy_score, classification_report
from sklearn.multioutput import ClassifierChain
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
if not sys.warnoptions:
    warnings.simplefilter("ignore")

stop_words = set(stopwords.words('english'))
stop_words.update(
    ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'may', 'also', 'across',
     'among', 'beside', 'however', 'yet', 'within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

stemmer = SnowballStemmer("english")


# ML preprocessing
def baseline_ml_preprocessing(X_train, X_test):
    word_vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        analyzer='word',
        max_features=3000,
        token_pattern=r'\w{1,}',
        ngram_range=(1, 3),
        stop_words='english',
        sublinear_tf=True)

    word_vectorizer.fit(X_train)
    ml_X_train = word_vectorizer.transform(X_train)
    ml_X_test = word_vectorizer.transform(X_test)
    return ml_X_train, ml_X_test


def Sequential_Preprocessing(X_train, X_test, y_train, y_test, max_seq_len):
    tokenizer = Tokenizer(num_words=20000, oov_token='<UNK>')
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = torch.from_numpy(pad_sequences(X_train, maxlen=max_seq_len))
    X_test = torch.from_numpy(pad_sequences(X_test, maxlen=max_seq_len))

    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    return tokenizer, X_train, y_train, X_test, y_test


def get_bert_embedding_matrix(tokenizer):
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = len(tokenizer.index_word) + 1

    tokens_tensor = []
    for i in tqdm(range(1, vocab_size)):
        word = tokenizer.index_word[i]
        tokens = bert_tokenizer.encode(word, add_special_tokens=False)[0]
        tokens_tensor.append(tokens)
    batch_size = 512
    vocab_batches = [tokens_tensor[i:i + batch_size] for i in range(0, len(tokens_tensor), batch_size)]

    with torch.no_grad():
        bert_embeddings = []
        for i in tqdm(range(len(vocab_batches))):
            batch = vocab_batches[i]
            inputs = {
                'input_ids': torch.tensor([batch]),
                'attention_mask': torch.ones(len(batch)).unsqueeze(0)
            }
            outputs = bert_model(**inputs)
            batch_embeddings = outputs[0][0]
            bert_embeddings.append(batch_embeddings)

    bert_embedding_matrix = torch.cat(bert_embeddings, dim=0)
    return bert_embedding_matrix


# Build adjacency matrix based on Co-Occurencies label
def create_adjacency_matrix_cooccurance(data_label):
    cooccur_matrix = np.zeros((data_label.shape[1], data_label.shape[1]), dtype=float)
    for y in data_label:
        y = list(y)
        for i in range(len(y)):
            for j in range(len(y)):
                # data_label
                if y[i] == 1 and y[j] == 1:
                    cooccur_matrix[i, j] += 1
    row_sums = data_label.sum(axis=0)

    for i in range(cooccur_matrix.shape[0]):
        for j in range(cooccur_matrix.shape[0]):
            if row_sums[i] != 0:
                cooccur_matrix[i][j] = cooccur_matrix[i, j] / row_sums[i]
            else:
                cooccur_matrix[i][j] = cooccur_matrix[i, j]

    return cooccur_matrix


def check_accuracy(model, label_embedding, X, y):
    model.eval()
    with torch.no_grad():
        out = model(X, label_embedding)
        y_pred = torch.sigmoid(out.detach()).round().cpu()
        f1score = f1_score(y, y_pred, average='micro')
        hammingloss = hamming_loss(y, y_pred)
    return hammingloss, f1score


class dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def create_adjacency_matrix_xavier(data_label):
    adj_matrix = torch.empty((data_label.shape[1], data_label.shape[1]))
    adj_matrix = nn.init.xavier_uniform_(adj_matrix)
    return adj_matrix


processor = GoToxicProcessor()
data_sentences, data_targets = processor.read_data()
X_train, X_test, y_train, y_test = processor.data_split(X=data_sentences, y=data_targets)
labels = np.array(["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "clean"], dtype=np.object)
ml_X_train, ml_X_test = baseline_ml_preprocessing(X_train, X_test)
# print("X_train shape for baseline models:", ml_X_train.shape)
# print("X_test shape for baseline models:", ml_X_test.shape)


print("====================BR============================")
model = BinaryRelevance(LogisticRegression(solver='sag'))
model.fit(ml_X_train, y_train)
y_pred = model.predict(ml_X_test)
arr = {'model_name': 'BR', 'micro_avg_f1_score': f1_score(y_test, y_pred, average="micro"),
       'macro_avg_f1_score': f1_score(y_test, y_pred, average="macro"),
       'hamming_loss': hamming_loss(y_test, y_pred), 'accuracy': accuracy_score(y_test, y_pred)}
# scores_df = scores_df.append(arr, ignore_index=True)
for eval_name in arr.keys():
    print(eval_name, ":", arr[eval_name])
print(classification_report(y_pred, y_test))

print("====================Classifier Chain============================")

model = ClassifierChain(LogisticRegression(solver='sag'))
model.fit(ml_X_train, y_train)
y_pred = model.predict(ml_X_test)
arr = {'model_name': 'ClassifierChain', 'micro_avg_f1_score': f1_score(y_test, y_pred, average="micro"),
        'macro_avg_f1_score': f1_score(y_test, y_pred, average="macro"),
       'hamming_loss': hamming_loss(y_test, y_pred), 'accuracy': accuracy_score(y_test, y_pred)}
for eval_name in arr.keys():
    print(eval_name, ":", arr[eval_name])
print(classification_report(y_pred, y_test))

max_seq_len = 100
tokenizer, X_train, y_train, X_test, y_test = Sequential_Preprocessing(X_train, X_test, y_train, y_test, max_seq_len)

bert_embeddding_matrix = get_bert_embedding_matrix(tokenizer)
VOCAB_SIZE = len(tokenizer.index_word) + 1
bert_embd_mat = np.array(bert_embeddding_matrix.to('cpu').detach().numpy())
bert_embd_mat = np.vstack((np.zeros((1, bert_embd_mat.shape[1])), bert_embd_mat))

print("=====================Bert BiLSTM===========================")
inp = Input(shape=(max_seq_len,))
x = Embedding(VOCAB_SIZE, 768, weights=[bert_embd_mat])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(7, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(np.asarray(X_train), np.asarray(y_train), batch_size=256, epochs=3)
y_pred = model.predict(np.asarray(X_test), batch_size=256, verbose=1)
y_pred = np.where(y_pred > 0.5, 1, 0)

arr = {'model_name': 'Bert BiLSTM', 'micro_avg_f1_score': f1_score(y_test, y_pred, average="micro"),
       'macro_avg_f1_score': f1_score(y_test, y_pred, average="macro"),
       'hamming_loss': hamming_loss(y_test, y_pred), 'accuracy': accuracy_score(y_test, y_pred)}
for eval_name in arr.keys():
    print(eval_name, ":", arr[eval_name])
print(classification_report(y_pred, y_test))

print("=====================Bert CNN===========================")
inp = Input(shape=(max_seq_len,))

x = Embedding(VOCAB_SIZE, 768, weights=[bert_embd_mat])(inp)
x = layers.Conv1D(filters=128, kernel_size=5, activation='relu')(x)
x = GlobalMaxPool1D()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(32, activation='relu')(x)
x = Dense(7, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(np.asarray(X_train), np.asarray(y_train), batch_size=256, epochs=3)
y_pred = model.predict(np.asarray(X_test), batch_size=256, verbose=1)
y_pred = np.where(y_pred > 0.5, 1, 0)

arr = {'model_name': 'CNN_bert', 'micro_avg_f1_score': f1_score(y_test, y_pred, average="micro"),
        'macro_avg_f1_score': f1_score(y_test, y_pred, average="macro"),
       'hamming_loss': hamming_loss(y_test, y_pred), 'accuracy': accuracy_score(y_test, y_pred)}
for eval_name in arr.keys():
    print(eval_name, ":", arr[eval_name])
print(classification_report(y_pred, y_test))
