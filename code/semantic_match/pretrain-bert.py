
import os
import math
import numpy as np
import pandas as pd
import random
from tqdm.autonotebook import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score

import torch
from torch.utils.data import Dataset,DataLoader,random_split
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn import Module, Linear, LayerNorm

from transformers import BertModel

# class PretrainBertDS(Dataset):
#     def __init__(self, train_path, test_path, max_seq_len=64):
#         super(PretrainBertDS, self).__init__()
#         self.train_path = train_path
#         self.test_path = test_path
#         self.max_seq_len = max_seq_len
#         # self.seq_1, self.seq_2 = self.load_data()
#         self.seq = self.load_data()
#         self.vocab = self.get_vocab()
#         return

#     def get_vocab(self):
#         min_count = 5
#         test_data = pd.read_csv(self.train_path, sep="\t", names=["seq1", "seq2"])
#         train_data = pd.read_csv(self.test_path, sep="\t", names=["seq1", "seq2", "label"])

#         data = pd.concat([train_data[["seq1", "seq2"]], test_data[["seq1", "seq2"]]])
#         data = data['seq1'].append(data['seq2']).to_numpy()
#         special_tokens = ["[PAD]",
#                           "[UNK]",
#                           "[CLS]",
#                           "[SEP]",
#                           "[MASK]"]

#         vocab = defaultdict(int)
#         for seq in data:
#             if isinstance(seq, int):
#                 if str(seq) in vocab.keys():
#                     vocab[str(seq)] += 1
#                 else:
#                     vocab[str(seq)] = 0
#             else:
#                 for w in seq.split(' '):
#                     if w in vocab.keys():
#                         vocab[w] += 1
#                     else:
#                         vocab[w] = 0

#         vocab = {token: count for token, count in vocab.items() if count > min_count}
#         vocab = dict(sorted(vocab.items(), key=lambda item: -item[1]))
#         vocab = special_tokens + list(vocab.keys())
#         vocab = dict(zip(vocab, range(len(vocab))))
#         return vocab

#     def load_data(self):
#         seq = []
#         with open(self.train_path, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 s1, s2 = line.strip().split('\t')[:2]
#                 seq.append(s1 + '\t' + s2)
#                 seq.append(s1 + '\t' + "#")
#                 seq.append(s2 + '\t' + "#")

#         with open(self.test_path, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 s1, s2 = line.strip().split('\t')[:2]
#                 seq.append(s1 + '\t' + s2)
#                 seq.append(s1 + '\t' + "#")
#                 seq.append(s2 + '\t' + "#")
#         return seq

#     def random_mask_token(self, seq):
#         tokens = seq.split(' ')
#         tokens_label = []

#         for i in range(len(tokens)):
#             mask_prob = random.random()
#             if mask_prob < 0.15:
#                 tokens_label.append(self.vocab.get(tokens[i], self.vocab['[UNK]']))
#                 mask_prob /= 0.15
#                 if mask_prob < 0.8:
#                     tokens[i] = '[MASK]'
#                 elif mask_prob < 0.9:
#                     tokens[i] = random.choice(list(self.vocab.keys()))
#                 else:
#                     tokens[i] = tokens[i]
#             else:
#                 tokens_label.append(-1)
#         return tokens, tokens_label


#     def __len__(self):
#         return len(self.seq)

#     def __getitem__(self, index):
#         seq = self.seq[index]
#         s1, s2 = seq.strip().split('\t')

#         if random.random() > 0.5:
#             s1, s2 = s2, s1

#         if s1 != "#" and s2 != "#":
#             s1, s1_label = self.random_mask_token(s1)
#             s2, s2_label = self.random_mask_token(s2)
#             seq = ['[CLS]'] + s1 + ['[SEP]'] + s2 + ['[SEP]']
#             seq = seq[:self.max_seq_len]

#             token_label = [self.vocab['[CLS]']] + s1_label + [self.vocab['[SEP]']] + s2_label + [self.vocab['[SEP]']]
#             token_label = token_label[:self.max_seq_len]

#             segment_label = [0] * (len(s1) + 2) + [1] * (len(s2) + 1)
#             segment_label = segment_label[:self.max_seq_len]

#             padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(seq))

#             input_ids = []
#             for i in range(len(seq)):
#                 input_ids.append(self.vocab.get(seq[i], self.vocab['[UNK]']))

#             input_ids.extend(padding)
#             segment_label.extend(padding)
#             token_label.extend(padding)
#             attention_mask = [1] * (len(input_ids) - len(padding)) + [0] * len(padding)

#             return {
#                 "input_ids": np.array(input_ids),
#                 "token_label": np.array(token_label),
#                 "token_type_ids": np.array(segment_label),
#                 "attention_mask": np.array(attention_mask)
#             }
#         else:
#             if s1 == "#":
#                 s1 = s2

#             s1, s1_label = self.random_mask_token(s1)
#             seq = ['[CLS]'] + s1 + ['[SEP]']
#             seq = seq[:self.max_seq_len]

#             token_label = [self.vocab['[CLS]']] + s1_label + [self.vocab['[SEP]']]
#             token_label = token_label[:self.max_seq_len]

#             segment_label = [0] * (len(s1) + 2)
#             segment_label = segment_label[:self.max_seq_len]

#             padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(seq))

#             input_ids = []
#             for i in range(len(seq)):
#                 input_ids.append(self.vocab.get(seq[i], self.vocab['[UNK]']))

#             input_ids.extend(padding)
#             segment_label.extend(padding)
#             token_label.extend(padding)
#             attention_mask = [1] * (len(input_ids) - len(padding)) + [0] * len(padding)

#             return {
#                 "input_ids": np.array(input_ids),
#                 "token_label": np.array(token_label),
#                 "token_type_ids": np.array(segment_label),
#                 "attention_mask": np.array(attention_mask)
#             }



# class PretrainedBERT(Module):
#     def __init__(self, embedding_size, embedding_dim, max_len, keep_index=None):
#         super(PretrainedBERT, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-chinese')
#         self.lr1 = Linear(in_features=embedding_dim, out_features=embedding_dim)
#         self.ln1 = LayerNorm([max_len, embedding_dim])
#         self.lr2 = Linear(in_features=embedding_dim, out_features=embedding_size)
#         return

#     def forward(self, x):
#         x = self.bert(input_ids=x['input_ids'],
#                       attention_mask=x['attention_mask'],
#                       token_type_ids=x['token_type_ids'])
#         x = self.lr1(x['last_hidden_state'])
#         x = self.ln1(x)
#         x = self.lr2(x)
#         return x

# def pretrain(train_path,test_path):

#     num_epochs = 10
#     batch_size = 64
#     learning_rate = 2e-5
#     save_model_dir = "/content/gdrive/MyDrive/model/"
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

#     max_seq_len = 64
#     dataset = PretrainBertDS(train_path, test_path, max_seq_len)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     embedding_dim = 768
#     vocab_size = len(dataset.vocab)
#     model = PretrainedBERT(embedding_size=vocab_size,
#                            embedding_dim=embedding_dim,
#                            max_len=max_seq_len)

#     criterion = CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=learning_rate)
#     model = model.to(device)

#     for i, epoch in enumerate(range(num_epochs)):
#         train_losses = []
#         show_bar = tqdm(data_loader)
#         for input_data in show_bar:
#             input_data['input_ids'] = input_data['input_ids'].to(device)
#             input_data['token_label'] = input_data['token_label'].to(device)
#             input_data['token_type_ids'] = input_data['token_type_ids'].to(device)
#             input_data['attention_mask'] = input_data['attention_mask'].to(device)
#             optimizer.zero_grad()
#             output = model(input_data)
#             token_mask = (input_data['token_label'] != -1)
#             train_loss = criterion(output[token_mask].view(-1, vocab_size),
#                                     input_data['token_label'][token_mask].view(-1))
#             train_loss.backward()
#             optimizer.step()
#             train_losses.append(train_loss.cpu().detach().numpy())
#             show_bar.set_description("Epoch {0}, Loss {1:0.2f}"
#                                     .format(epoch, np.mean(train_losses)))
#         model_path = save_model_dir + "pretrain_{0}_{1:0.2f}_bert.pth".format(epoch,np.mean(train_losses))
#         torch.save({"model": model.state_dict(),}, model_path)
#     return



class PretrainBertDSV1(Dataset):
    def __init__(self, train_path, test_path, max_seq_len=64):
        super(PretrainBertDSV1, self).__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.max_seq_len = max_seq_len
        # self.seq_1, self.seq_2 = self.load_data()
        self.seq = self.load_data()
        self.vocab = self.get_vocab()

        self.span_min_length = 1
        self.span_max_length = 10
        self.span_mask_ratio = 0.15
        self.span_lengths = list(range(self.span_min_length, self.span_max_length + 1))

        self.geometric_prop = 0.3
        self.prop_distribute = [self.geometric_prop * (1 - self.geometric_prop) ** (length - 1)
                                for length in self.span_lengths]
        self.prop_distribute = [x / sum(self.prop_distribute) for x in self.prop_distribute]

        return

    def get_span_mask_index(self, seq):
        seq_length = len(seq)
        mask_mum = math.ceil(seq_length * self.span_mask_ratio)

        mask_index = set()
        while len(mask_index) < mask_mum:
            mask_length = np.random.choice(self.span_lengths, p=self.prop_distribute)

            start = np.random.choice(seq_length)
            if start in mask_index:
                continue
            end = start + mask_length

            for index in range(start, end):
                if len(mask_index) >= mask_mum:
                    break
                if index > seq_length - 1:
                    break
                mask_index.add(index)

        return mask_index

    def get_vocab(self):
        min_count = 5
        test_data = pd.read_csv(self.train_path, sep="\t", names=["seq1", "seq2"])
        train_data = pd.read_csv(self.test_path, sep="\t", names=["seq1", "seq2", "label"])

        data = pd.concat([train_data[["seq1", "seq2"]], test_data[["seq1", "seq2"]]])
        data = data['seq1'].append(data['seq2']).to_numpy()
        special_tokens = ["[PAD]",
                          "[UNK]",
                          "[CLS]",
                          "[SEP]",
                          "[MASK]"]

        vocab = defaultdict(int)
        for seq in data:
            if isinstance(seq, int):
                if str(seq) in vocab.keys():
                    vocab[str(seq)] += 1
                else:
                    vocab[str(seq)] = 0
            else:
                for w in seq.split(' '):
                    if w in vocab.keys():
                        vocab[w] += 1
                    else:
                        vocab[w] = 0

        vocab = {token: count for token, count in vocab.items() if count > min_count}
        vocab = dict(sorted(vocab.items(), key=lambda item: -item[1]))
        vocab = special_tokens + list(vocab.keys())
        vocab = dict(zip(vocab, range(len(vocab))))
        return vocab

    def load_data(self):
        seq = []
        with open(self.train_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                s1, s2 = line.strip().split('\t')[:2]
                seq.append(s1 + "\t" + "#")
                seq.append(s2 + "\t" + "#")
                seq.append(s1 + "\t" + s2)

        with open(self.test_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                s1, s2 = line.strip().split('\t')[:2]
                seq.append(s1 + "\t" + "#")
                seq.append(s2 + "\t" + "#")
                seq.append(s1 + "\t" + s2)
        return seq

    def random_mask_token(self, seq):
        tokens = seq.split(' ')
        tokens_label = []

        mask_index = self.get_span_mask_index(tokens)

        for index in range(len(tokens)):
            if index not in mask_index:
                tokens_label.append(-1)
            else:
                tokens_label.append(self.vocab.get(tokens[index], self.vocab['[UNK]']))
                mask_prob = random.random()
                if mask_prob < 0.8:
                    tokens[index] = '[MASK]'
                elif mask_prob < 0.9:
                    tokens[index] = random.choice(list(self.vocab.keys()))
                else:
                    tokens[index] = tokens[index]
        return tokens, tokens_label

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        seq = self.seq[index]
        s1, s2 = seq.strip().split('\t')

        if random.random() > 0.5:
            s1, s2 = s2, s1

        if s1 != "#" and s2 != "#":
            s1, s1_label = self.random_mask_token(s1)
            s2, s2_label = self.random_mask_token(s2)
            seq = ['[CLS]'] + s1 + ['[SEP]'] + s2 + ['[SEP]']
            seq = seq[:self.max_seq_len]

            token_label = [self.vocab['[CLS]']] + s1_label + [self.vocab['[SEP]']] + s2_label + [self.vocab['[SEP]']]
            token_label = token_label[:self.max_seq_len]

            segment_label = [0] * (len(s1) + 2) + [1] * (len(s2) + 1)
            segment_label = segment_label[:self.max_seq_len]

            padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(seq))

            input_ids = []
            for i in range(len(seq)):
                input_ids.append(self.vocab.get(seq[i], self.vocab['[UNK]']))

            input_ids.extend(padding)
            segment_label.extend(padding)
            token_label.extend(padding)
            attention_mask = [1] * (len(input_ids) - len(padding)) + [0] * len(padding)

            return {
                "input_ids": np.array(input_ids),
                "token_label": np.array(token_label),
                "token_type_ids": np.array(segment_label),
                "attention_mask": np.array(attention_mask)
            }
        else:
            if len(s1) == 0:
                s1 = s2

            s1, s1_label = self.random_mask_token(s1)
            seq = ['[CLS]'] + s1 + ['[SEP]']
            seq = seq[:self.max_seq_len]

            token_label = [self.vocab['[CLS]']] + s1_label + [self.vocab['[SEP]']]
            token_label = token_label[:self.max_seq_len]

            segment_label = [0] * (len(s1) + 2)
            segment_label = segment_label[:self.max_seq_len]

            padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(seq))

            input_ids = []
            for i in range(len(seq)):
                input_ids.append(self.vocab.get(seq[i], self.vocab['[UNK]']))

            input_ids.extend(padding)
            segment_label.extend(padding)
            token_label.extend(padding)
            attention_mask = [1] * (len(input_ids) - len(padding)) + [0] * len(padding)

            return {
                "input_ids": np.array(input_ids),
                "token_label": np.array(token_label),
                "token_type_ids": np.array(segment_label),
                "attention_mask": np.array(attention_mask)
            }


class PretrainedBERTV1(Module):
    def __init__(self, embedding_size, embedding_dim, max_len, keep_index=None):
        super(PretrainedBERTV1, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.lr1 = Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.ln1 = LayerNorm([max_len, embedding_dim])
        self.lr2 = Linear(in_features=embedding_dim, out_features=embedding_size)
        return

    def forward(self, x):
        x = self.bert(input_ids=x['input_ids'],
                      attention_mask=x['attention_mask'],
                      token_type_ids=x['token_type_ids'])
        x = self.lr1(x['last_hidden_state'])
        x = self.ln1(x)
        x = self.lr2(x)
        return x


def pretrainV1(train_path, test_path):
    num_epochs = 10
    batch_size = 64
    learning_rate = 2e-5
    save_model_dir = "/content/gdrive/MyDrive/model/"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

    max_seq_len = 64
    dataset = PretrainBertDSV1(train_path, test_path, max_seq_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    embedding_dim = 768
    vocab_size = len(dataset.vocab)
    model = PretrainedBERTV1(embedding_size=vocab_size,
                             embedding_dim=embedding_dim,
                             max_len=max_seq_len)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)

    for i, epoch in enumerate(range(num_epochs)):
        train_losses = []
        show_bar = tqdm(data_loader)
        for input_data in show_bar:
            input_data['input_ids'] = input_data['input_ids'].to(device)
            input_data['token_label'] = input_data['token_label'].to(device)
            input_data['token_type_ids'] = input_data['token_type_ids'].to(device)
            input_data['attention_mask'] = input_data['attention_mask'].to(device)
            optimizer.zero_grad()
            output = model(input_data)
            token_mask = (input_data['token_label'] != -1)
            train_loss = criterion(output[token_mask].view(-1, vocab_size),
                                   input_data['token_label'][token_mask].view(-1))
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.cpu().detach().numpy())
            show_bar.set_description("Epoch {0}, Loss {1:0.2f}"
                                     .format(epoch, np.mean(train_losses)))
        model_path = save_model_dir + "pretrainV1_{0}_{1:0.2f}_bert.pth".format(epoch, np.mean(train_losses))
        torch.save({"model": model.state_dict(), }, model_path)
    return

# class SBertDataSet(Dataset):
#     def __init__(self, train_path, test_path, type, max_seq_len=64):
#         super(SBertDataSet, self).__init__()
#         self.train_path = train_path
#         self.test_path = test_path
#         self.type = type
#         self.vocab = self.get_vocab()
#         self.max_seq_len = max_seq_len
#         self.seq_1, self.seq_2, self.label = self.load_data()
#         return
#
#     def get_vocab(self):
#         min_count = 5
#         test_data = pd.read_csv(self.train_path, sep="\t", names=["seq1", "seq2"])
#         train_data = pd.read_csv(self.test_path, sep="\t", names=["seq1", "seq2", "label"])
#
#         data = pd.concat([train_data[["seq1", "seq2"]], test_data[["seq1", "seq2"]]])
#         data = data['seq1'].append(data['seq2']).to_numpy()
#         special_tokens = ["[PAD]",
#                           "[UNK]",
#                           "[CLS]",
#                           "[SEP]",
#                           "[MASK]"]
#
#         vocab = defaultdict(int)
#         for seq in data:
#             if isinstance(seq, int):
#                 if str(seq) in vocab.keys():
#                     vocab[str(seq)] += 1
#                 else:
#                     vocab[str(seq)] = 0
#             else:
#                 for w in seq.split(' '):
#                     if w in vocab.keys():
#                         vocab[w] += 1
#                     else:
#                         vocab[w] = 0
#
#         vocab = {token: count for token, count in vocab.items() if count > min_count}
#         vocab = dict(sorted(vocab.items(), key=lambda item: -item[1]))
#         vocab = special_tokens + list(vocab.keys())
#         vocab = dict(zip(vocab, range(len(vocab))))
#         return vocab
#
#     def load_data(self):
#         seq_1 = []
#         seq_2 = []
#         label = []
#         if self.type == 'train':
#             with open(self.train_path, 'r') as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     data = line.strip().split('\t')
#                     s1 = data[0].split(' ')
#                     s2 = data[1].split(' ')
#                     seq_1.append(s1)
#                     seq_2.append(s2)
#                     label.append(int(data[2]))
#         else:
#             with open(self.test_path, 'r') as f:
#                 lines = f.readlines()
#                 for line in lines:
#                     data = line.strip().split('\t')
#                     s1 = data[0].split(' ')
#                     s2 = data[1].split(' ')
#                     seq_1.append(s1)
#                     seq_2.append(s2)
#                     label.append(2)
#         return seq_1, seq_2, label
#
#     def __len__(self):
#         return len(self.seq_1)
#
#     def __getitem__(self, index):
#         s1 = self.seq_1[index]
#         s2 = self.seq_2[index]
#         label = self.label[index]
#
#         s1 = ['[CLS]'] + s1 + ['[SEP]']
#         s2 = ['[CLS]'] + s2 + ['[SEP]']
#         s1 = s1[:self.max_seq_len]
#         s2 = s2[:self.max_seq_len]
#
#         s1_segment_label = [1] * len(s1)
#         s2_segment_label = [1] * len(s2)
#
#         s1_padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(s1))
#         s2_padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(s2))
#
#         s1_ids = []
#         for i in range(len(s1)):
#             s1_ids.append(self.vocab.get(s1[i], self.vocab['[UNK]']))
#
#         s2_ids = []
#         for i in range(len(s2)):
#             s2_ids.append(self.vocab.get(s2[i], self.vocab['[UNK]']))
#
#         s1_ids.extend(s1_padding)
#         s2_ids.extend(s2_padding)
#         s1_segment_label.extend(s1_padding)
#         s2_segment_label.extend(s2_padding)
#
#         s1_attention_mask = [1] * (len(s1_ids) - len(s1_padding)) + [0] * len(s1_padding)
#         s2_attention_mask = [1] * (len(s2_ids) - len(s2_padding)) + [0] * len(s2_padding)
#
#         return {
#                    "s1_ids": torch.tensor(s1_ids),
#                    "s2_ids": torch.tensor(s2_ids),
#                    "s1_segment_label": torch.tensor(s1_segment_label),
#                    "s2_segment_label": torch.tensor(s2_segment_label),
#                    "s1_attention_mask": torch.tensor(s1_attention_mask),
#                    "s2_attention_mask": torch.tensor(s2_attention_mask)
#                }, label
#
#
# class SBert(Module):
#     def __init__(self, bert, embedding_dim, n_classes):
#         super(SBert, self).__init__()
#         self.bert = bert
#         self.embedding_dim = embedding_dim
#         self.n_classes = n_classes
#         self.fc = Linear(self.embedding_dim * 3, self.n_classes)
#
#     def forward(self, s1, s2):
#         s1 = self.bert(input_ids=s1['input_ids'],
#                        attention_mask=s1['attention_mask'],
#                        token_type_ids=s1['token_type_ids'])
#
#         s2 = self.bert(input_ids=s2['input_ids'],
#                        attention_mask=s2['attention_mask'],
#                        token_type_ids=s2['token_type_ids'])
#
#         s1 = torch.mean(s1['last_hidden_state'], dim=1)
#         s2 = torch.mean(s2['last_hidden_state'], dim=1)
#
#         output = self.fc(torch.cat([s1, s2, torch.abs(s1 - s2)], dim=1))
#         return output


# def evaluate(model, val_loader, device):
#     model.eval()
#     predicts = []
#     labels = []
#     for input, label in val_loader:
#         s1 = {}
#         s2 = {}
#         s1['input_ids'] = input['s1_ids'].to(device)
#         s2['input_ids'] = input['s2_ids'].to(device)
#         s1['attention_mask'] = input['s1_attention_mask'].to(device)
#         s2['attention_mask'] = input['s2_attention_mask'].to(device)
#         s1['token_type_ids'] = input['s1_segment_label'].to(device)
#         s2['token_type_ids'] = input['s2_segment_label'].to(device)
#         label = label.to(device)
#         output = torch.softmax(model(s1, s2), dim=1)[:, 1]
#         predicts.append(output.cpu().detach().numpy())
#         labels.append(label.cpu().detach().numpy())
#     labels = np.concatenate(labels)
#     predicts = np.concatenate(predicts)
#     auc_score = roc_auc_score(labels, predicts)
#     return auc_score


# def train(train_path, test_path):
#     num_epochs = 100
#     batch_size = 64
#     learning_rate = 2e-5
#     max_seq_len = 64
#     pretrained_model_path = "/content/gdrive/MyDrive/model/pretrain_0_0.25_bert.pth"
#     save_model_dir = "/content/gdrive/MyDrive/model/"
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
#     embedding_dim = 768
#     n_classes = 2
#
#     dataset = SBertDataSet(train_path, test_path, 'train')
#
#     train_size = int(0.9 * len(dataset))
#     val_size = len(dataset) - train_size
#
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#
#     vocab_size = len(dataset.vocab)
#     pretrained_model = PretrainedBERT(embedding_size=vocab_size,
#                                       embedding_dim=embedding_dim,
#                                       max_len=max_seq_len)
#     pretrained_model.load_state_dict(torch.load(pretrained_model_path)['model'])
#     model = SBert(pretrained_model.bert, embedding_dim, n_classes)
#     model = model.to(device)
#
#     criterion = CrossEntropyLoss()
#     optimizer = Adam(model.parameters(), lr=learning_rate)
#
#     best_auc_score = 0.0
#     last_model_path = None
#
#     for i, epoch in enumerate(range(num_epochs)):
#         train_losses = []
#         show_bar = tqdm(train_loader)
#         for input, label in show_bar:
#             s1 = {}
#             s2 = {}
#             s1['input_ids'] = input['s1_ids'].to(device)
#             s2['input_ids'] = input['s2_ids'].to(device)
#             s1['attention_mask'] = input['s1_attention_mask'].to(device)
#             s2['attention_mask'] = input['s2_attention_mask'].to(device)
#             s1['token_type_ids'] = input['s1_segment_label'].to(device)
#             s2['token_type_ids'] = input['s2_segment_label'].to(device)
#             label = label.to(device)
#
#             optimizer.zero_grad()
#             output = model(s1, s2)
#             train_loss = criterion(output, label)
#             train_loss.backward()
#             optimizer.step()
#             train_losses.append(train_loss.cpu().detach().numpy())
#             show_bar.set_description("Epoch {0}, Loss {1:0.2f}"
#                                      .format(epoch, np.mean(train_losses)))
#
#         val_auc_score = evaluate(model, val_loader, device)
#         print("*" * 50)
#         print("The val AUC Score is {0}".format(val_auc_score))
#
#         if val_auc_score > best_auc_score:
#             best_auc_score = val_auc_score
#             model_path = save_model_dir + "Sentence_Bert_{0}_{1:0.2f}.pth".format(epoch, best_auc_score)
#             torch.save({
#                 "model": model.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#                 "epoch": epoch
#             }, model_path)
#             if last_model_path is not None:
#                 os.remove(last_model_path)
#             last_model_path = model_path
#     return


# def predict(train_path, test_path, result_path):
#     batch_size = 64
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
#     embedding_dim = 768
#     n_classes = 2
#     max_seq_len = 64
#     pretrained_model_path = "/content/gdrive/MyDrive/model/pretrain_0_0.25_bert.pth"
#     model_path = "/content/gdrive/MyDrive/model/Sentence_Bert_0_0.87.pth"
#
#     dataset = SBertDataSet(train_path, test_path, 'test')
#     vocab_size = len(dataset.vocab)
#     data_loader = DataLoader(dataset, batch_size=batch_size)
#
#     pretrained_model = PretrainedBERT(embedding_size=vocab_size,
#                                       embedding_dim=embedding_dim,
#                                       max_len=max_seq_len)
#     pretrained_model.load_state_dict(torch.load(pretrained_model_path)['model'])
#
#     model = SBert(pretrained_model.bert, embedding_dim, n_classes)
#     model.load_state_dict(torch.load(model_path)['model'])
#     model = model.to(device)
#
#     result = []
#     for input, _ in tqdm(data_loader):
#         s1 = {}
#         s2 = {}
#         s1['input_ids'] = input['s1_ids'].to(device)
#         s2['input_ids'] = input['s2_ids'].to(device)
#         s1['attention_mask'] = input['s1_attention_mask'].to(device)
#         s2['attention_mask'] = input['s2_attention_mask'].to(device)
#         s1['token_type_ids'] = input['s1_segment_label'].to(device)
#         s2['token_type_ids'] = input['s2_segment_label'].to(device)
#         output = torch.softmax(model(s1, s2), dim=1)[:, 1]
#         result.append(output.cpu().detach().numpy())
#     result = np.concatenate(result)
#     result = pd.DataFrame(result, columns=['label'])
#     result['label'].to_csv(result_path, sep='\t', index=0, header=False)
#     return

class SBertDataSetV1(Dataset):
    def __init__(self, train_path, test_path, type, max_seq_len=64):
        super(SBertDataSetV1, self).__init__()
        self.train_path = train_path
        self.test_path = test_path
        self.type = type
        self.vocab = self.get_vocab()
        self.max_seq_len = max_seq_len
        self.seq_1, self.seq_2, self.label = self.load_data()
        return

    def get_vocab(self):
        min_count = 5
        test_data = pd.read_csv(self.train_path, sep="\t", names=["seq1", "seq2"])
        train_data = pd.read_csv(self.test_path, sep="\t", names=["seq1", "seq2", "label"])

        data = pd.concat([train_data[["seq1", "seq2"]], test_data[["seq1", "seq2"]]])
        data = data['seq1'].append(data['seq2']).to_numpy()
        special_tokens = ["[PAD]",
                          "[UNK]",
                          "[CLS]",
                          "[SEP]",
                          "[MASK]"]

        vocab = defaultdict(int)
        for seq in data:
            if isinstance(seq, int):
                if str(seq) in vocab.keys():
                    vocab[str(seq)] += 1
                else:
                    vocab[str(seq)] = 0
            else:
                for w in seq.split(' '):
                    if w in vocab.keys():
                        vocab[w] += 1
                    else:
                        vocab[w] = 0

        vocab = {token: count for token, count in vocab.items() if count > min_count}
        vocab = dict(sorted(vocab.items(), key=lambda item: -item[1]))
        vocab = special_tokens + list(vocab.keys())
        vocab = dict(zip(vocab, range(len(vocab))))
        return vocab

    def load_data(self):
        seq_1 = []
        seq_2 = []
        label = []
        if self.type == 'train':
            with open(self.train_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split('\t')
                    s1 = data[0].split(' ')
                    s2 = data[1].split(' ')
                    seq_1.append(s1)
                    seq_2.append(s2)
                    label.append(int(data[2]))
        else:
            with open(self.test_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split('\t')
                    s1 = data[0].split(' ')
                    s2 = data[1].split(' ')
                    seq_1.append(s1)
                    seq_2.append(s2)
                    label.append(2)
        return seq_1, seq_2, label

    def __len__(self):
        return len(self.seq_1)

    def __getitem__(self, index):
        s1 = self.seq_1[index]
        s2 = self.seq_2[index]
        label = self.label[index]

        seq = ['[CLS]'] + s1 + ['[SEP]'] + s2 + ['[SEP]']
        seq = seq[:self.max_seq_len]

        segment_label = [0] * (len(s1) + 2) + [1] * (len(s2) + 1)

        padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(seq))

        seq_ids = []
        for i in range(len(seq)):
            seq_ids.append(self.vocab.get(seq[i], self.vocab['[UNK]']))

        seq_ids.extend(padding)
        segment_label.extend(padding)

        attention_mask = [1] * (len(seq_ids) - len(padding)) + [0] * len(padding)

        return {
                   "input_ids": torch.tensor(seq_ids),
                   "token_type_ids": torch.tensor(segment_label),
                   "attention_mask": torch.tensor(attention_mask)
               }, label

class SBertV1(Module):
    def __init__(self, bert, embedding_dim, n_classes):
        super(SBertV1, self).__init__()
        self.bert = bert
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.fc = Linear(self.embedding_dim * 2, self.n_classes)

    def forward(self, seq):
        seq = self.bert(input_ids=seq['input_ids'],
                        attention_mask=seq['attention_mask'],
                        token_type_ids=seq['token_type_ids'])
        output = self.fc(torch.cat([seq['last_hidden_state'][:,0,:], torch.mean(seq['last_hidden_state'], dim=1)], dim=1))
        return output


def evaluateV1(model, val_loader, device):
    model.eval()
    predicts = []
    labels = []
    for input, label in val_loader:
        label = label.to(device)
        input['input_ids'] = input['input_ids'].to(device)
        input['attention_mask'] = input['attention_mask'].to(device)
        input['token_type_ids'] = input['token_type_ids'].to(device)
        output = torch.softmax(model(input), dim=1)[:, 1]
        predicts.append(output.cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())
    labels = np.concatenate(labels)
    predicts = np.concatenate(predicts)
    auc_score = roc_auc_score(labels, predicts)
    return auc_score

def trainV1(train_path, test_path):
    num_epochs = 100
    batch_size = 64
    learning_rate = 2e-5
    max_seq_len = 64
    pretrained_model_path = "/content/gdrive/MyDrive/model/pretrain_0_0.25_bert.pth"
    save_model_dir = "/content/gdrive/MyDrive/model/"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    embedding_dim = 768
    n_classes = 2

    dataset = SBertDataSetV1(train_path, test_path, 'train')

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    vocab_size = len(dataset.vocab)
    pretrained_model = PretrainedBERT(embedding_size=vocab_size,
                                      embedding_dim=embedding_dim,
                                      max_len=max_seq_len)
    pretrained_model.load_state_dict(torch.load(pretrained_model_path)['model'])
    model = SBertV1(pretrained_model.bert, embedding_dim, n_classes)
    model = model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_auc_score = 0.0
    last_model_path = None

    for i, epoch in enumerate(range(num_epochs)):
        train_losses = []
        show_bar = tqdm(train_loader)
        for input, label in show_bar:
            label = label.to(device)
            input['input_ids'] = input['input_ids'].to(device)
            input['attention_mask'] = input['attention_mask'].to(device)
            input['token_type_ids'] = input['token_type_ids'].to(device)
            optimizer.zero_grad()
            output = model(input)
            train_loss = criterion(output, label)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.cpu().detach().numpy())
            show_bar.set_description("Epoch {0}, Loss {1:0.2f}"
                                     .format(epoch, np.mean(train_losses)))

        val_auc_score = evaluateV1(model, val_loader, device)
        print("*" * 50)
        print("The val AUC Score is {0}".format(val_auc_score))

        if val_auc_score > best_auc_score:
            best_auc_score = val_auc_score
            model_path = save_model_dir + "Sentence_Bert_{0}_{1:0.2f}.pth".format(epoch, best_auc_score)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, model_path)
            if last_model_path is not None:
                os.remove(last_model_path)
            last_model_path = model_path
    return

def predictV1(train_path, test_path, result_path):
    batch_size = 64
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    embedding_dim = 768
    n_classes = 2
    max_seq_len = 64
    pretrained_model_path = "/content/gdrive/MyDrive/model/pretrain_0_0.25_bert.pth"
    model_path = "/content/gdrive/MyDrive/model/Sentence_Bert_0_0.87.pth"

    dataset = SBertDataSetV1(train_path, test_path, 'test')
    vocab_size = len(dataset.vocab)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    pretrained_model = PretrainedBERT(embedding_size=vocab_size,
                                      embedding_dim=embedding_dim,
                                      max_len=max_seq_len)
    pretrained_model.load_state_dict(torch.load(pretrained_model_path)['model'])

    model = SBertV1(pretrained_model.bert, embedding_dim, n_classes)
    model.load_state_dict(torch.load(model_path)['model'])
    model = model.to(device)

    result = []
    for input, _ in tqdm(data_loader):
        input['input_ids'] = input['input_ids'].to(device)
        input['attention_mask'] = input['attention_mask'].to(device)
        input['token_type_ids'] = input['token_type_ids'].to(device)
        output = torch.softmax(model(input), dim=1)[:, 1]
        result.append(output.cpu().detach().numpy())
    result = np.concatenate(result)
    result = pd.DataFrame(result, columns=['label'])
    result['label'].to_csv(result_path, sep='\t', index=0, header=False)
    return


