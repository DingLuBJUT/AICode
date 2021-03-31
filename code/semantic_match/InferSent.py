import os
import pickle
import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
from torch.nn import Module, Linear, Softmax, Sequential,CrossEntropyLoss
from torch.utils.data import Dataset,DataLoader,random_split
from torch.optim import Adam


# class InferSentDataSet(Dataset):
#     def __init__(self, data):
#         super(InferSentDataSet, self).__init__()
#         self.data = data
#         return
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         s1, s2, label = self.data[index]
#         s1 = torch.tensor(s1)
#         s2 = torch.tensor(s2)
#         label = torch.tensor(label)
#         return s1, s2, label


# class InferSentModel(Module):
#     def __init__(self, embedding_dim, n_classes):
#         super(InferSentModel, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.n_classes = n_classes
#
#         self.classifier = Sequential(
#             Linear(self.embedding_dim * 2, 512),
#             Linear(512, 512),
#             Linear(512, self.n_classes)
#         )
#         return
#
#     def forward(self, s1, s2):
#         input = torch.cat([torch.abs(s1 - s2), s1 * s2], dim=1)
#         output = self.classifier(input)
#         return output

class InferSentDataSet(Dataset):
    def __init__(self, data_path, data_type, vocab, token_weight, max_seq_len=64):
        super(InferSentDataSet, self).__init__()
        self.data_path = data_path
        self.data_type = data_type
        self.vocab = vocab
        self.token_weight = token_weight
        self.max_seq_len = max_seq_len
        self.seq_1, self.seq_2, self.label = self.load_data()
        return

    def load_data(self):
        seq_1 = []
        seq_2 = []
        label = []
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split('\t')
                s1 = data[0].split(' ')
                s2 = data[1].split(' ')
                seq_1.append(s1)
                seq_2.append(s2)
                if self.data_type == 'train':
                    label.append(int(data[2]))
                else:
                    label.append(2)

        return seq_1, seq_2, label

    def __len__(self):
        return len(self.seq_1)

    def __getitem__(self, index):
        s1 = self.seq_1[index]
        s2 = self.seq_2[index]
        label = self.label[index]

        s1 = ['[CLS]'] + s1 + ['[SEP]']
        s2 = ['[CLS]'] + s2 + ['[SEP]']
        s1 = s1[:self.max_seq_len]
        s2 = s2[:self.max_seq_len]

        s1_segment_label = [1] * len(s1)
        s2_segment_label = [1] * len(s2)

        s1_padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(s1))
        s2_padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(s2))

        s1_ids = []
        for i in range(len(s1)):
            s1_ids.append(self.vocab.get(s1[i], self.vocab['[UNK]']))

        s2_ids = []
        for i in range(len(s2)):
            s2_ids.append(self.vocab.get(s2[i], self.vocab['[UNK]']))

        s1_ids.extend(s1_padding)
        s2_ids.extend(s2_padding)
        s1_segment_label.extend(s1_padding)
        s2_segment_label.extend(s2_padding)

        s1_attention_mask = [1] * (len(s1_ids) - len(s1_padding)) + [0] * len(s1_padding)
        s2_attention_mask = [1] * (len(s2_ids) - len(s2_padding)) + [0] * len(s2_padding)

        s1_weight = [self.token_weight.get(id) for id in s1_ids]
        s2_weight = [self.token_weight.get(id) for id in s2_ids]

        return {
                   "s1_ids": torch.tensor(s1_ids),
                   "s2_ids": torch.tensor(s2_ids),
                   "s1_segment_label": torch.tensor(s1_segment_label),
                   "s2_segment_label": torch.tensor(s2_segment_label),
                   "s1_attention_mask": torch.tensor(s1_attention_mask),
                   "s2_attention_mask": torch.tensor(s2_attention_mask),
                   "s1_weight": torch.tensor(s1_weight),
                   "s2_weight": torch.tensor(s2_weight)
               }, label


def make_sentence_embedding(input_path, output_path, data_type, vocab, token_weight, model, device):
    dataset = InferSentDataSet(input_path, data_type, vocab, token_weight)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    total_s1_emb = []
    total_s2_emb = []
    total_label_emb = []

    for input_data, label in data_loader:
        s1_ids = input_data['s1_ids'].to(device)
        s2_ids = input_data['s2_ids'].to(device)
        s1_attention_mask = input_data['s1_attention_mask'].to(device)
        s2_attention_mask = input_data['s2_attention_mask'].to(device)
        s1_segment_label = input_data['s1_segment_label'].to(device)
        s2_segment_label = input_data['s2_segment_label'].to(device)
        s1_weight = input_data['s1_weight'].to(device)
        s2_weight = input_data['s2_weight'].to(device)

        s1_output = model.bert(input_ids=s1_ids,
                               attention_mask=s1_attention_mask,
                               token_type_ids=s1_segment_label)

        s2_output = model.bert(input_ids=s2_ids,
                               attention_mask=s2_attention_mask,
                               token_type_ids=s2_segment_label)

        s1_emb = torch.mean(s1_weight.unsqueeze(2) * s1_output['last_hidden_state'], 1)
        s2_emb = torch.mean(s2_weight.unsqueeze(2) * s2_output['last_hidden_state'], 1)

        total_s1_emb.append(s1_emb.cpu().detach().numpy())
        total_s2_emb.append(s2_emb.cpu().detach().numpy())
        total_label_emb.append(label.cpu().detach().numpy())

    total_s1_emb = np.concatenate(total_s1_emb)
    total_s2_emb = np.concatenate(total_s2_emb)
    total_label_emb = np.concatenate(total_label_emb)

    with open(output_path, 'wb') as f:
        pickle.dump([total_s1_emb, total_s2_emb, total_label_emb], f)
    return


class InferSentModel(Module):
    def __init__(self, num_embeddings, embedding_dim, token_weight, pretrained_embedding, n_classes):
        super(InferSentModel, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.word_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
        self.word_embedding = pretrained_embedding
        self.token_weight = token_weight
        self.n_classes = n_classes

        self.classifier = Sequential(
            Linear(self.embedding_dim * 2, 512),
            Linear(512, 512),
            Linear(512, self.n_classes)
        )
        return

    def forward(self, s1, s2, w1, w2):

        s1_emb = w1.unsqueeze(dim=2) * self.word_embedding(s1)
        s2_emb = w2.unsqueeze(dim=2) * self.word_embedding(s2)
        s1_emb = torch.mean(s1_emb, dim=1)
        s2_emb = torch.mean(s2_emb, dim=1)

        combine_seq = torch.cat([torch.abs(s1_emb - s2_emb), s1_emb * s2_emb], dim=1)
        output = self.classifier(combine_seq)
        return output




def evaluate(model, val_loader, device):
    model.eval()
    predicts = []
    labels = []
    for s1, s2, label in val_loader:
        s1 = s1.to(device)
        s2 = s2.to(device)
        output = torch.softmax(model(s1, s2), dim=1)
        predicts.append(output.cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())
    labels = np.concatenate(labels)
    predicts = np.concatenate(predicts)
    auc_score = roc_auc_score(labels, predicts)
    return auc_score


def train(data_path):
    num_epochs = 100
    batch_size = 32
    learning_rate = 2e-3
    save_model_dir = ""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    embedding_dim = 768
    n_classes = 2

    model = InferSentModel(embedding_dim=embedding_dim,
                           n_classes=n_classes)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    data = pickle.load(open(data_path, 'rb'))
    train_index, val_index = train_test_split(range(len(data)),
                                              test_size=0.1,
                                              shuffle=True)
    model = model.to(device)

    train_data = data[train_index]
    val_data = data[val_index]
    train_dataset = InferSentDataSet(train_data)
    val_dataset = InferSentDataSet(val_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_auc_score = 0.0
    last_model_path = None

    for i, epoch in enumerate(range(num_epochs)):
        train_losses = []
        show_bar = tqdm(train_loader)
        for s1, s2, label in show_bar:
            s1 = s1.to(device)
            s2 = s2.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output = model(s1, s2)
            train_loss = criterion(output,label)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.cpu().detach().numpy())
            show_bar.set_description("Epoch {0}, Loss {1:0.2f}"
                                     .format(epoch, np.mean(train_losses)))

        val_auc_score = evaluate(model, val_loader, device)
        print("*" * 50)
        print("The val AUC Score is {0}".format(val_auc_score))

        if val_auc_score > best_auc_score:
            best_auc_score = val_auc_score
            model_path = save_model_dir + "model_{0}_{1:0.2f}.pth".format(epoch, best_auc_score)
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, model_path)
            if last_model_path is not None:
                os.remove(last_model_path)
            last_model_path = model_path