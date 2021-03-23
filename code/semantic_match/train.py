import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from model import PretrainedBERT
from utils import get_vocab_dict
from dataset import BertDataset


def evaluate(model, val_loader, device):
    model.eval()
    predicts = []
    labels = []
    for data, _ in val_loader:
        data['input_ids'] = data['input_ids'].to(device)
        data['token_label'] = data['token_label'].to(device)
        data['token_type_ids'] = data['token_type_ids'].to(device)
        data['attention_mask'] = data['attention_mask'].to(device)
        output = model(data)
        predict = output[:, 0, 5:7].cpu().detach().numpy()
        predict = predict[:, 1] / (predict.sum(axis=1) + 1e-8)
        predicts.append(predict)
        labels.append(data['token_label'][:, 0].cpu().numpy() - 5)
    labels = np.concatenate(labels)
    predicts = np.concatenate(predicts)
    auc_score = roc_auc_score(labels, predicts)
    return auc_score


def train(data, vocab):
    num_epochs = 100
    batch_size = 32
    early_stopping = 5
    learning_rate = 2e-5
    save_model_path = "/content/"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    max_len = 64
    embedding_dim = 768

    model = PretrainedBERT(embedding_size=len(vocab),
                           embedding_dim=embedding_dim,
                           max_len=max_len)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_index, val_index = train_test_split(range(len(data)),
                                              test_size=0.1,
                                              shuffle=True)
    train_data = data.iloc[train_index]
    val_data = data.iloc[val_index]
    train_dataset = BertDataset(train_data.values, vocab, max_seq_len=64, data_type='train')
    val_dataset = BertDataset(val_data.values, vocab, max_seq_len=64, data_type='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    best_auc_score = 0.0
    stopping_num = 0
    for i, epoch in enumerate(range(num_epochs)):
        train_losses = []
        show_bar = tqdm(train_loader)
        for data, _ in show_bar:
            data['input_ids'] = data['input_ids'].to(device)
            data['token_label'] = data['token_label'].to(device)
            data['token_type_ids'] = data['token_type_ids'].to(device)
            data['attention_mask'] = data['attention_mask'].to(device)
            optimizer.zero_grad()
            output = model(data)
            token_mask = (data['token_label'] != -1)
            train_loss = criterion(output[token_mask].view(-1, len(vocab)),
                                   data['token_label'][token_mask].view(-1))
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.cpu().detach().numpy())
            show_bar.set_description("The Epoch is {0}, the Train Loss is {1:0.2f}"
                                     .format(epoch, np.mean(train_losses)))

        val_auc_score = evaluate(model, val_loader, device)
        print("*" * 50)
        print("The val AUC Score is {0}".format(val_auc_score))

        if val_auc_score > best_auc_score:
            stopping_num = 0
            best_auc_score = val_auc_score
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }, save_model_path + "model_{0}_{1:0.2f}.pth".format(epoch, best_auc_score))
        else:
            stopping_num += 1
            if stopping_num == early_stopping:
                break
    return


def main():
    test_data = pd.read_csv("../../data/semantic_match/gaiic_track3_round1_testA_20210228.tsv",
                       sep="\t",
                       names=["seq1", "seq2"])
    train_data = pd.read_csv("../../data/semantic_match/gaiic_track3_round1_train_20210228.tsv",
                        sep="\t",
                        names=["seq1", "seq2", "label"])
    data = pd.concat([train_data[["seq1", "seq2"]], test_data[["seq1", "seq2"]]])
    data = data['seq1'].append(data['seq2']).to_numpy()
    list_special_tokens = ["[PAD]",
                           "[UNK]",
                           "[CLS]",
                           "[SEP]",
                           "[MASK]",
                           "yes_similarity",
                           "no_similarity",
                           "un_certain"]
    vocab = get_vocab_dict(data, list_special_tokens)

    train(train_data, vocab)


    return

if __name__ == '__main__':
    main()