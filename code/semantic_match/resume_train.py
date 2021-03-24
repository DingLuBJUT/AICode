import os
import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from model import PretrainedBERT
from dataset import BertDataset
from train import evaluate


def k_fold_train(data, vocab, k_fold=5, is_resume=True, resume_path=None, start_k_fold=0):
    num_epochs = 100
    batch_size = 32
    early_stopping = 5
    learning_rate = 2e-5
    save_model_dir = "/content/gdrive/MyDrive/model/"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    max_len = 64
    embedding_dim = 768

    model = PretrainedBERT(embedding_size=len(vocab),
                           embedding_dim=embedding_dim,
                           max_len=max_len)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    start_epoch = None
    if is_resume is True:
        resume_data = torch.load(resume_path)
        model.load_state_dict(resume_data['model'])
        optimizer = model['optimizer']
        start_epoch = resume_data['epoch']

    model = model.to(device)

    skf = StratifiedKFold(n_splits=k_fold)
    index = list(range(len(data)))
    num_fold = 1
    for train_index, val_index in skf.split(index, data['label'].to_numpy()):

        if num_fold < start_k_fold:
            num_fold += 1
            break

        train_data = data.iloc[train_index]
        val_data = data.iloc[val_index]
        train_dataset = BertDataset(train_data.values, vocab, max_seq_len=64, data_type='train')
        val_dataset = BertDataset(val_data.values, vocab, max_seq_len=64, data_type='train')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_auc_score = 0.0
        stopping_num = 0
        last_model_path = None
        for i, epoch in enumerate(range(start_epoch + 1, num_epochs)):
            train_losses = []
            show_bar = tqdm(train_loader)
            for input_data, _ in show_bar:
                input_data['input_ids'] = input_data['input_ids'].to(device)
                input_data['token_label'] = input_data['token_label'].to(device)
                input_data['token_type_ids'] = input_data['token_type_ids'].to(device)
                input_data['attention_mask'] = input_data['attention_mask'].to(device)
                optimizer.zero_grad()
                output = model(input_data)
                token_mask = (input_data['token_label'] != -1)
                train_loss = criterion(output[token_mask].view(-1, len(vocab)),
                                       input_data['token_label'][token_mask].view(-1))
                train_loss.backward()
                optimizer.step()
                train_losses.append(train_loss.cpu().detach().numpy())
                show_bar.set_description("K-Fold {0}, Epoch {1}, Loss {2:0.2f}"
                                         .format(num_fold, epoch, np.mean(train_losses)))

            val_auc_score = evaluate(model, val_loader, device)
            print("*" * 50)
            print("The val AUC Score is {0}".format(val_auc_score))

            if val_auc_score > best_auc_score:
                best_auc_score = val_auc_score
                model_path = save_model_dir + "model_{0}_{1}_{2:0.2f}.pth".format(num_fold, epoch, best_auc_score)
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                }, model_path)
                if last_model_path is not None:
                    os.remove(last_model_path)
                last_model_path = model_path
            else:
                stopping_num += 1
                if stopping_num >= early_stopping:
                    break
        num_fold += 1
        start_k_fold = 0
        start_epoch = -1


def train(data, vocab, is_resume_train=False, resume_path=None):
    num_epochs = 100
    batch_size = 32
    early_stopping = 5
    learning_rate = 2e-5
    save_model_dir = "/content/"
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    max_len = 64
    embedding_dim = 768

    model = PretrainedBERT(embedding_size=len(vocab),
                           embedding_dim=embedding_dim,
                           max_len=max_len)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    if is_resume_train is True:
        resume_data = torch.load(resume_path)
        model.load_state_dict(resume_data['model'])
        optimizer = resume_data['optimizer']
        start_epoch = resume_data['epoch']
    model = model.to(device)

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
    last_model_path = None
    for i, epoch in enumerate(range(start_epoch + 1, num_epochs)):
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
        else:
            stopping_num += 1
            if stopping_num >= early_stopping:
                break
    return
