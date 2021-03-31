
import os
import pickle
import numpy as np
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torch
from torch.nn import Module, Linear, Softmax, Sequential,CrossEntropyLoss
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam





class InferSentDataSet(Dataset):
    def __init__(self,data):
        super(InferSentDataSet, self).__init__()
        self.data = data
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        s1, s2, label =  self.data[index]
        s1 = torch.tensor(s1)
        s2 = torch.tensor(s2)
        label = torch.tensor(label)
        return s1, s2, label


class InferSentModel(Module):
    def __init__(self, embedding_dim, n_classes):
        super(InferSentModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_classes = n_classes

        self.classifier = Sequential(
            Linear(self.embedding_dim * 2, 512),
            Linear(512, 512),
            Linear(512, self.n_classes)
        )
        return

    def forward(self, s1, s2):
        input = torch.cat([torch.abs(s1 - s2), s1 * s2], dim=1)
        output = self.classifier(input)
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












