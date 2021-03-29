import numpy as np
from sklearn.metrics import roc_auc_score


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
