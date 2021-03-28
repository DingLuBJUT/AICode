import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

import torch
from torch.utils.data import DataLoader

def predict(model_path, test_data, vocab, result_path, keep_index):

    batch_size = 64
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

    embedding_dim = 768
    max_len = 64
    model = PretrainedBERT(embedding_size=len(vocab),
                               embedding_dim=embedding_dim,
                               max_len=max_len,
                               keep_index=keep_index)
    model.load_state_dict(torch.load(model_path)['model'])
    model = model.to(device)

    test_dataset = BertDataset(test_data.values, vocab, max_seq_len=64, data_type='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    result = []
    for data in tqdm(test_loader):
        data['input_ids'] = data['input_ids'].to(device)
        data['token_type_ids'] = data['token_type_ids'].to(device)
        data['attention_mask'] = data['attention_mask'].to(device)
        predict_output = model(data)
        predict_output = predict_output[:, 0, 5:7].cpu().detach().numpy()
        predict_output = predict_output[:, 1] / (predict_output.sum(axis=1) + 1e-8)
        result.append(predict_output)
    result = np.concatenate(result)
    result = pd.DataFrame(result, columns=['label'])
    result['label'].to_csv(result_path, sep='\t', index=0, header=False)
    return
