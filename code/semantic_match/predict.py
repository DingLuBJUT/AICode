import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm

import torch
from torch.utils.data import DataLoader

from model import PretrainedBERT
from utils import get_vocab_dict
from dataset import BertDataset


def predict(model_path, test_data, vocab, result_path):

    batch_size = 64
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

    embedding_dim = 768
    max_len = 64
    model = PretrainedBERT(embedding_size=len(vocab),
                           embedding_dim=embedding_dim,
                           max_len=max_len)
    model.load_state_dict(torch.load(model_path)['model'])
    model = model.to(device)

    test_dataset = BertDataset(test_data, vocab, max_seq_len=64, data_type='test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    result = []
    for data, _ in tqdm(test_loader):
        data['input_ids'] = data['input_ids'].to(device)
        data['token_label'] = data['token_label'].to(device)
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


# def main():
#     test = pd.read_csv("../data/gaiic_track3_round1_testA_20210228.tsv",
#                        sep="\t",
#                        names=["seq1", "seq2"])
#
#     train = pd.read_csv("../data/gaiic_track3_round1_train_20210228.tsv",
#                         sep="\t",
#                         names=["seq1", "seq2", "label"])
#     data = pd.concat([train[["seq1", "seq2"]], test[["seq1", "seq2"]]])
#     data = data['seq1'].append(data['seq2']).to_numpy()
#     list_special_tokens = ["[PAD]",
#                            "[UNK]",
#                            "[CLS]",
#                            "[SEP]",
#                            "[MASK]",
#                            "yes_similarity",
#                            "no_similarity",
#                            "un_certain"]
#     vocab = get_vocab_dict(data, list_special_tokens)
#
#     # test_data = test[["seq1", "seq2"]].values
#     # test_dataset = BertDataset(test_data, vocab, seq_len=64, data_type='test')
#     model_path = ""
#     result_path = ""
#     predict(model_path, test, vocab, result_path)
#     return
#
#
# if __name__ == '__main__':
#     main()