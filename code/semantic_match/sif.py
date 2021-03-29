import pandas as pd
from torch.utils.data import Dataset
from collections import Counter
from utils import get_vocab_dict
from utils import get_token_weight


class SIF(Dataset):
    def __init__(self, model, data_iter, device, token_weight):
        super(SIF, self).__init__()
        self.model = model
        self.data_iter = data_iter
        self.device = device
        self.token_weight = token_weight

        return

    def __len__(self):
        return len(self.data_iter)


    def forward(self):
        for input_data, _ in self.data_iter:
            input_data['input_ids'] = input_data['input_ids'].to(self.device)
            input_data['token_label'] = input_data['token_label'].to(self.device)
            input_data['token_type_ids'] = input_data['token_type_ids'].to(self.device)
            input_data['attention_mask'] = input_data['attention_mask'].to(self.device)

            output = self.model(input_data)
        return


def main():
    # test_data = pd.read_csv("/Users/dingjunlu/PycharmProjects/AICode/data/semantic_match/gaiic_track3_round1_testA_20210228.tsv", sep="\t", names=["seq1", "seq2"])
    # train_data = pd.read_csv("/Users/dingjunlu/PycharmProjects/AICode/data/semantic_match/gaiic_track3_round1_train_20210228.tsv", sep="\t", names=["seq1", "seq2", "label"])
    # sentences = pd.concat([train_data[["seq1"]], train_data[["seq2"]], test_data[["seq1"]], test_data[["seq2"]]])
    # for sent in sentences.to_numpy():
    #     print(sent)

    test_path = "/Users/dingjunlu/PycharmProjects/AICode/data/semantic_match/gaiic_track3_round1_testA_20210228.tsv"
    train_path = "/Users/dingjunlu/PycharmProjects/AICode/data/semantic_match/gaiic_track3_round1_train_20210228.tsv"

    total_tokens = []
    with open(train_path, 'r') as f:
        for line in f.readlines():
            s1, s2 = line.strip().split('\t')[:2]
            total_tokens.extend(s1.split(' '))
            total_tokens.extend(s2.split(' '))

    with open(test_path, 'r') as f:
        for line in f.readlines():
            s1, s2 = line.strip().split('\t')[:2]
            total_tokens.extend(s1.split(' '))
            total_tokens.extend(s2.split(' '))
    weight = Counter(total_tokens)
    weight['[PAD]'] = 0
    weight['[UNK]'] = 0
    weight['[CLS]'] = 0
    weight['[SEP]'] = 0
    weight['[MASK]'] = 0
    weight['yes_similarity'] = 0
    weight['no_similarity'] = 0
    weight['un_certain'] = 0
    weight = {k: 1 / (1 + v) for k, v in weight.items()}
    print(weight)
    return


def main():
    test_data = pd.read_csv("/Users/dingjunlu/PycharmProjects/AICode/data/semantic_match/gaiic_track3_round1_testA_20210228.tsv",sep="\t",names=["seq1", "seq2"])
    train_data = pd.read_csv("/Users/dingjunlu/PycharmProjects/AICode/data/semantic_match/gaiic_track3_round1_train_20210228.tsv",sep="\t",names=["seq1", "seq2", "label"])

    data = pd.concat([train_data[["seq1", "seq2"]], test_data[["seq1", "seq2"]]])
    data = data['seq1'].append(data['seq2']).to_numpy()
    special_tokens = ["[PAD]",
                       "[UNK]",
                       "[CLS]",
                       "[SEP]",
                       "[MASK]",
                       "yes_similarity",
                       "no_similarity",
                       "un_certain"]
    vocab = get_vocab_dict(data, special_tokens)
    weight = get_token_weight("/Users/dingjunlu/PycharmProjects/AICode/data/semantic_match/gaiic_track3_round1_train_20210228.tsv",
                             "/Users/dingjunlu/PycharmProjects/AICode/data/semantic_match/gaiic_track3_round1_testA_20210228.tsv",
                             vocab, special_tokens, param=1e-4)
    print(weight)



if __name__ == '__main__':
    main()