import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from utils import get_vocab_dict


class BertDataset(Dataset):
    def __init__(self, corpus, vocab, max_seq_len, data_type='train'):
        super(BertDataset, self).__init__()
        self.corpus = corpus
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.data_type = data_type
        return

    def __len__(self):
        return len(self.corpus)

    def random_mask_token(self, seq):
        tokens = seq.split(' ')
        tokens_label = []

        mask_prob = random.random()
        for i in range(len(tokens)):
            if mask_prob < 0.15:
                mask_prob /= 0.15
                if mask_prob < 0.8:
                    tokens[i] = '[MASK]'
                    tokens_label.append(self.vocab.get(tokens[i], self.vocab['[UNK]']))
                elif mask_prob < 0.9:
                    pass
                else:
                    pass



        return

    def __getitem__(self, index):
        if self.data_type == 'train':
            seq_1, seq_2, label = self.corpus[index]
            seq_1 = self.random_mask_token(seq_1)
            seq_1 = self.random_mask_token(seq_1)
        else:
            seq_1, seq_2 = self.corpus[index]

        seq = ['[CLS]'] + seq_1 + ['[SEP]'] + seq_2 + ['[SEP]']
        seq = seq[:self.max_seq_len]

        segment_label = [0] * (len(seq_1) + 2) + [1] * (len(seq_2) + 1)
        segment_label = segment_label[:self.max_seq_len]

        padding = ['[PAD]'] * (self.max_seq_len - len(seq))

        seq.extend(padding)
        segment_label.extend(padding)
        attention_mask = [1] * (len(seq) - len(padding)) + [0] * len(padding)
        return







        return



class BertDataset(Dataset):
    def __init__(self, corpus, vocab, max_seq_len, data_type='train'):
        super(BertDataset, self).__init__()
        self.corpus = corpus
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.data_type = data_type
        return

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        seq_1, seq_2, label, in_next_seq = self.get_seq(index)

        input_ids = ['[CLS]'] + seq_1 + ['[SEP]'] + seq_2 + ['[SEP]']
        segment_label = [0] * (len(seq_1) + 2) + [1] * (len(seq_2) + 1)

        input_ids = input_ids[:self.max_seq_len]
        segment_label = segment_label[:self.max_seq_len]

        padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(input_ids))
        attention_mask = [1] * len(input_ids) + [0] * len(padding)
        input_ids.extend(padding)
        segment_label.extend(padding)

        if self.data_type == 'train':
            seq_1, seq_label_1 = self.random_mask_seq(seq_1)
            seq_2, seq_label_2 = self.random_mask_seq(seq_2)

            pair_label = None
            if self.data_type == 'train':
                if label == 1:
                    pair_label = self.vocab['yes_similarity']
                elif label == 0:
                    pair_label = self.vocab['no_similarity']
                else:
                    pair_label = self.vocab['un_certain']
            elif self.data_type == 'test':
                pair_label = -1

            input_ids = [self.vocab['[CLS]']] + seq_1 + [self.vocab['[SEP]']] + seq_2 + [self.vocab['[SEP]']]

            token_label = [pair_label] + seq_label_1 + [self.vocab['[SEP]']] + seq_label_2 + [self.vocab['[SEP]']]
            token_label = token_label[:self.max_seq_len]
            padding_label = [-1] * (self.max_seq_len - len(input_ids))
            token_label.extend(padding_label)
            data = {
                "input_ids": np.array(input_ids),
                "token_label": np.array(token_label),
                "token_type_ids": np.array(segment_label),
                "attention_mask": np.array(attention_mask)
            }
            return data, label
        elif self.data_type == 'test':
            data = {
                "input_ids": np.array(input_ids),
                "token_type_ids": np.array(segment_label),
                "attention_mask": np.array(attention_mask)
            }
            return data, label

    def get_seq(self, index):
        seq_1 = None
        seq_2 = None
        label = None

        if self.data_type == 'train':
            seq_1, seq_2, label = self.corpus[index]
            if random.random() > 0.5:
                return seq_1, seq_2, label, 0
            else:
                return seq_2, seq_1, label, 0
        elif self.data_type == 'test':
            seq_1, seq_2 = self.corpus[index]
            label = 2
            return seq_1, seq_2, label, 0

    def random_mask_seq(self, seq):
        tokens = seq.split(' ')
        tokens_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                random_token = None
                # 80%
                if prob < 0.8:
                    random_token = '[MASK]'
                # 90%
                elif prob < 0.9:
                    random_token = random.choice(self.vocab.keys())
                # 10%
                else:
                    random_token = token
                tokens[i] = self.vocab.get(random_token, self.vocab['[UNK]'])
                tokens_label.append(self.vocab.get(token, self.vocab['[UNK]']))
            else:
                tokens[i] = self.vocab.get(token, self.vocab['[UNK]'])
                tokens_label.append(-1)
        return tokens, tokens_label


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
    train_dataset = BertDataset(train_data.values, vocab, max_seq_len=64, data_type='train')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for data in train_loader:
        print(data)
        break
    return

if __name__ == '__main__':
    main()