import random
import numpy as np
from torch.utils.data import Dataset


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
        segment_label = [0] * (len(seq_1) + 2) + [1] * (len(seq_2) + 1)

        input_ids = input_ids[:self.max_seq_len]
        token_label = token_label[:self.max_seq_len]
        segment_label = segment_label[:self.max_seq_len]

        padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(input_ids))
        padding_label = [-1] * (self.max_seq_len - len(input_ids))
        attention_mask = [1] * len(input_ids) + [0] * len(padding)

        input_ids.extend(padding)
        token_label.extend(padding_label)
        segment_label.extend(padding)

        data = {
            "input_ids": np.array(input_ids),
            "token_label": np.array(token_label),
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
        elif self.data_type == 'test':
            seq_1, seq_2 = self.corpus[index]
            label = 2

        if random.random() > 0.5:
            return seq_1, seq_2, label, 0
        else:
            return seq_2, seq_1, label, 0

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
