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

    def random_mask_token(self, seq):
        tokens = seq.split(' ')
        tokens_label = []

        for i in range(len(tokens)):
            mask_prob = random.random()
            if mask_prob < 0.15:
                tokens_label.append(self.vocab.get(tokens[i], self.vocab['[UNK]']))
                mask_prob /= 0.15
                if mask_prob < 0.8:
                    tokens[i] = '[MASK]'
                elif mask_prob < 0.9:
                    tokens[i] = random.choice(list(self.vocab.keys()))
                else:
                    tokens[i] = tokens[i]
            else:
                tokens_label.append(-1)
        return tokens, tokens_label

    def __getitem__(self, index):

        seq_1 = None
        seq_2 = None
        token_label = None
        label = None
        if self.data_type == 'train':
            if random.random() > 0.5:
                seq_1, seq_2, label = self.corpus[index]
            else:
                seq_2, seq_1, label = self.corpus[index]
            seq_1, seq_label_1 = self.random_mask_token(seq_1)
            seq_2, seq_label_2 = self.random_mask_token(seq_2)
            token_label = [label + 5] + seq_label_1 + [self.vocab['[SEP]']] + seq_label_2 + [self.vocab['[SEP]']]
            token_label = token_label[:self.max_seq_len]
            padding = [-1] * (self.max_seq_len - len(token_label))
            token_label.extend(padding)
        elif self.data_type == 'test':
            seq_1, seq_2 = self.corpus[index]
            seq_1 = seq_1.split(' ')
            seq_2 = seq_2.split(' ')

        seq = ['[CLS]'] + seq_1 + ['[SEP]'] + seq_2 + ['[SEP]']
        seq = seq[:self.max_seq_len]

        segment_label = [0] * (len(seq_1) + 2) + [1] * (len(seq_2) + 1)
        segment_label = segment_label[:self.max_seq_len]

        padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(seq))

        input_ids = []
        for i in range(len(seq)):
            input_ids.append(self.vocab.get(seq[i], self.vocab['[UNK]']))

        input_ids.extend(padding)
        segment_label.extend(padding)
        attention_mask = [1] * (len(input_ids) - len(padding)) + [0] * len(padding)

        if self.data_type == 'train':
            return {
                       "input_ids": np.array(input_ids),
                       "token_label": np.array(token_label),
                       "token_type_ids": np.array(segment_label),
                       "attention_mask": np.array(attention_mask)
                   }, label
        else:
            return {
                "input_ids": np.array(input_ids),
                "token_type_ids": np.array(segment_label),
                "attention_mask": np.array(attention_mask)
            }
