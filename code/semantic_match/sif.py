
# class SIF:
#     def __init__(self, input_path, output_path, vocab, token_weight, model, data_type='train', max_seq_len=64):
#         super(SIF, self).__init__()
#         self.data_path = input_path
#         self.output_path = output_path
#         self.data_type = data_type
#         self.vocab = vocab
#         self.token_weight = token_weight
#         self.model = model
#         self.max_seq_len = max_seq_len
#         self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
#         self.svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
#         return
#
#     def load_data(self):
#         seq_1 = []
#         seq_2 = []
#         label = []
#         with open(self.data_path, 'r') as f:
#             lines = f.readlines()
#             for line in lines:
#                 data = line.strip().split('\t')
#                 s1 = data[0].split(' ')
#                 s2 = data[1].split(' ')
#                 seq_1.append(s1)
#                 seq_2.append(s2)
#                 if self.data_type == 'train':
#                     label.append([data[2]])
#                 else:
#                     label.append([2])
#
#         return seq_1, seq_2, label
#
#     def prepare_data(self):
#         seq_1, seq_2, label = self.load_data()
#
#         bert_seq_1 = []
#         bert_seq_1_segment = []
#         bert_seq_1_mask = []
#
#         bert_seq_2 = []
#         bert_seq_2_segment = []
#         bert_seq_2_mask = []
#
#         seq_weight_1 = []
#         seq_weight_2 = []
#
#         for s1, s2 in zip(seq_1, seq_2):
#             s1 = [self.vocab.get(token, self.vocab['[UNK]']) for token in s1]
#             s2 = [self.vocab.get(token, self.vocab['[UNK]']) for token in s2]
#             s1 = [self.vocab['[CLS]']] + s1 + [self.vocab['[SEP]']]
#             s2 = [self.vocab['[CLS]']] + s2 + [self.vocab['[SEP]']]
#
#             s1_padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(s1))
#             s2_padding = [self.vocab['[PAD]']] * (self.max_seq_len - len(s2))
#
#             s1.extend(s1_padding)
#             s2.extend(s2_padding)
#
#             s1_segment = [1] * len(s1)
#             s2_segment = [1] * len(s2)
#
#             s1_mask = [1] * (len(s1) - len(s1_padding)) + [0] * len(s1_padding)
#             s2_mask = [1] * (len(s2) - len(s2_padding)) + [0] * len(s2_padding)
#
#             w1 = [self.token_weight.get(id) for id in s1]
#             w2 = [self.token_weight.get(id) for id in s2]
#             seq_weight_1.append(w1)
#             seq_weight_2.append(w2)
#
#             bert_seq_1.append(s1)
#             bert_seq_1_segment.append(s1_segment)
#             bert_seq_1_mask.append(s1_mask)
#
#             bert_seq_2.append(s2)
#             bert_seq_2_segment.append(s2_segment)
#             bert_seq_2_mask.append(s2_mask)
#
#         data = (
#             bert_seq_1,
#             bert_seq_1_segment,
#             bert_seq_1_mask,
#             bert_seq_2,
#             bert_seq_2_segment,
#             bert_seq_2_mask,
#             seq_weight_1,
#             seq_weight_2,
#             label
#         )
#
#         return data
#
#     def weighted_sentence_embedding(self):
#         data = self.prepare_data()
#         bert_seq_1 = data[0]
#         bert_seq_1_segment = data[1]
#         bert_seq_1_mask = data[2]
#         bert_seq_2 = data[3]
#         bert_seq_2_segment = data[4]
#         bert_seq_2_mask = data[5]
#         seq_weight_1 = data[6]
#         seq_weight_2 = data[7]
#         label = data[8]
#
#         total_s1_embedding = []
#         for s1, g1, m1, w1 in zip(bert_seq_1, bert_seq_1_segment, bert_seq_1_mask, seq_weight_1):
#
#             input_data = {
#                 'input_ids': torch.unsqueeze(torch.tensor(s1), 0).to(self.device),
#                 'token_type_ids': torch.unsqueeze(torch.tensor(g1), 0).to(self.device),
#                 'attention_mask': torch.unsqueeze(torch.tensor(m1), 0).to(self.device),
#                 'token_weight': torch.unsqueeze(torch.tensor(w1), 1).to(self.device)
#             }
#
#             output = self.model(input_ids=input_data['input_ids'],
#                                 attention_mask=input_data['attention_mask'],
#                                 token_type_ids=input_data['token_type_ids'])
#             hidden_states = output[2]
#             s1_embedding = extract_embedding(hidden_states, input_data['token_weight'])
#             total_s1_embedding.append(s1_embedding)
#
#         total_s2_embedding = []
#         for s2, g2, m2, w2 in zip(bert_seq_2, bert_seq_2_segment, bert_seq_2_mask, seq_weight_2):
#
#             input_data = {
#                 'input_ids': torch.unsqueeze(torch.tensor(s2), 0).to(self.device),
#                 'token_type_ids': torch.unsqueeze(torch.tensor(g2), 0).to(self.device),
#                 'attention_mask': torch.unsqueeze(torch.tensor(m2), 0).to(self.device),
#                 'token_weight': torch.unsqueeze(torch.tensor(w2), 1).to(self.device)
#             }
#
#             output = self.model(input_ids=input_data['input_ids'],
#                                 attention_mask=input_data['attention_mask'],
#                                 token_type_ids=input_data['token_type_ids'])
#             hidden_states = output[2]
#             s2_embedding = extract_embedding(hidden_states, input_data['token_weight'])
#             total_s2_embedding.append(s2_embedding)
#
#         s1_embedding = torch.stack(total_s1_embedding, dim=0).squeeze(dim=1)
#         s2_embedding = torch.stack(total_s2_embedding, dim=0).squeeze(dim=1)
#         total_embedding = torch.cat([s1_embedding, s2_embedding], dim=0)
#         return s1_embedding, s2_embedding, total_embedding, label
#
#     def run(self):
#         s1_emb, s2_emb, all_emb, label = self.weighted_sentence_embedding()
#
#         all_emb = all_emb.detach().cpu().numpy()
#         self.svd.fit(all_emb)
#         first_singular_vector = torch.tensor(self.svd.components_, device=self.device)
#         singular_matrix = first_singular_vector * torch.transpose(first_singular_vector)
#         s1_emb = s1_emb - singular_matrix * s1_emb
#         s2_emb = s2_emb - singular_matrix * s2_emb
#
#         data = list(zip(s1_emb, s2_emb, label))
#         print(data)
#
#         with open(self.output_path, 'wb') as f:
#             pickle.dump(data, f)
#         return
import pickle
from sklearn.decomposition import TruncatedSVD

import torch

from utils import weighted_embedding


class SIF:
    def __init__(self, input_path, output_path, vocab, token_weight, embeddings, data_type='train'):
        super(SIF, self).__init__()
        self.data_path = input_path
        self.output_path = output_path
        self.data_type = data_type
        self.vocab = vocab
        self.token_weight = token_weight
        self.embeddings = embeddings
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
        self.svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        return

    def load_data(self):
        seq_1 = []
        seq_2 = []
        label = []
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split('\t')
                s1 = data[0].split(' ')
                s2 = data[1].split(' ')
                seq_1.append(s1)
                seq_2.append(s2)
                if self.data_type == 'train':
                    label.append(data[2])
                else:
                    label.append(2)

        return seq_1, seq_2, label

    def prepare_data(self):
        seq_1, seq_2, label = self.load_data()

        bert_seq_1 = []
        bert_seq_2 = []
        seq_weight_1 = []
        seq_weight_2 = []

        for s1, s2 in zip(seq_1, seq_2):
            s1 = [self.vocab.get(token, self.vocab['[UNK]']) for token in s1]
            s2 = [self.vocab.get(token, self.vocab['[UNK]']) for token in s2]

            w1 = [self.token_weight.get(id) for id in s1]
            w2 = [self.token_weight.get(id) for id in s2]
            seq_weight_1.append(w1)
            seq_weight_2.append(w2)
            bert_seq_1.append(s1)
            bert_seq_2.append(s2)

        data = (
            bert_seq_1,
            bert_seq_2,
            seq_weight_1,
            seq_weight_2,
            label
        )

        return data

    def weighted_sentence_embedding(self):
        data = self.prepare_data()
        bert_seq_1 = data[0]
        bert_seq_2 = data[1]
        seq_weight_1 = data[2]
        seq_weight_2 = data[3]
        label = data[4]

        total_s1_embedding = []
        for s1, w1 in zip(bert_seq_1, seq_weight_1):
            index = torch.unsqueeze(torch.LongTensor(s1), 0).to(self.device)
            weight = torch.unsqueeze(torch.tensor(w1), 1).to(self.device)
            embedding = self.embeddings(index)
            s1_embedding = weighted_embedding(embedding, weight)
            total_s1_embedding.append(s1_embedding)
            break

        total_s2_embedding = []
        for s2, w2 in zip(bert_seq_2, seq_weight_2):
            index = torch.unsqueeze(torch.LongTensor(s2), 0).to(self.device)
            weight = torch.unsqueeze(torch.tensor(w2), 1).to(self.device)
            embedding = self.embeddings(index)
            s2_embedding = weighted_embedding(embedding, weight)
            total_s2_embedding.append(s2_embedding)
            break

        s1_embedding = torch.stack(total_s1_embedding, dim=0).squeeze(dim=1)
        s2_embedding = torch.stack(total_s2_embedding, dim=0).squeeze(dim=1)
        total_embedding = torch.cat([s1_embedding, s2_embedding], dim=0)
        return s1_embedding, s2_embedding, total_embedding, label

    def run(self):
        s1_emb, s2_emb, all_emb, label = self.weighted_sentence_embedding()
        all_emb = all_emb.detach().cpu().numpy()
        self.svd.fit(all_emb)
        first_singular_vector = torch.tensor(self.svd.components_, device=self.device)
        singular_matrix = first_singular_vector.T * first_singular_vector

        s1_emb = s1_emb.T - torch.matmul(singular_matrix, s1_emb.T)
        s2_emb = s2_emb.T - torch.matmul(singular_matrix, s2_emb.T)

        data = list(zip(s1_emb.detach().cpu().numpy(), s2_emb.detach().cpu().numpy(), label))
        with open(self.output_path, 'wb') as f:
            pickle.dump(data, f)
        return