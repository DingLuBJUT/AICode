from torch.nn import Module, Linear, LayerNorm, Embedding, Parameter
from transformers import BertModel

class PretrainedBERT(Module):
    def __init__(self, embedding_size, embedding_dim, max_len, keep_index=None):
        super(PretrainedBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        if keep_index is not None:
            self.embedding = Embedding(embedding_size,embedding_dim)
            weight = self.bert.embeddings.word_embeddings.weight[keep_index]
            self.embedding.weight = Parameter(weight)
            self.bert.embeddings.word_embeddings = self.embedding

        self.lr1 = Linear(in_features=embedding_dim, out_features=embedding_dim)
        self.ln1 = LayerNorm([max_len, embedding_dim])
        self.lr2 = Linear(in_features=embedding_dim, out_features=embedding_size)
        return

    def forward(self, x):
        x = self.bert(input_ids=x['input_ids'],
                      attention_mask=x['attention_mask'],
                      token_type_ids=x['token_type_ids'])
        x = self.lr1(x['last_hidden_state'])
        x = self.ln1(x)
        x = self.lr2(x)
        return x
