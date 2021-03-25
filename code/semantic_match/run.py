import pandas as pd
from utils import  get_vocab_dict
from train import train

test_data = pd.read_csv("gaiic_track3_round1_testA_20210228.tsv",sep="\t",names=["seq1", "seq2"])
train_data = pd.read_csv("gaiic_track3_round1_train_20210228.tsv",sep="\t",names=["seq1", "seq2", "label"])

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
train(train_data, vocab)