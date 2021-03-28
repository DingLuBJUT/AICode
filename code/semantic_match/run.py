import pandas as pd
from utils import  get_vocab_dict
from utils import get_keep_index


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

corpus_vocab_path = "vocab.txt"
count_path = "counts.json"
list_special_index = [0, 100, 101, 102, 103, 6, 7, 8]
keep_index = get_keep_index(corpus_vocab_path,count_path,list_special_index,len(vocab))

# train(train_data, vocab, keep_index)
# k_fold_train(train_data, vocab, keep_index)
