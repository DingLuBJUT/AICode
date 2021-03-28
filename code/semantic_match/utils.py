import json
import numpy as np
from collections import defaultdict

def get_vocab_dict(data, list_special_tokens=None, min_count=5):
    vocab = defaultdict(int)
    for seq in data:
        for w in seq.split(' '):
            if w in vocab.keys():
                vocab[w] += 1
            else:
                vocab[w] = 0
    vocab = {token: count for token, count in vocab.items() if count > min_count}
    vocab = dict(sorted(vocab.items(), key=lambda item: -item[1]))
    vocab = list_special_tokens + list(vocab.keys())
    vocab = dict(zip(vocab, range(len(vocab))))
    return vocab


def get_keep_index(corpus_vocab_path, count_path, list_special_index, used_size):
    count = json.load(open(count_path))
    del count['[CLS]']
    del count['[SEP]']

    corpus_vocab = []
    with open(corpus_vocab_path,'r') as f:
        for line in f.readlines():
            corpus_vocab.append(line.strip())

    frequency = [count.get(token, 0) for token in corpus_vocab]
    keep_index = list(np.argsort(frequency)[::-1])
    keep_index = list_special_index + keep_index[:used_size]
    return keep_index
