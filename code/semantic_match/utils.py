import json
import numpy as np
from collections import defaultdict
from collections import Counter


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
    with open(corpus_vocab_path, 'r') as f:
        for line in f.readlines():
            corpus_vocab.append(line.strip())

    frequency = [count.get(token, 0) for token in corpus_vocab]
    keep_index = list(np.argsort(frequency)[::-1])
    keep_index = list_special_index + keep_index[:used_size]
    return keep_index


def get_token_weight(train_path, test_path, vocab, special_tokens, param=1e-4):
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

    token_counter = Counter(total_tokens)
    for token in special_tokens:
        token_counter[token] = 0

    token_weight = {}
    for k, v in token_counter.items():
        weight = param / (param + v)
        if k in special_tokens:
            weight = 0
        token_weight[vocab.get(k, vocab['[UNK]'])] = weight
    return token_weight

