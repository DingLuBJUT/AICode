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
