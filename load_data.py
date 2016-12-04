import json
from collections import Counter
import numpy as np
import itertools
from functools import partial

LIMIT = 10 ** 2
step  = 5


def flatmap(func, *iterable):
    return itertools.chain.from_iterable(map(func, *iterable))

def _process(doc, seq_len, step, start_char, end_char, unknown_char, char_to_int):
    doc = start_char + doc.lower() + end_char
    return [
        [to_int_func(z, unknown_char=unknown_char, char_to_int=char_to_int) for z in doc[i:i + seq_len + 1]]
        for i in range(0, len(doc) - seq_len, step)
    ]

def to_int_func(char, unknown_char, char_to_int):
    # checking if it's a good or bad char
    if char not in char_to_int:
        char = unknown_char
    return char_to_int[char]

def get_data(seq_len):
    with open('text.json') as fp:
      all_listings = json.load(fp)['text']

    joined_listing = "".join(all_listings)
    counter = Counter(joined_listing.lower().replace("\"", "'").replace("’", "'"))
    chars = set(joined_listing)

    bad_chars = [c for c, v in counter.most_common() if v < 2000] + ['—', '•']
    good_chars = list(set(counter) - set(bad_chars))

    start_char = '\x02'
    end_char = '\x03'
    unknown_char = '\x04'

    # we don't want to pick characters that are already used
    assert start_char not in good_chars
    assert end_char not in good_chars
    assert unknown_char not in good_chars

    good_chars.extend([start_char, end_char, unknown_char])

    char_to_int = {ch: i for i, ch in enumerate(good_chars)}
    int_to_char = {i: ch for i, ch in enumerate(good_chars)}

    print('tranforming data')
    process = partial(_process, seq_len=seq_len, step=step, start_char=start_char, end_char=end_char, unknown_char=unknown_char, char_to_int=char_to_int)
    transofmed = np.array(list(flatmap(process, all_listings[:LIMIT])))
    x_ = transofmed[:, :seq_len]
    y_ = transofmed[:, seq_len]

    X = np.zeros((len(x_), seq_len, len(good_chars)), dtype=np.bool)
    Y = np.zeros((len(y_), len(good_chars)), dtype=np.bool)
    print("preparing indexes")
    for time, sentence in enumerate(x_):
        for index, char_index in enumerate(sentence):
            X[time, index, char_index] = 1
        Y[time, y_[time]] = 1

    return X, Y





