"""
    文本预处理
"""

import collections
import re
from d2l import torch as d2l

#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """ 加载data machine这本书 """
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
# print(f'文本行数: {len(lines)}')
# print(f'lines[0]: {lines[0]}')
# print(f'lines[100]: {lines[100]}')

def tokenize(lines, token_type='word'):
    if token_type == 'word':
        return [line.split() for line in lines]
    elif token_type == 'char':
        return [list(line) for line in lines]
    else:
        print(f'unknown type of token: {token_type}')

tokens = tokenize(lines, token_type='word')

class Vocab:
    """ 文本词汇表 """
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        # 将token按频率从大到小排序
        self._token_freqs = sorted(counter.items(), key=lambda x:x[1], reverse=True)
            # Counter内部是[(word1, time1), (word2, freq2)...] lambda x:x[1]相当于每次都取出time对比

        self.idx_to_token, self.token_to_idx = [], dict()
        # unknown token频率为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token) # ['<unk>', reserved_tokens, 'token1', 'token2'...]
                self.token_to_idx[token] = len(self.token_to_idx) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """ 给出一系列token字符串，返回对应index """
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens)
        return [self.__getitem__(token) for token in tokens] # 递归直到token是单个str

    def to_tokens(self, indices):
        """ 给出一系列index， 返回对应token字符串 """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.to_tokens(index) for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """ 统计token频率 """
    if isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)  # Counter只接受一维列表

vocab = Vocab(tokens)
# print(vocab.idx_to_token[:10])
# print(list(vocab.token_to_idx.items())[:10])
for i in range(10):
    print(f'text: {tokens[i]}')
    print(f'index: {vocab[tokens[i]]}') # 实际上就是__getitem__


""" 打包到一个函数 """
def load_corpus_time_machine(max_tokens=-1):
    """ 返回time machine文本集的token索引列表和vocab词汇表 """
    lines = read_time_machine()
    tokens = tokenize(lines)
    vocab = Vocab(tokens)
    """ 展平到一个list中 """
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
print(f'len of corpus: {len(corpus)}')
print(f'len of vocab: {len(vocab)}') # ·Vocab.__len__ 返回的是len(self.idx_to_token)