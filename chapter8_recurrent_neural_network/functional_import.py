import collections
import re
import random
import torch
from d2l import torch as d2l

#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():
    """ 加载data machine这本书 """
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


def tokenize(lines, token_type='word'):
    if token_type == 'word':
        return [line.split() for line in lines]
    elif token_type == 'char':
        return [list(line) for line in lines]
    else:
        print(f'unknown type of token: {token_type}')

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

""" 随机采样 """
def seq_data_iter_rand(corpus, batch_size, num_steps):
    """ 在corpus中随机取出长度为num_steps的子序列
        :params
        batch_size: 每次取样的样本个数
        num_steps: 每个样本中的元素个数"""
    corpus = corpus[random.randint(0, num_steps-1):] # -1是为了将向后移动1的序列作为label
    num_subseqs = len(corpus) // num_steps
    initial_indices = list(range(0, num_subseqs*num_steps, num_steps)) # start, stop, steps
    random.shuffle(initial_indices)

    def data(pos):
        """ 返回从pos开始长为num_steps的子序列"""
        return corpus[pos:pos+num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, num_batches * batch_size, batch_size):
        initial_indices_per_batch = initial_indices[i:i+batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j+1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

""" 随机采样顺序分区 """
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """ 每一个相邻batch的第i个样本在原始corpus也是相邻的 """
    offset = random.randint(0, num_steps)   # 计算一个[0, num_steps]的偏移，舍弃掉0~offset
    num_token = (len(corpus)-offset-1) // batch_size * batch_size # //整除
    Xs = torch.tensor(corpus[offset:offset+num_token]) # X能用的所有token
    Ys = torch.tensor(corpus[offset+1:offset+1+num_token]) # Y能用的所有token
    Xs = Xs.reshape(batch_size, -1) # reshape成每个batch的大小batch_size，也就是每个batch的行数
    Ys = Ys.reshape(batch_size, -1)
    num_sequence_per_row = Xs.shape[1] // num_steps # 看每一行能取出多少个num_steps的子序列
    for i in range(0, num_steps*num_sequence_per_row, num_steps):
        X = Xs[:, i:i+num_steps]
        Y = Ys[:, i+1:i+1+num_steps]
        yield X,Y

""" 两种采样函数包装到类中 """
class SeqDataLoader:
    """ 加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_rand
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
