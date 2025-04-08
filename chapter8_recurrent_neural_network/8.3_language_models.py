""" N元语法(也就是马尔可夫假设下tau=N-1)
    顺序分区"""

import torch
import random
import re
import functional_import
from matplotlib import pyplot as plt
from d2l import torch as d2l

#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')
def read_time_machine():
    """ 加载data machine这本书 """
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

# 一元语法 每次只用单个词(tau=0)
tokens = d2l.tokenize(read_time_machine()) # tokens是二维列表，按行存每行的token
corpus = [token for lines in tokens for token in lines] # corpus是一位列表，将tokens展平
vocab = d2l.Vocab(corpus)
# print(f'token freq top10:\n{vocab.token_freqs[:10]}')

# freqs较大的为停用词 stop words
# freqs = [freq for token, freq in vocab.token_freqs]
# plt.plot(freqs)
# plt.xlabel('token')
# plt.ylabel('time')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# 二元语法 每次取两个词(tau=1)
# bi_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])] # zip从不同列表取出元素组成配对(token1, token2)
# bi_vocab = d2l.Vocab(bi_tokens)
# print(f'token freq top10:\n{bi_vocab.token_freqs[:10]}')

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

# 生成从0到34的序列
# my_seq = list(range(35))
# for X,Y in seq_data_iter_rand(my_seq, batch_size=3, num_steps=7):
#     print(f'X: {X}\n'
#           f'Y: {Y}\n')
""" 随机采样顺序分区 """
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """ 每一个相邻batch的第i个样本在原始corpus也是相邻的 """
    offset = random.randint(0, num_steps)   # 计算一个[0, num_steps]的偏移，舍弃掉0~offset
    num_token = (len(corpus)-offset-1) // batch_size * batch_size # //整除
    Xs = torch.tensor(corpus[offset:offset+num_token]) # X能用的所有token
    Ys = torch.tensor(corpus[offset+1:offset+1+num_token]) # Y能用的所有token
    Xs.reshape(batch_size, -1) # reshape成每个batch的大小batch_size，也就是每个batch的行数
    Ys.reshape(batch_size, -1)
    num_sequence_per_row = Xs.shape[1] // num_steps # 看每一行能取出多少个num_steps的子序列
    for i in range(0, num_steps*num_sequence_per_row, num_steps):
        X = Xs[:, i:i+num_steps]
        Y = Ys[:, i+1:i+1+num_steps]
        yield X,Y

#
# my_seq = list(range(35))
# for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
#     print('X: ', X, '\nY:', Y)


""" 两种采样函数包装到类中 """
class SeqDataLoader:
    """ 加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_rand
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = functional_import.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)