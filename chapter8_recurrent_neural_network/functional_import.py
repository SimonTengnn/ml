import collections
import re
import random
import torch
import math
from torch import nn
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

def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size # one hot以后就是vocab的长度(分类的类别个数)

    def normal(shape):
        return torch.rand(size=shape, device=device) * 0.01
        # return nn.init.xavier_uniform_(torch.empty(size=shape, device=device))
    W_xh = normal((num_inputs, num_hiddens)) # num_hiddens是一个时间步的隐藏单元个数，每个时间步都有num_hiddensz个隐藏单元
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_ho = normal((num_hiddens, num_outputs))
    b_o = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_ho, b_o]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    """ RNN在训练过程中会使用多个batch
        对每个batch都要保存(batch_size,num_hiddens)大小的隐藏状态
    """
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def rnn(inputs, state, params):
    """ 作为RNN的前向传播forward
        inputs 输入序列(时间步数, batch_size, vocab_size) 也就是每个时间步所有输入X的one-hot
        state 隐藏状态，上一个时间步传递的信息
        params 模型参数 包括W_xh W_hh b_h W_ho b_o"""
    W_xh, W_hh, b_h, W_ho, b_o = params # 解包
    H, = state # 解包 state是只有H为元素的tuple(H,)
    outputs = []
    for X in inputs: # 每个时间步上，X的形状是(batch_size, vocab_size)
        H = torch.tanh(torch.matmul(H, W_hh)+
                       torch.matmul(X, W_xh)+
                       b_h) # tanh是激活函数

        Y = torch.matmul(H, W_ho) + b_o
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,) # 输出的是 Y, state
                                            # Y形状为(batch_size, vocab_size)

class RNNModelScratch:
    """ 将上述方法包装起来 """
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # X.T是为了将(batch_size, num_steps)转成(num_steps, batch_size)
        # 按照时间步顺序喂入RNN
        # 最终X的形状是(num_steps, batch_size, vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)

def predict_ch8(prefix, num_preds, net, vocab, device):
    """ 在字符串prefix后生成新的字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))

    # 未超出原字符串
    for y in prefix[1:]: # prefix字符串中给出的token
        _, state = net(get_input(), state)  # 不记录生成的输出
        outputs.append(vocab[y]) # 直接放入prefix原本的token

    # 超出原字符串
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1))) # dim为去掉的维度
        # 从每一列取出最大值，那就去掉列的维度
    return ''.join([vocab.idx_to_token[i] for i in outputs])

""" 梯度裁剪 """
def grad_clipping(net, theta):
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = math.sqrt(sum(torch.sum(p.grad**2) for p in params))
    if norm < theta:    # theta/norm < 1
        for param in params:
            param.grad[:] *= theta/norm # grad和grad[:]都可以，是一样的

""" 训练 """
def train_epoch(net, train_iter, loss, updater, device, use_random_iter):
    """ 一个epoch的训练操作 """
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2) # 训练损失和， 词元总数
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 第一次迭代 or 随机采样(不是顺序分区)
            state = net.begin_state(X.shape[0], device=device)
        else:
            # 顺序分区 state会保留前一轮数据, 用detach保证反向传播不依赖之前的梯度
            if isinstance(net, nn.Module) and not isinstance(state, tuple): # 第一次迭代
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel()) # .numel为tensor元素总数
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_epochs(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))