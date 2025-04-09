import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import functional_import

batch_size, num_steps = 32, 35
train_iter, vocab = functional_import.load_data_time_machine(batch_size, num_steps)

""" 使用pytorchAPI定义模型"""
num_hiddens = 256
rnn_layer = nn.RNN(input_size=len(vocab),  hidden_size=num_hiddens) # rnn_layer不包括输出层
print(f'rnn model: {rnn_layer}')

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__( **kwargs)
        self.vocab_size = vocab_size
        self.rnn = rnn_layer
        self.num_hiddens = self.rnn.hidden_size

        # 如果RNN是双向，num_directional为2 否则为1
        if self.rnn.bidirectional:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size) # linear将隐藏状态转换为token概率
        else:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state) # Y是所有时间步的隐藏状态集合(num_steps, batch_size, num_hiddens)， state是当前最后一个时间步的隐藏状态
        output = self.linear(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, device, batch_size=1):
        def begin_state(self, device, batch_size=1):
            if not isinstance(self.rnn, nn.LSTM):
                # nn.GRU以张量作为隐状态
                return torch.zeros((self.num_directions * self.rnn.num_layers,
                                    batch_size, self.num_hiddens),
                                   device=device)
            else:
                # nn.LSTM以元组作为隐状态
                return (torch.zeros((
                    self.num_directions * self.rnn.num_layers,
                    batch_size, self.num_hiddens), device=device),
                        torch.zeros((
                            self.num_directions * self.rnn.num_layers,
                            batch_size, self.num_hiddens), device=device))

""" 初始化隐藏状态 """
state = torch.zeros((1, batch_size, num_hiddens))
print(f'state.shape: {state.shape}')

inputs = torch.rand(size=(num_steps, batch_size, len(vocab)))
Hs, state_new = rnn_layer(inputs, state) # rnn_layer不包括输出层，实际上它的输出就是H

device = d2l.try_all_gpus()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
num_epochs, lr = 500, 0.01
functional_import.train_epochs(net, train_iter, vocab, lr, num_epochs, device)