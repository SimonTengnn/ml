"""
    利用马尔可夫假设 将时序序列变成MLP多层感知机做自回归
"""
import torch
import matplotlib.pyplot as plt
from torch import nn
from d2l import torch as d2l

T = 1000
time = torch.arange(1, T+1, dtype=torch.float32)
x = torch.sin(0.01 * time)+ torch.normal(0, 0.2, (T,))
# 单独画出x
# plt.plot(time, x, label='sin(0.01t)+noise')
# plt.xlabel('Time')
# plt.ylabel('value')
# plt.xlim(1,1000)
# plt.show()

tau = 4
features = torch.zeros((T-tau, tau))
for i in range(tau):
    features[:, i] = x[i:T-tau+i] # T-tau滑动窗口
labels = x[tau:].reshape((-1, 1))

# 只取前n_train个用于训练
batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]), batch_size, is_train=True)

""" 初始化网络权重 """
def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)
#       nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
#       kaiming_uniform更适用于ReLu(方差更稳定，收敛更快)
""" 有两个全连接层的多层感知机 """
def get_net():
    net = nn.Sequential(nn.Linear(4, 10), # features每一行个元素
                        nn.ReLU(),
                        nn.Linear(10,1))
    net.apply(init_weights) # 遍历net的每一层，初始化weight
    return net

loss = nn.MSELoss()

def train(net, train_iter, lr, epochs, loss):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, Y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), Y)
            l.backward()
            trainer.step()
        print(f'epoch: {epoch+1}\t'
              f'loss: {d2l.evaluate_loss(net, train_iter, loss)}')
        # 评估出的loss（平均loss）=所有loss总和/loss的元素个数

lr, epochs = 0.01, 50
net = get_net()
train(net, train_iter, lr, epochs, loss)

""" 单步预测 """
onestep_preds = net(features)
# # d2l.plot([time, time[tau:]],
# #          [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
# #          'x', legend=['data', '1-step preds'], xlim=[1, 1000],
# #          figsize=(6, 3))
# plt.plot(time, x.detach().numpy(), label='data')
# plt.plot(time[tau:], onestep_preds.detach().numpy(), label='1-step preds')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.xlim(1, 1000)
# plt.legend()
# plt.show()

""" 多步预测（用前n_train个已知数训练，对后面的进行预测，误差会叠加导致效果越来越差） """
multistep_preds = torch.zeros(T)
multistep_preds[:n_train + tau] = x[:n_train + tau]
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i-tau:i].reshape((1, -1)) # nn.Linear要求输入是(batch_size, num_features)
        # reshape后就是(1, tau)
    )
# d2l.plot([time, time[tau:], time[n_train + tau:]],
#          [x.detach().numpy(), onestep_preds.detach().numpy(),
#           multistep_preds[n_train + tau:].detach().numpy()], 'time',
#          'x', legend=['data', '1-step preds', 'multistep preds'],
#          xlim=[1, 1000], figsize=(6, 3))
# plt.show()

""" k步预测 """
