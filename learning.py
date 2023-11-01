import numpy
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def learn_variable():
    x = Variable(torch.randn(2, 2), requires_grad=True)
    print(x)
    y = torch.tanh(x)
    print(y)

    y.grad_fn
    print(y.grad_fn)
    # 反向传播
    y.backward()
    x.grad
    print(x.grad)


def learn_tensor():
    randoms = torch.randint(0, 17, (120, 1))
    print(randoms)
    one_hots = F.one_hot(randoms)
    print(one_hots, one_hots.shape)
    one_hots = one_hots.squeeze(1)
    print(one_hots, one_hots.shape)


if __name__ == '__main__':
    learn_tensor()
