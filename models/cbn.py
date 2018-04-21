import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torch.nn.modules.batchnorm import _BatchNorm as BatchNorm

# def CBN(nn.Module):
    # def __init__(self, input_channels, condition_size, gamma = 1.0):
        # self.initial_gamma = gamma
        # # self.beta_predictor = nn.Sequential(nn.Linear(condition_size, input_channels))
        # self.beta_predictor = nn.Linear(condition_size, input_channels)
        # self.gamma_predictor = nn.Linear(condition_size, input_channels)

        # # self. = torch.nn.Linear(condition_size, input_channels)
        # # self.FC2 = torch.nn.Linear(input_channels, input_channels)

    # def forward(input, condition_signal):
        # beta = self.beta_predictor(condition_signal)
        # gamma = self.initial_gamma + self.gamma_predictor(condition_signal)

        # return F.batch_norm(input, self.

class CBN(BatchNorm):
    def __init__(self, num_features, condition_size, eps=1e-5, momentum=0.1, affine=True):
        print(super(CBN, self))
        super(CBN, self).__init__(num_features, eps, momentum, False)
        assert affine and not self.affine
        self.condition_size = condition_size

        self.beta_predictor = nn.Linear(condition_size, num_features)
        self.gamma_predictor = nn.Linear(condition_size, num_features)

    def forward(self, input, condition_signal):
        bias = self.beta_predictor(condition_signal)
        weight = self.gamma_predictor(condition_signal)

        # return super(CBN, self).forward(input)
        return F.batch_norm(input, self.running_mean, self.running_var,
                        weight, bias,
                        self.training, self.momentum, self.eps)


if __name__ == '__main__':
    print("Hello world")
    A = Var(torch.randn((1, 3, 3, 3)))
    B = A.clone()
    bn = BatchNorm(3, affine=False)
    x_1 = bn(A)
    print(x_1[0][0])

    c = CBN(3, condition_size = 1)
    x_2 = c(B, Var(torch.FloatTensor([1])))
    print(x_2[0][0])
