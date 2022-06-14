from torch import nn


class JINetModel(nn.Module):
    def __init__(self):
        super(JINetModel, self).__init__()

    def forward(self, x):
        return x


def n2t_jinet_train_loop():
    raise NotImplementedError
