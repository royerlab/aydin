import torch
from torch import nn


class SplitAndRot90(nn.Module):
    def __init__(self, spacetime_ndim):
        self.spacetime_ndim = spacetime_ndim
        self.split0 = None  # TODO: assign here
        self.split1 = None  # TODO: assign here
        self.split2 = None  # TODO: assign here
        self.split3 = None  # TODO: assign here
        self.split4 = None  # TODO: assign here
        self.split5 = None  # TODO: assign here

        self.rot90_1 = None  # TODO: assign rot90 here
        self.rot90_2 = None  # TODO: assign rot90 here
        self.rot90_3 = None  # TODO: assign rot90 here
        self.rot90_5 = None  # TODO: assign rot90 here
        self.rot90_6 = None  # TODO: assign rot90 here

    def forward(self, x):
        if self.spacetime_ndim == 2:
            x0 = self.split0(x)
            x1 = self.split1(x)
            x2 = self.split2(x)
            x3 = self.split3(x)

            x1 = self.rot90_1(x1)
            x2 = self.rot90_2(x2)
            x3 = self.rot90_3(x3)
            x = torch.cat((x0, x1, x2, x3), 0)
        elif self.spacetime_ndim == 3:
            x0 = self.split0(x)
            x1 = self.split1(x)
            x2 = self.split2(x)
            x3 = self.split3(x)
            x5 = self.split4(x)
            x6 = self.split5(x)

            x1 = self.rot90_1(x1)
            x2 = self.rot90_2(x2)
            x3 = self.rot90_3(x3)
            x5 = self.rot90_5(x5)
            x6 = self.rot90_6(x6)
            x = torch.cat((x0, x1, x2, x3, x5, x6), 0)
        else:
            raise ValueError("CustomRot90 only supports 2D and 3D operations.")

        return x
