from torch import nn


class CustomRot90(nn.Module):
    def __init__(self, spacetime_ndim):
        self.spacetime_ndim = spacetime_ndim
        self.rot90_1 = None  # TODO: assign rot90 here
        self.rot90_2 = None  # TODO: assign rot90 here
        self.rot90_3 = None  # TODO: assign rot90 here
        self.rot90_5 = None  # TODO: assign rot90 here
        self.rot90_6 = None  # TODO: assign rot90 here

    def forward(self, x):
        if self.spacetime_ndim == 2:
            x1 = self.rot90_1(x)
            x2 = self.rot90_2(x)
            x3 = self.rot90_3(x)
            x = torch.cat((x1, x2, x3), 0)
        elif self.spacetime_ndim == 3:
            x1 = self.rot90_1(x)
            x2 = self.rot90_2(x)
            x3 = self.rot90_3(x)
            x5 = self.rot90_5(x)
            x6 = self.rot90_6(x)
            x = torch.cat((x1, x2, x3, x5, x6), 0)
        else:
            raise ValueError("CustomRot90 only supports 2D and 3D operations.")

        return x
