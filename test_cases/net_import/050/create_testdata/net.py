"""Neural network import test generation"""

import os
import torch
from torch import nn
from test_cases.net_import.helper import make_yaml, test_nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm((4, 10, 11, 12))
        self.layer1 = nn.Conv3d(4, 1, 5)
        self.flatten1 = nn.Flatten()

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        x = self.norm1(net_input)
        x = self.layer1(x)
        x = self.flatten1(x)
        return x

# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', 'net_import', "050")
net = Net()
make_yaml(net, dir_save)
test_nn(net, dir_save, ["layer1", "norm1"])
