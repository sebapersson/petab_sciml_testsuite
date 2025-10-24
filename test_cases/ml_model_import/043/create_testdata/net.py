"""Neural network import test generation"""

import os
import torch
from torch import nn
from torch.nn import functional as F
from test_cases.net_import.helper import make_yaml, test_nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm1d(50)
        self.layer1 = nn.Linear(5, 50)
        self.layer2 = nn.Linear(50, 2)

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        x = self.layer1(net_input)
        x = self.norm1(x)
        x = self.layer2(x)
        return x

# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', 'net_import', "043")
net = Net()
make_yaml(net, dir_save)
test_nn(net, dir_save, ["layer1", "layer2", "norm1"])
