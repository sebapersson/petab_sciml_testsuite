"""Neural network import test generation"""

import os
import torch
from torch import nn
from test_cases.net_import.helper import make_yaml, test_nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, 5)
        self.layer2 = nn.Linear(2, 10)
        self.layer3 = nn.Bilinear(5, 10, 2)

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        x1 = self.layer1(net_input)
        x2 = self.layer2(net_input)
        out = self.layer3(x1, x2)
        return out


# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', 'net_import', "002")
net = Net()
make_yaml(net, dir_save)
test_nn(net, dir_save, ["layer1", "layer2", "layer3"])
