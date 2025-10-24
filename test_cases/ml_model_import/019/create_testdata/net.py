"""Neural network import test generation"""

import os
import torch
from torch import nn
from test_cases.net_import.helper import make_yaml, test_nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(10, 2)
        self.drop = nn.Dropout(0.5)

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        x = self.drop(net_input)
        out = self.layer1(x)
        return out

# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', 'net_import', "019")
net = Net()
make_yaml(net, dir_save)
test_nn(net, dir_save, ["layer1"], dropout=True, atol=1e-2)
