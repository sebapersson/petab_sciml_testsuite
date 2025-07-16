"""Neural network import test generation"""

import os
import torch
from torch import nn
from test_cases.net_import.helper import make_yaml, test_nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.AdaptiveMaxPool3d((3, 2, 1))

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        x = self.layer1(net_input)
        return x

# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', 'net_import', "016")
net = Net()
make_yaml(net, dir_save)
test_nn(net, dir_save, None)
