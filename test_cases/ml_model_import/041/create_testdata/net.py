"""Neural network import test generation"""

import os
import torch
from torch import nn
from torch.nn import functional as F
from test_cases.net_import.helper import make_yaml, test_nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, 5)

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        out = F.log_softmax(net_input, dim=0)
        return out

# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', 'net_import', "041")
net = Net()
make_yaml(net, dir_save)
test_nn(net, dir_save, None)
