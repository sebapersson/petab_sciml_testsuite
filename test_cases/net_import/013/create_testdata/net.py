import torch
import torch.nn as nn
import os

from src.python.helper import make_yaml, test_nn

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten1 = nn.Flatten(start_dim=0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.flatten1(input)
        return x

# Create a pytorch module, convert it to PEtab SciML, then save it to disk.
dir_save = os.path.join(os.getcwd(), 'test_cases', 'net_import', "013")
net = Net()
make_yaml(net, dir_save)
test_nn(net, dir_save, None)
