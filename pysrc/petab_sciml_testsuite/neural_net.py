"""Neural network YAML generation for hybrid tests"""

import os
import torch
from torch import nn
from torch.nn import functional as F
from petab_sciml.standard import Input, NNModel, NNModelStandard


class Net1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, 5)
        self.layer2 = nn.Linear(5, 5)
        self.layer3 = nn.Linear(5, 1)

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        x = self.layer1(net_input)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        x = self.layer3(x)
        return x


class Net2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, 5)
        self.layer2 = nn.Linear(5, 10)
        self.layer3 = nn.Linear(10, 1)

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        x = self.layer1(net_input)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x


class Net3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Conv2d(3, 1, (5, 5))
        self.layer2 = nn.Flatten()
        self.layer3 = nn.Linear(36, 1)

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        x = self.layer1(net_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(x)
        return x


class Net4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, 5)
        self.layer2 = nn.Linear(5, 5)
        self.layer3 = nn.Linear(5, 2)

    def forward(self, net_input: torch.Tensor) -> torch.Tensor:
        x = self.layer1(net_input)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        x = self.layer3(x)
        return x


class Net5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(2, 5)
        self.layer2 = nn.Linear(5, 5)
        self.layer3 = nn.Linear(5, 1)

    def forward(self, net_input1: torch.Tensor, net_input2: torch.Tensor) -> torch.Tensor:
        net_input = torch.cat((net_input1, net_input2))
        x = self.layer1(net_input)
        x = F.tanh(x)
        x = self.layer2(x)
        x = F.tanh(x)
        x = self.layer3(x)
        return x


class Net6(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Linear(4, 5)
        self.layer2 = nn.Linear(5, 10)
        self.layer3 = nn.Linear(10, 1)

    def forward(self, net_input1: torch.Tensor, net_input2: torch.Tensor) -> torch.Tensor:
        net_input = torch.cat((net_input1, net_input2))
        x = self.layer1(net_input)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        return x

def get_net_yaml(net_id):
    if net_id == "net1":
        net = Net1()
    elif net_id == "net2":
        net = Net2()
    elif net_id == "net3":
        net = Net3()
    elif net_id == "net4":
        net = Net4()
    elif net_id == "net5":
        net = Net5()
    else:
        net = Net6()

    if net_id == "net5":
        inputs = [Input(input_id="input0"), Input(input_id="input1")]
    else:
        inputs = [Input(input_id="input0")]

    net_model = NNModel.from_pytorch_module(
        module=net,
        nn_model_id=net_id,
        inputs=inputs
    )
    dir_save = os.path.join(os.getcwd(), 'assets', 'net_yaml')
    NNModelStandard.save_data(
        data=net_model, filename=os.path.join(dir_save, net_id + ".yaml"),
    )
    return None


get_net_yaml("net1")
get_net_yaml("net2")
get_net_yaml("net3")
get_net_yaml("net4")
get_net_yaml("net5")
get_net_yaml("net6")
