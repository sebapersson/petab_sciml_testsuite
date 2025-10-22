# Test Case 030

Test case for when a feed-forward neural network sets one of the model parameters, with the input to the network provided by
two parameters from the parameters table. Since the network is not a part of the ODE model's right-hand side (RHS), it should
only be evaluated once per likelihood computation for computational efficiency.

## Model Structure

Lotka-Volterra model with a neural network for one of the interaction terms:

$$\frac{\mathrm{d} \text{prey}}{\mathrm{d} t} = \alpha \cdot \text{prey} - \beta \cdot \text{prey} \cdot \text{predator} $$

$$\frac{\mathrm{d} \text{predator}}{\mathrm{d} t} = N[1] - \delta \cdot \text{predator}$$

The SBML model for this problem is given as:

$$\frac{\mathrm{d} \text{prey}}{\mathrm{d} t} = \alpha \cdot \text{prey} - \beta \cdot \text{prey} \cdot \text{predator} $$

$$\frac{\mathrm{d} \text{predator}}{\mathrm{d} t} = \gamma - \delta \cdot \text{predator}$$

Hence, the neural network replaces $\gamma$ during the import.

## Data-Driven Model Structure

`N` is a feed-forward neural network with the following specification:

| LayerID | Layer                                                  | Activation Function |
|---------|--------------------------------------------------------|---------------------|
| layer1  | Linear(in_features = 2, out_features = 5, bias = true) | tanh                |
| layer2  | Linear(in_features = 5, out_features = 5, bias = true) | tanh                |
| layer3  | Linear(in_features = 5, out_features = 1, bias = true) | identity            |

The inputs to the network are provided by the `input1` and `input2` parameters from the parameters table, which are each provided
as separate input arguments.
