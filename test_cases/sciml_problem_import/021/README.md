# Test Case 021

Test case for when one neural network with two outputs that appear before the ODE model, for which it sets the alpha and gamma parameters.

## Model Structure

Lotka-Volterra model with a neural network setting the first growth term, and the second interaction term:

$$\frac{\mathrm{d} \text{prey}}{\mathrm{d} t} = \alpha - \beta \cdot \text{prey} \cdot \text{predator} $$

$$\frac{\mathrm{d} \text{predator}}{\mathrm{d} t} = \gamma - \delta \cdot \text{predator}$$

## Data-Driven Model Structure

`net4` is a feed-forward neural network with the following specification:

| LayerID | Layer                                                  | Activation Function |
|---------|--------------------------------------------------------|---------------------|
| layer1  | Linear(in_features = 2, out_features = 5, bias = true) | tanh                |
| layer2  | Linear(in_features = 5, out_features = 5, bias = true) | tanh                |
| layer3  | Linear(in_features = 5, out_features = 2, bias = true) | identity            |

The inputs to the network are provided by the `prey` and `predator` species from the ODE-solution, and its outputs sets the alpha and gamma parameters.
