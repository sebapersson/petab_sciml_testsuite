# Test Case 031

Test case for when a feed-forward neural network appears in one of the observable formulas.

## Model Structure

The SBML model for this problem is given as:

$$\frac{\mathrm{d} \text{prey}}{\mathrm{d} t} = \alpha \cdot \text{prey} - \beta \cdot \text{prey} \cdot \text{predator} $$

$$\frac{\mathrm{d} \text{predator}}{\mathrm{d} t} = \gamma \cdot \text{prey} \cdot \text{predator} - \delta \cdot \text{predator}$$

And the observable formula for `prey` is given by `N[1]` (the output from a neural network).

## Data-Driven Model Structure

`N` is a feed-forward neural network with the following specification:

| LayerID | Layer                                                  | Activation Function |
|---------|--------------------------------------------------------|---------------------|
| layer1  | Linear(in_features = 2, out_features = 5, bias = true) | tanh                |
| layer2  | Linear(in_features = 5, out_features = 5, bias = true) | tanh                |
| layer3  | Linear(in_features = 5, out_features = 1, bias = true) | identity            |

The inputs to the network are given by the `prey` and `predator` species., which are each provided as separate input arguments.
