# Test Case 034

Test case for when a feed-forward neural network sets one of the model parameters, and
all the parameters in the neural network are assigned a `Normal(0.0, 1.0)` prior,
except `layer1.weight` which has a `Normal(0.0, 2.0)` prior.

## Model Structure

The SBML model for this problem is given as:

$$\frac{\mathrm{d} \text{prey}}{\mathrm{d} t} = \alpha \cdot \text{prey} - \beta \cdot \text{prey} \cdot \text{predator} $$

$$\frac{\mathrm{d} \text{predator}}{\mathrm{d} t} = \gamma \cdot \text{prey} \cdot \text{predator} - \delta \cdot \text{predator}$$

## Data-Driven Model Structure

`N` is a feed-forward neural network replacing the `gamma` parameter with the following specification:

| LayerID | Layer                                                  | Activation Function |
|---------|--------------------------------------------------------|---------------------|
| layer1  | Linear(in_features = 2, out_features = 5, bias = true) | tanh                |
| layer2  | Linear(in_features = 5, out_features = 5, bias = true) | tanh                |
| layer3  | Linear(in_features = 5, out_features = 1, bias = true) | identity            |

The inputs to the network are provided by the `input1` and `input2` parameters from the parameters table. All parameters in the NN-model have a `Normal(0.0, 1.0)` prior, except
for the weights in layer 1 which has a `Normal(0.0, 2.0)` prior.
