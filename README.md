# PEtab SciML test suite

The PEtab SciML test suite is a collection of test problems for the
[PEtab SciML](https://github.com/PEtab-dev/petab_sciml) data format for scientific machine
learning (SciML). Its purpose is to verify and quantify the capabilities of tools that
support the PEtab SciML format.

## Downloading and installing the test suite

The PEtab SciML test suite can be downloaded from GitHub via:

```bash
git clone https://github.com/sebapersson/petab_sciml_testsuite.git
```

All required test files are pre-built in the repository.

### Installing the Julia utility package

The test suite includes a [Julia](https://julialang.org/) package for generating new test
cases. To install it, with Julia ≥1.10, from the root directory of repository start Julia
and run:

```julia
import Pkg; Pkg.instantiate()
```

## Using the test suite

The PEtab SciML test cases are divided into three categories in the `test_cases` directory:

- Machine learning import. Tests import of models in the PEtab SciML independent YAML format
  for machine learning models.
- SciML problem import. Tests import of SciML problems combining machine learning models and
  the mechanistic model in multiple ways.
- Initialization tests. Tests imported SciML problems are initialized correctly.

Note, arrays test values (e.g., ML model inputs, outputs, gradient) are provided in the
[HDF5](https://www.hdfgroup.org/solutions/hdf5/) format, in which arrays are stored in
row-major format following PyTorch indexing. Therefore, in addition to accounting for the
indexing, importers in column-major languages (e.g., Julia, R) need to account for indexing.

### Machine learning model import tests

These tests verify that different ML architectures (e.g., layers, activation functions)
defined in the PEtab SciML YAML format (layers, activations) are imported correctly.

For each test case, three random input tensors, parameter sets, and expected outputs are
provided as `net_input_i.hdf5`, `net_ps_i.hdf5`, and `net_output_i.hdf5` for i = 1..3. The
file `solutions.yaml` lists the input/parameter/output combinations to verify. For tests
that include dropout, `solutions.yaml` provides a `dropout` field indicating how many
forward passes to run for computing the mean that should be compared against the reference.

Currently, supported and tested layers and activation functions are:

- **Layers**: `Conv1-3d`, `ConvTranspose1-3d`, `AvgPool1-3d`, `MaxPool1-3d`, `LPPool1-3d`,
  `AdaptiveMaxPool1-3d`, `AdaptiveMeanPool1-3d`, `BatchNorm1-3d`, `InstanceNorm1-3d`,
  `LayerNorm`, `Dropout1-3d`, `Dropout`, `AlphaDropout`, `Linear`, `Bilinear` and `Flatten`.
- **Activation**: `relu`, `relu6`, `hardtanh`, `hardswish`, `selu`, `leaky_relu`, `gelu`,
  `logsigmoid`, `tanhshrink`, `softsign`, `softplus`, `tanh`, `sigmoid`, `hardsigmoid`,
  `mish`, `elu`, `celu`, `softmax` and `log_softmax`.

A list of supported layers and activation functions for each tool supporting PEtab SciML can
be found in the PEtab SciML
[documentation](https://petab-sciml.readthedocs.io/latest/layers.html).

### SciML problem import tests

These tests verify that problems combining mechanistic and ML modules are imported
correctly. For each test case, reference values are provided for:

- the objective function (likelihood/loss),
- the gradient of the objective
- simulated outputs at the measurement points defined in the PEtab measurement table.

All evaluations should be performed at the PEtab nominal parameter values. The
`solutions.yaml` file lists the reference files and the absolute tolerances to use.
Gradients should be checked elementwise using the given tolerance. Gradient references files
are split by parameter type: mechanistic parameters are provided in TSV, and ML parameters
in HDF5 (following the PEtab SciML parameter format). Simulation values are provided in the
corresponding order of the measurements table.

These tests can also be adapted to run **without the PEtab SciML YAML format**. The problems
here define the ML model via the PEtab SciML YAML, but importers may accept framework-native
models (e.g., Lux.jl in Julia). To test such cases, replace the ML YAML asset with one
containing the ML module in the target format. Note that the ML model parameter HDF5 files
still follow the PEtab SciML ordering; if the target framework uses a different indexing
order, adjust the parameters accordingly.

### Initialization tests

These tests verify that nominal ML parameters are imported correctly (e.g., when a subset of
layers in a ML model are frozen). For each test case, `solutions.yaml` specifies the
reference ML parameter values (in the PEtab SciML neural-network parameter format) that the
ML models in the imported PEtab SciML problem should be initialized to.

### Indexing note (non-Python tools)

Following the PEtab SciML standard, all arrays/tensors in the test HDF5 use PyTorch’s
indexing and memory layout (row-major). To aid importers in other ecosystems (e.g., Julia),
the machine-learning import tests include per-ecosystem axis orders in `solutions.yaml`
(e.g., `input_order_py` `output_order_py`, `input_order_jl`/`output_order_jl`). These
indicate how tensors should be permuted in respective tools for performance (efficient
memory access). For example, for images PyTorch expects `(C, H, W)`, whereas Julia uses
`(H, W, C)`; accordingly, test case 018 sets `input_order_py = (C, H, W)` and
`input_order_jl = (H, W, C)`. If you add an importer for another language (e.g., R), provide
the corresponding input/output orders in `solutions.yaml` for the ML import tests.

## Contributing

Contributions to the PEtab SciML test suite are welcome. Please either:

- (Preferred) open a
  [pull request](https://github.com/sebapersson/petab_sciml_testsuite/pulls) (drafts
  welcome) adding one or more test cases.
- Or open an [issue](https://github.com/sebapersson/petab_sciml_testsuite/issues) describing
  the missing coverage.

### Adding a new test case

To add a new test case, create a subdirectory in the relevant suite under `test_cases/`,
named with the next three-digit ID (e.g., `003`). Include a short README describing what the
test covers. Most test files should be generated automatically using utility functions from
this repository’s associated Julia library. We use Julia for its strong SciML ecosystem and
mature tooling for computing reference gradients (high-order finite-difference methods). To
install this Julia library, with Julia ≥1.10, from the root directory of repository start
Julia and run:

```julia
import Pkg; Pkg.instantiate()
```

Which files to add depends on the test type.

For **ML model import tests**, provide both `net.jl` and `net.py` script. The `net.jl`
script should generate the test inputs, ML parameters, expected outputs, and
`solutions.yaml`. The `net.py` script should create the PEtab SciML ML-model YAML file and
checks consistency with the ML model used to produce the Julia reference outputs. This
ensures the case is importable in both Python (e.g., PyTorch/Equinox) and Julia, aligning
with the goal that the PEtab SciML YAML is exchangeable across ecosystems. An example can be
found
[here](https://github.com/sebapersson/petab_sciml_testsuite/tree/main/test_cases/net_import/001).

For **PEtab SciML import problem tests**, provide a `create.jl` script. It should generate
the PEtab problem files and the reference values. For the PEtab problem files, the mapping
and hybridisation PEtab tables must be provided manually, the other components can be
selected from predefined assets (`assets/` for SBML models, ML-model YAMLs, ML parameters,
and array inputs) and defaults in `src/` (new values can be added if needed). For the
reference values, include an `llh_id` that maps to a likelihood implementation in
`src/test_values/nllh/`. Because these SciML problems are not analytically solvable, the
likelihood must be implemented explicitly. An example can be found
[here](https://github.com/sebapersson/petab_sciml_testsuite/tree/main/test_cases/hybrid/001).

**Initialization tests** are specified similarly to PEtab SciML import problem.
