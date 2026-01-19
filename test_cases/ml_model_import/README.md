# Machine-learning model import

These tests cover importing ML models defined in the PEtab SciML, language-independent YAML
format.

A key implementation detail is handling the `flatten` operation. In PEtab SciML, all tensor
data provided via HDF5 follows PyTorch’s indexing conventions. PyTorch uses row-major order
and flattens accordingly, whereas languages like Julia use column-major (Fortran) order and
flatten accordingly. The recommended, and likely most performant (due to memory access),
approach when importing into a column-major ecosystem is to convert tensors to the target
framework’s native layout before any convolutions or `flatten`. For example, PyTorch expects
image tensors in `(C, H, W)`, while Julia uses `(H, W, C)`. If, during ML-model import into
Julia, both the input data and the convolution kernels (and similar layers) are permuted to
`(H, W, C)` to match Julia’s indexing convention, subsequent `flatten` operations will
produce outputs consistent with the reference. See test case `018` for a concrete example.
