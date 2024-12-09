"""Simple function to evaluate Hessian bilinear products."""

from typing import Union

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.autograd.functional import hessian
from typeguard import typechecked as typechecker

from examples.affine_map_summed import affine_map_summed
from examples.common import set_seed
from examples.pow_adder_reducer import pow_adder_reducer

_HessianSingleInputType = Float[Tensor, "..."]
_HessianMultiInputType = tuple[tuple[Float[Tensor, "..."], ...], ...]
_HessianType = Union[_HessianSingleInputType, _HessianMultiInputType]
_InputType = Union[Float[Tensor, "..."], tuple[Float[Tensor, "..."], ...]]

# pylint: disable=consider-using-enumerate
# pylint: disable=invalid-name


@jaxtyped(typechecker=typechecker)
def evaluate_hessian_bilinear_product(
    hess: _HessianType, x: _InputType, y: _InputType
) -> Float[Tensor, ""]:
    """Evaluate Hessian bilinear product.

    This computes x^t H y, where H is the Hessian matrix represented by `hess`.

    Args:
        hess: The output of `F.hessian()`.
        x: Input for "x" argument.
        y: Input for "y" argument.

    Raises:
        ValueError: If arguments have mismatched types or dimensions.  The
        expectation is that `x`, `y` should have the same type and dimensions
        as the inputs to `F.hessian()` that were used to compute `hess`.

        More explicitly, if `hess` was computed by

            `hess = F.hessian(func, inputs, ...)`,

        then `x`, `y` should have the same type and dimensions as `inputs`.

    Returns:
        Scalar output tensor.
    """
    # Sanity check on argument types
    if not type(hess) is type(x) is type(y):
        raise ValueError("h, x, y have mismatched types")

    # Promote non-tuples to 1-tuples, for common handling
    if not isinstance(hess, tuple):
        hess, x, y = tuple(hess), tuple(x), tuple(y)

    # Get tuple lengths
    nh, nx, ny = len(hess), len(x), len(y)

    # Sanity check on tuple lengths
    if nh != nx:
        raise ValueError("h, x have mismatched lengths {nh}, {nx}")
    for i in range(nh):
        if len(hess[i]) != ny:
            raise ValueError(f"hess[{i}], y have mismatched lengths {len(hess[i])}, {ny}")

    # Sanity check on dimensions
    for i in range(nh):
        for j in range(nh):
            if hess[i][j].numel() != x[i].numel() * y[j].numel():
                err_msg = f"hess[{i}][{j}], x[{i}], y[{j}] have mismatched dimensions \
                    {hess[i][j].numel()}, {x[i].numel()}, {y[j].numel()}"
                raise ValueError(err_msg)

    # Evaluate bilinear product
    res = torch.as_tensor(0.0)

    for i in range(nh):
        xi_flat = x[i].flatten().unsqueeze(-1)
        for j in range(nh):
            yj_flat = y[j].flatten().unsqueeze(-1)
            hij_matrix = hess[i][j].view(xi_flat.shape[0], yj_flat.shape[0])
            res += (xi_flat.transpose(-1, -2) @ hij_matrix @ yj_flat).item()

    return res


def demo_evaluate_hessian_bilinear_product_affine_map_summed():
    """Demo `evaluate_hessian_bilinear_product()` for `affine_map_summed()`."""
    n, m = 3, 4
    W = torch.randn(n, m)
    b = torch.randn(n, 1)
    x = torch.randn(m, 1)

    hess = hessian(affine_map_summed, (W, b, x))
    x_arg = (torch.randn(W.shape), torch.randn(b.shape), torch.randn(x.shape))
    y_arg = (torch.randn(W.shape), torch.randn(b.shape), torch.randn(x.shape))

    print(f"result = {evaluate_hessian_bilinear_product(hess, x_arg, y_arg)}")


def demo_evaluate_hessian_bilinear_product_pow_adder_reducer() -> None:
    """Demo `evaluate_hessian_bilinear_product()` for `pow_adder_reducer()`."""
    x = torch.randn(2)
    y = torch.randn(x.shape)

    hess = hessian(pow_adder_reducer, (x, y))
    x_arg = (torch.ones(x.shape), torch.ones(y.shape))
    y_arg = x_arg

    print(f"result = {evaluate_hessian_bilinear_product(hess, x_arg, y_arg)}")


if __name__ == "__main__":
    set_seed(1)
    demo_evaluate_hessian_bilinear_product_affine_map_summed()
    demo_evaluate_hessian_bilinear_product_pow_adder_reducer()
