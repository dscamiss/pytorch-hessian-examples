"""Test code."""

import torch
from torch.autograd.functional import hessian

from examples.evaluate_hessian_bilinear_product import evaluate_hessian_bilinear_product
from examples.pow_adder_reducer import pow_adder_reducer


def test_evaluate_hessian_bilinear_product() -> None:
    """Test for `evaluate_hessian()`."""
    # [1 1 1 1] [4 0 0 0] [1] = [1 1 1 1] [4] = 20.
    #           [0 4 0 0] [1]             [4]
    #           [0 0 6 0] [1]             [6]
    #           [0 0 0 6] [1]             [6]

    x = torch.randn(2)
    y = torch.randn(x.shape)

    hess = hessian(pow_adder_reducer, (x, y))
    x_arg = (torch.ones(x.shape), torch.ones(y.shape))
    y_arg = x_arg

    assert evaluate_hessian_bilinear_product(hess, x_arg, y_arg).item() == 20.0
