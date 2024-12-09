"""Example: The `pow_adder_reducer()` function from PyTorch docs."""

import emoji
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.autograd.functional import hessian
from typeguard import typechecked as typechecker

from examples.common import set_seed


@jaxtyped(typechecker=typechecker)
def pow_adder_reducer(x: Float[Tensor, "..."], y: Float[Tensor, "..."]) -> Float[Tensor, ""]:
    """Sum a linear combination of the squared components of two tensors.

    Args:
        x: Input tensor for summand 1.
        y: Input tensor for summand 2.

    Returns:
        Scalar output tensor.

    Raises:
        ValueError: If there is a shape mismatch between x and y.

    Reference:
        https://pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html
    """
    if x.shape != y.shape:
        raise ValueError("Shape mismatch between x and y arguments")

    return (2.0 * x.pow(2.0) + 3.0 * y.pow(2.0)).sum()


def demo_pow_adder_reducer() -> None:
    """Demo Hessian calculation for `pow_adder_reducer()`.

    Observe that f: R^n x R^n --> R can be written as

        f(x, y) = 2 <1_n, s(x)> + 3 <1_n, s(y)>

    where

        * <.,.> is the Euclidean inner product on R^n,
        * 1_n is the all 1's vector in R^n, and
        * s: R^n --> R^n is the map s(x) = ((x^1)^2, ..., (x^n)^2)^t.

    The second-order total derivative of s at x is the map

        d^2 s(x).(x1, x2) = 2 x1 (*) x2,

    where v (*) w is the Hadamard (elementwise) product of v and w.

    Applying the Leibniz rule, the first-order total derivative of f at (x, y)
    is the map

        df(x, y).(x1, y1) = 2 <1_n, ds(x).x1> + 3 <1_n, ds(y).y1>.

    The second-order total derivative of f at (x, y) is the map

        d^2 f(x, y).((x1, y1), (x2, y2))
            = 2 <1_n, d^2 s(x).(x1, x2)> + 3 <1_n, d^2 s(y).(y1, y2)>
            = 4 <1_n, x1 (*) x2> + 6 <1_n, y1 (*) y2>.

    This implies that

        Hess(f)(x, y) = [Hess_{x,x} f(x, y)  Hess_{x,y} f(x, y)]
                        [Hess_{y,x} f(x, y)  Hess_{y,y} f(x, y)]
                      = [4 I_n    Z_n]
                        [  Z_n  6 I_n],

    where

        * I_n is the n-by-n identity matrix, and
        * Z_n is the n-by-n zero matrix.
    """
    # Case 1: Vector inputs
    x = torch.randn(4)
    y = torch.randn(x.shape)

    # Compute autograd Hessian
    # - For a 2-tuple input (x, y) where each component has shape (n), this is
    #   a tuple of tuples `h` such that:
    #
    #       h[0][0] = Hess_{x,x} f(x, y)
    #       h[0][1] = Hess_{x,y} f(x, y)
    #       h[1][0] = Hess_{y,x} f(x, y)
    #       h[1][1] = Hess_{y,y} f(x, y)
    #
    hess_autograd = hessian(pow_adder_reducer, (x, y))

    # Compute blocks of expected (analytical) Hessian
    hess_expected_xx = 4.0 * torch.eye(x.shape[0])
    hess_expected_yy = 6.0 * torch.eye(y.shape[0])
    hess_expected_xy = torch.zeros(x.shape[0], y.shape[0])
    hess_expected_yx = hess_expected_xy

    # Compare blocks of autograd and expected Hessians
    err_msg = "Mismatched Hessians of pow_adder_reducer() for vector inputs: "

    hess_autograd_xx = hess_autograd[0][0]
    hess_autograd_xy = hess_autograd[0][1]
    hess_autograd_yx = hess_autograd[1][0]
    hess_autograd_yy = hess_autograd[1][1]

    assert torch.all(hess_autograd_xx == hess_expected_xx), err_msg + "(x, x)"
    assert torch.all(hess_autograd_xy == hess_expected_xy), err_msg + "(x, y)"
    assert torch.all(hess_autograd_yx == hess_expected_yx), err_msg + "(y, x)"
    assert torch.all(hess_autograd_yy == hess_expected_yy), err_msg + "(y, y)"

    # Case 2: Higher-order tensor inputs
    x = torch.randn(4, 5, 6)
    y = torch.randn(x.shape)

    # Compute autograd Hessian
    # - For a 2-tuple input (x, y) where each component has shape (n, m, p),
    #   this is a tuple of tuples `h` such that:
    #
    #       h[0][0] ~ Hess_{x,x} f(x, y) has shape (n, m, p, n, m, p)
    #       h[0][1] ~ Hess_{x,y} f(x, y) has shape (n, m, p, n, m, p)
    #       h[1][0] ~ Hess_{y,x} f(x, y) has shape (n, m, p, n, m, p)
    #       h[1][1] ~ Hess_{y,y} f(x, y) has shape (n, m, p, n, m, p)
    #
    #   The tilde ~ means "represents".
    #
    #   The actual Hessian matrices can be recovered by reshaping each h[i][j],
    #   as in the actual/expected comparisons below.
    hess_autograd = hessian(pow_adder_reducer, (x, y))

    # Compute blocks of expected (analytical) Hessian
    hess_expected_xx = 4.0 * torch.eye(x.flatten().shape[0])
    hess_expected_yy = 6.0 * torch.eye(y.flatten().shape[0])
    hess_expected_xy = torch.zeros(x.flatten().shape[0], y.flatten().shape[0])
    hess_expected_yx = hess_expected_xy

    xx_shape = hess_expected_xx.shape
    xy_shape = hess_expected_xy.shape
    yx_shape = hess_expected_yx.shape
    yy_shape = hess_expected_yy.shape

    # Compare blocks of autograd and expected Hessians
    err_msg = "Mismatched Hessians of pow_adder_reducer() for higher-order tensor inputs: "

    hess_autograd_xx = hess_autograd[0][0]
    hess_autograd_xy = hess_autograd[0][1]
    hess_autograd_yx = hess_autograd[1][0]
    hess_autograd_yy = hess_autograd[1][1]

    assert torch.all(hess_autograd_xx.reshape(xx_shape) == hess_expected_xx), err_msg + "(x, x)"
    assert torch.all(hess_autograd_xy.reshape(xy_shape) == hess_expected_xy), err_msg + "(x, y)"
    assert torch.all(hess_autograd_yx.reshape(yx_shape) == hess_expected_yx), err_msg + "(y, x)"
    assert torch.all(hess_autograd_yy.reshape(yy_shape) == hess_expected_yy), err_msg + "(y, y)"

    print(emoji.emojize(":sparkles: success! :sparkles:"))


if __name__ == "__main__":
    set_seed(1)
    demo_pow_adder_reducer()
