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
        x: Input tensor 1.
        y: Input tensor 2.

    Returns:
        Scalar output tensor.

    Raises:
        ValueError: If there is a shape mismatch between x and y.

    Reference:
        https://pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html
    """
    if x.shape != y.shape:
        raise ValueError("Shape mismatch between x and y arguments")

    return (2.0 * x.pow(2.0) + 3.0 * y.pow(2.0)).sum()  # Sums all compoennts


def demo_pow_adder_reducer() -> None:
    """Demo Hessian calculation for the `pow_adder_reducer()` function.

    To evaluate the Hessians of f = `pow_adder_reducer()`, assume that
    x = vec(x) and y = vec(y), where the "vectorizing" map uses the same
    component ordering as `torch.flatten()`.

    Observe that f(x, y) can be written as

        f(x, y) = 2 <1_n, s(x)> + 3 <1_n, s(y)>

    where <.,.> is the Euclidean inner product on R^n, 1_n is the all 1's
    vector in R^n, and s: R^n --> R^n is the map

        s(x) = ((x^1)^2, ..., (x^n)^2)^t.

    The second total derivative of s at x is

        d^2 s(x).(v, w) = 2 v (*) w

    where v (*) w is the Hadamard (elementwise) product of v and w.

    Applying the Leibniz rule, the first-order total derivative of f at x is

        df(x, y).(v, w) = 2 <1_n, ds(x).v> + 3 <1_n, ds(y).w>

    and the second-order total derivative of f at x is

        d^2 f(x, y).((v, w), (a, b))
            = 2 <1_n, d^2 s(x).(v, a)> + 3 <1_n, d^2 s(y).(w, b)>
            = 4 <1_n, v (*) a> + 6 <1_n, w (*) b>

    or alternatively

        d^2 f(x, y).((v, w), (a, b))
            = 4 sum_{i=1}^{n} v^i a^i + 6 sum_{i=1}^{n} w^i b^i.

    In other words, the Hessian of f at (x, y) is the block-diagonal matrix

        Hess(f)(x, y) = [Hess_{x,x} f(x, y)  Hess_{x,y} f(x, y)]
                        [Hess_{y,x} f(x, y)  Hess_{y,y} f(x, y)]
                      = [4 I_n    Z_n]
                        [  Z_n  6 I_n],

    where I_n is the n-by-n identity matrix and Z_n is the n-by-n zero matrix.
    """
    # Case 1: Vector inputs
    x = torch.randn(4)
    y = torch.randn(x.shape)

    # Compute autograd Hessian
    # - For a 2-tuple input (x, y) where each component has shape (n), this is
    #   a tuple of tuples `h` such that:
    #
    #       h[0][0] ~ Hess_{x,x} f(x, y) has shape (n, n)
    #       h[0][1] ~ Hess_{x,y} f(x, y) has shape (n, n)
    #       h[1][0] ~ Hess_{y,x} f(x, y) has shape (n, n)
    #       h[1][1] ~ Hess_{y,y} f(x, y) has shape (n, n)
    #
    hess_autograd = hessian(pow_adder_reducer, (x, y))

    # Compute blocks of expected (analytical) Hessian
    hess_expected_xx = 4.0 * torch.eye(x.shape[0])
    hess_expected_yy = 6.0 * torch.eye(y.shape[0])
    hess_expected_xy = torch.zeros(x.shape[0], y.shape[0])
    hess_expected_yx = hess_expected_xy

    # Compare blocks of autograd and expected Hessians
    err_msg = "Mismatched Hessians of pow_adder_reducer() for vector inputs"

    hess_autograd_xx = hess_autograd[0][0]
    hess_autograd_xy = hess_autograd[0][1]
    hess_autograd_yx = hess_autograd[1][0]
    hess_autograd_yy = hess_autograd[1][1]

    assert torch.all(hess_autograd_xx == hess_expected_xx), err_msg
    assert torch.all(hess_autograd_xy == hess_expected_xy), err_msg
    assert torch.all(hess_autograd_yx == hess_expected_yx), err_msg
    assert torch.all(hess_autograd_yy == hess_expected_yy), err_msg

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
    # - Specifically, we have the relationships
    #
    #       mat(h[0][0]) = Hess_{vec(x), vec(y)} f(vec(x), vec(y))
    #       mat(h[0][0]) = Hess_{vec(x), vec(y)} f(vec(x), vec(y))
    #       mat(h[0][0]) = Hess_{vec(y), vec(x)} f(vec(x), vec(y))
    #       mat(h[0][0]) = Hess_{vec(y), vec(y)} f(vec(x), vec(y))
    #
    #   where mat is the inverse of the "vectorization" map.
    #
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
    err_msg = "Mismatched Hessians of pow_adder_reducer() for higher-order tensor inputs"

    hess_autograd_xx = hess_autograd[0][0]
    hess_autograd_xy = hess_autograd[0][1]
    hess_autograd_yx = hess_autograd[1][0]
    hess_autograd_yy = hess_autograd[1][1]

    assert torch.all(hess_autograd_xx.reshape(xx_shape) == hess_expected_xx), err_msg
    assert torch.all(hess_autograd_xy.reshape(xy_shape) == hess_expected_xy), err_msg
    assert torch.all(hess_autograd_yx.reshape(yx_shape) == hess_expected_yx), err_msg
    assert torch.all(hess_autograd_yy.reshape(yy_shape) == hess_expected_yy), err_msg

    print(emoji.emojize(":sparkles: success! :sparkles:"))


if __name__ == "__main__":
    set_seed()
    demo_pow_adder_reducer()
