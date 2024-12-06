"""Example: The `pow_reducer()` function from PyTorch docs."""

import emoji
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.autograd.functional import hessian
from typeguard import typechecked as typechecker

from examples.common import set_seed


@jaxtyped(typechecker=typechecker)
def pow_reducer(x: Float[Tensor, "..."]) -> Float[Tensor, ""]:
    """Sum the cubed components of a tensor.

    Args:
        x: Input tensor.

    Returns:
        Scalar output tensor.

    Reference:
        https://pytorch.org/docs/stable/generated/torch.autograd.functional.hessian.html
    """
    return x.pow(3.0).sum()  # Sums all components


def demo_pow_reducer() -> None:
    """Demo Hessian calculation for the `pow_reducer()` function.

    To evaluate the Hessians of f = `pow_reducer()`, assume that x = vec(x)
    where the "vectorizing" map uses the same component ordering as
    `torch.flatten()`.

    Observe that f(x) can be written as

        f(x) = <1_n, k(x)>,

    where <.,.> is the Euclidean inner product on R^n, 1_n is the all 1's
    vector in R^n, and k: R^n --> R^n is the map

        k(x) = ((x^1)^3, ..., (x^n)^3)^t.

    By a straightforward computation, the second total derivative of k at x is

        d^2 k(x).(v, w) = (6x^1, ..., 6x^n)^t (*) v (*) w
                        = 6x (*) v (*) w

    where x (*) y is the Hadamard (elementwise) product of x and y.

    Applying the Leibniz rule, the second total derivative of f at x is

        d^2 f(x).(v, w) = <1_n, d^2 k(x).(v, w)>
                        = 6 <1_n, x (*) v (*) w>

    or alternatively

        d^2 f(x).(v, w) = 6 sum_{i=1}^{n} x^i v^i w^i
                        = v^t (6 diag(x)) w.

    In other words, the Hessian of f at x is the diagonal matrix

        Hess(f)(x) = 6 diag(x).
    """
    # Case 1: Vector input
    x = torch.randn(4)

    # Compute autograd Hessian
    # - For an input of size (n), this has shape (n, n)
    hess_autograd = hessian(pow_reducer, x)

    # Compute expected (analytical) Hessian
    hess_expected = 6.0 * torch.diag(x)

    # Compare autograd and expected Hessians
    error_msg = "Mismatched Hessians of pow_reducer() for vector input"
    assert torch.all(hess_autograd == hess_expected), error_msg

    # Case 2: Higher-order tensor input
    x = torch.randn(4, 5, 6)

    # Compute autograd Hessian
    # - For an input of size (n, m, p), this has shape (n, m, p, n, m, p)
    hess_autograd = hessian(pow_reducer, x)

    # Compute expected (analytical) Hessian for vectorized x
    hess_expected = 6.0 * torch.diag(x.flatten())

    # Compare autograd and expected Hessians
    # -- Reshaping the expected Hessian orders the components to match
    err_msg = "Mismatched Hessians of pow_reducer() for higher-order tensor input"
    assert torch.all(hess_autograd == hess_expected.reshape(hess_autograd.shape)), err_msg

    print(emoji.emojize(":sparkles: success! :sparkles:"))


if __name__ == "__main__":
    set_seed()
    demo_pow_reducer()
