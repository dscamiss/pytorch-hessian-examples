"""Example: Locally-weighted logistic regression."""

import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor, nn
from torch.autograd.functional import hessian
from typeguard import typechecked as typechecker

from examples.common import set_seed

# pylint: disable=too-few-public-methods
# pylint: disable=invalid-name


class _Likelihood:
    """Implements likelihood function for locally-weighted logistic regression.

    Args:
        X: Input tensor.
        y: Output tensor.
        w: Weight tensor.
        lambda: Regularization parameter (default = 1e-4).
    """

    @jaxtyped(typechecker=typechecker)
    def __init__(  # noqa: DCO010
        self,
        X: Float[Tensor, "m n"],
        y: Float[Tensor, " m"],
        w: Float[Tensor, " m"],
        lamb: float = 1e-4,
    ) -> None:
        self._X = X
        self._y = y
        self._w = w
        self._lamb = lamb

    @jaxtyped(typechecker=typechecker)
    def likelihood(self, theta: Float[Tensor, " n"]) -> Float[Tensor, ""]:
        """Likelihood function for locally-weighted logistic regression.

        Args:
            theta: Parameter tensor.

        Returns:
            Scalar output tensor.
        """
        bce_loss = nn.BCEWithLogitsLoss(weight=self._w, reduction="sum")
        reg = (-self._lamb / 2.0) * (torch.norm(theta) ** 2.0)
        h = torch.inner(theta, self._X)
        return reg - bce_loss(h, self._y)


def demo_locally_weighted_logistic_regression() -> None:
    """Demo Hessian calculation for `Likelihood.likelihood()`.

    Observe that f: R^n --> R (the `Likelihood.likelihood()` function) can be
    written as

        f(theta) = -(lambda / 2) theta^t theta +
                   <1_m, w (*) y (*) Log(h(theta))> +
                   <1_m, w (*) (1_m - y) (*) Log(1_m - h(theta))>,

    where

        * <.,.> is the Euclidean inner product on R^m,
        * 1_m is the all 1's vector in R^m,
        * Log(v) = (log(v^1), ..., log(v^n))^t, and
        * h(theta) = (s(<theta, X_1^t>), ..., s(<theta, X_n^t>))^t, where X_i
          is the i-th row of X and s is the sigmoid function.

    The first-order total derivative of h at theta is the map

        dh(theta) = h(theta) (*) (1_m - h(theta)) (*) X a,

    where v (*) w is the Hadamard (elementwise) product of v and w.

    Using this fact, the first-order total derivative of f at theta is the map

        df(theta).a = -lambda <a, theta> + <1_m, w (*) (y - h(theta) (*) X a>

    and the second-order total derivative of f at theta is the map

        d^2 f(theta).(a, b)
            = -lambda <a, b> -
              <1_m, w (*) h(theta) (*) (1_m - h(theta)) (*) X a (*) X b>.

    This implies that

        Hess(f)(theta) = -lambda I_n +
                         X^t diag(w (*) h(theta) (*) (1_m - h(theta))) X,

    where

        * I_n is the n-by-n identity matrix, and
        * diag(v) is the n-by-n matrix whose (i, i)th entry is v^i.

    To see this, use the fact that <u, v (*) w> = v^t diag(u) w to obtain:

        <1_m, w (*) h(theta) (*) (1_m - h(theta)) (*) X a (*) X b>
            = <w (*) h(theta) (*) (1_m - h(theta)), X a (*) X b>
            = a^t X^t [w (*) h(theta) (*) (1_m - h(theta))] X b.
    """
    m, n = 3, 4
    X = torch.randn(m, n)
    y = torch.rand(m)
    w = torch.randn(m)
    lamb = 1e-4
    likelihood = _Likelihood(X, y, w, lamb)
    theta = torch.randn(n)

    # Compute autograd Hessian
    # - For an input theta of size (n), this has shape (n, n) and is equal to
    #   Hess f(theta).
    hess_autograd = hessian(likelihood.likelihood, theta)

    # Compute expected (analytical) Hessian
    h = torch.inner(theta, X).sigmoid()
    p = (w * h * (1.0 - h)).squeeze(-1)
    hess_expected = -lamb * torch.eye(n) - (X.t() @ torch.diag(p) @ X)

    # Compare autograd and expected Hessians
    # - We use `torch.allclose()` here because the Hessian of the norm-squared
    #   term is incorrect, in the sense that it contains extraneous values
    #   which are small but non-negligible.  Tested with PyTorch 2.3.0 on CPU.
    err_msg = "Mismatched Hessians of likelihood() for vector input"
    assert torch.allclose(hess_autograd, hess_expected), err_msg


if __name__ == "__main__":
    set_seed(1)
    demo_locally_weighted_logistic_regression()
