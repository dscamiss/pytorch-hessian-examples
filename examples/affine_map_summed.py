"""Example: Sum the outputs of an affine map with ReLU activation."""

import emoji
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.autograd.functional import hessian
from typeguard import typechecked as typechecker

from examples.common import set_seed

# pylint: disable=invalid-name


@jaxtyped(typechecker=typechecker)
def affine_map_summed(
    W: Float[Tensor, "n m"], b: Float[Tensor, "n 1"], x: Float[Tensor, "m 1"]
) -> Float[Tensor, ""]:
    """Sum the outputs of an affine map with ReLU activation.

    Args:
        W: Weight tensor.
        b: Bias tensor.
        x: Input tensor.

    Returns:
        Output tensor.
    """
    return ((W @ x) + b).relu().sum()  # pylint: disable=not-callable


def demo_affine_map_summed() -> None:
    """Demo Hessian calculation for `affine_map_summed()`.

    Observe that f: R^{n x m} x R^n x R^m -> R can be written as

        f(W, b, x) = <1_n, ReLU(W x + b)>,

    where

        * <.,.> is the Euclidean inner product on R^n, and
        * 1_n is the all 1's vector in R^n.

    Applying the Leibniz rule, the first-order total derivative of f at
    (W, b, x) is the map

        df(W, b, x).(W1, b1, x1) = <1_n, ReLU'(z) (*) W1 x> +
                                   <1_n, ReLU'(z) (*) b1> +
                                   <1_n, ReLU'(z) (*) W x1>,

    where

        * z = W x + b, and
        * v (*) w is the Hadamard (elementwise) product of v and w.

    The second-order total derivative of f at (W, b, x) is the map

        d^2 f(W, b, x).((W1, b1, x1), (W2, b2, x2)) =
            <1_n, ReLU''(z) (*) W1 x (*) W2 x> +
            <1_n, ReLU''(z) (*) W1 x (*) b2> +
            <1_n, ReLU''(z) (*) W1 x (*) W x2> + <1_n, ReLU'(z) (*) W1 x2> +
            <1_n, ReLU''(z) (*) b1 (*) W2 x> +
            <1_n, ReLU''(z) (*) b1 (*) b2> +
            <1_n, ReLU''(z) (*) b1 (*) W x2> +
            <1_n, ReLU''(z) (*) W x1 (*) W2 x> + <1_n, ReLU'(z) (*) W2 x1> +
            <1_n, ReLU''(z) (*) W x1 (*) b2> +
            <1_n, ReLU''(z) (*) W x1 (*) W x2>.

    Since ReLU'' = 0 (where it is well-defined), this simplifies to

        d^2 f(W, b, x).((W1, b1, x1), (W2, b2, x2)) =
            <1_n, ReLU'(z) (*) (W1 x2)> + <1_n, ReLU'(z) (*) (W2 x1)>.

    This implies that

        Hess(f)(W, b, x)
            = [Hess_{W, W} f(#)  Hess_{W, b} f(#)  Hess_{W, x} f(#)]
              [Hess_{b, W} f(#)  Hess_{b, b} f(#)  Hess_{b, x} f(#)]
              [Hess_{x, W} f(#)  Hess_{x, b} f(#)  Hess_{x, x} f(#)]
            = [Z_{nm,nm}           Z_{nm,n}  ReLU'(z) (X) I_m]
              [Z_{n,nm}            Z_{n,n}   Z_{n,m}         ]
              [ReLU'(z)^t (X) I_m  Z_{m,n}   Z_{m,n}         ],

    where

        * # is shorthand for (W, b, x),
        * Z_{k,l} is the k-by-l zero matrix,
        * I_m is the m-by-m identity matrix, and
        * A (X) B is the Kronecker product of A and B.

    For example, Hess_{W, x} f(#) is the nm-by-n matrix M such that

        vec(W)^t M x = <1_n, ReLU'(z) (*) (W x)>.

    Using the identity (see, e.g., the Matrix Cookbook)

        vec(A)^t (D (X) B) c = trace(A^t B c D^t)

    we see that

        vec(W)^t [ReLU'(z) (X) I_m] x
            = trace(W I_m x ReLU'(z)')
            = trace(W x ReLU'(z)^t)
            = trace(ReLU'(z)^t W x)
            = <1_n, ReLU'(z) (*) (W x)>,

    so that M = ReLU'(z) (X) I_m.

    By symmetry,

        Hess_{x, W} f(#) = (ReLU'(z) (X) I_m)^t = ReLU'(z)^t (X) I_m.
    """
    n, m = 3, 4
    W = torch.randn(n, m)
    b = torch.randn(n, 1)
    x = torch.randn(m, 1)
    z = (W @ x) + b
    relu_deriv_z = torch.where(z <= 0.0, 0.0, 1.0)

    # Compute autograd Hessian
    # - For a 3-tuple input (W, b, x) where the components have shapes (n, m),
    #   (n), and (m), respectively, this is a tuple of tuples `h` such that:
    #
    #       h[0][0] ~ Hess_{W,W} f(W, b, x) has shape (n, m, n, m)
    #       h[0][1] ~ Hess_{W,b} f(W, b, x) has shape (n, m, n)
    #       h[0][2] ~ Hess_{W,x} f(W, b, x) has shape (n, m, m)
    #       h[1][0] ~ Hess_{b,W} f(W, b, x) has shape (n, n, m)
    #       h[1][1] = Hess_{b,b} f(W, b, x) has shape (n, n)
    #       h[1][2] = Hess_{b,x} f(W, b, x) has shape (n, m)
    #       h[2][0] ~ Hess_{x,W} f(W, b, x) has shape (m, n, m)
    #       h[2][1] = Hess_{x,b} f(W, b, x) has shape (m, n)
    #       h[2][2] = Hess_{x,x} f(W, b, x) has shape (m, m)
    #
    #   The tilde ~ means "represents".
    #
    #   For the "tilde cases," the actual Hessian matrices can be recovered by
    #   reshaping each h[i][j], as in the actual/expected comparisons below.
    hess_autograd = hessian(affine_map_summed, (W, b, x))

    # Compute blocks of expected (analytical) Hessian
    W_dim = n * m

    hess_expected_WW = torch.zeros(W_dim, W_dim)
    hess_expected_Wb = torch.zeros(W_dim, n)
    hess_expected_Wx = torch.kron(relu_deriv_z, torch.eye(m))
    hess_expected_bW = torch.zeros(n, W_dim)
    hess_expected_bb = torch.zeros(n, n)
    hess_expected_bx = torch.zeros(n, m)
    hess_expected_xW = torch.kron(relu_deriv_z.transpose(-1, -2), torch.eye(m))
    hess_expected_xb = torch.zeros(m, n)
    hess_expected_xx = torch.zeros(m, m)

    WW_shape = hess_expected_WW.shape
    Wb_shape = hess_expected_Wb.shape
    Wx_shape = hess_expected_Wx.shape
    bW_shape = hess_expected_bW.shape
    bb_shape = hess_expected_bb.shape
    bx_shape = hess_expected_bx.shape
    xW_shape = hess_expected_xW.shape
    xb_shape = hess_expected_xb.shape
    xx_shape = hess_expected_xx.shape

    # Compare blocks of autograd and expected Hessians
    err_msg = "Mismatched Hessians of affine_map_summed(): "

    hess_autograd_WW = hess_autograd[0][0]
    hess_autograd_Wb = hess_autograd[0][1]
    hess_autograd_Wx = hess_autograd[0][2]
    hess_autograd_bW = hess_autograd[1][0]
    hess_autograd_bb = hess_autograd[1][1]
    hess_autograd_bx = hess_autograd[1][2]
    hess_autograd_xW = hess_autograd[2][0]
    hess_autograd_xb = hess_autograd[2][1]
    hess_autograd_xx = hess_autograd[2][2]

    assert torch.all(hess_autograd_WW.reshape(WW_shape) == hess_expected_WW), err_msg + "(W, W)"
    assert torch.all(hess_autograd_Wb.reshape(Wb_shape) == hess_expected_Wb), err_msg + "(W, b)"
    assert torch.all(hess_autograd_Wx.reshape(Wx_shape) == hess_expected_Wx), err_msg + "(W, x)"
    assert torch.all(hess_autograd_bW.reshape(bW_shape) == hess_expected_bW), err_msg + "(b, W)"
    assert torch.all(hess_autograd_bb.reshape(bb_shape) == hess_expected_bb), err_msg + "(b, b)"
    assert torch.all(hess_autograd_bx.reshape(bx_shape) == hess_expected_bx), err_msg + "(b, x)"
    assert torch.all(hess_autograd_xW.reshape(xW_shape) == hess_expected_xW), err_msg + "(x, W)"
    assert torch.all(hess_autograd_xb.reshape(xb_shape) == hess_expected_xb), err_msg + "(x, b)"
    assert torch.all(hess_autograd_xx.reshape(xx_shape) == hess_expected_xx), err_msg + "(x, x)"

    print(emoji.emojize(":sparkles: success! :sparkles:"))


if __name__ == "__main__":
    set_seed()
    demo_affine_map_summed()
