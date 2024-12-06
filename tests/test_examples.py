"""Test code for examples."""

import pytest

from examples.affine_map_summed import demo_affine_map_summed
from examples.common import set_seed
from examples.pow_adder_reducer import demo_pow_adder_reducer
from examples.pow_reducer import demo_pow_reducer


@pytest.fixture(autouse=True)
def fixture_set_seed():
    """Fixture to call `set_seed()`."""
    set_seed()


def test_demo_pow_reducer() -> None:
    """Test for `demo_pow_reducer()`."""
    demo_pow_reducer()


def test_demo_pow_adder_reducer() -> None:
    """Test for `demo_pow_adder_reducer()`."""
    demo_pow_adder_reducer()


def test_demo_affine_map_summed() -> None:
    """Test for `demo_affine_map_summed()`."""
    demo_affine_map_summed()
