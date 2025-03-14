from math import prod
from typing import Callable, Literal, Optional

import equinox as eqx
import jax
from jaxtyping import PRNGKeyArray


def _identity(x):
    return x


class MLP(eqx.Module):
    num_spatial_dims: int
    num_points: int
    in_channels: int
    out_channels: int
    flat_mlp: eqx.nn.MLP

    _in_shape: tuple[int, ...]
    _out_shape: tuple[int, ...]

    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        num_points: int,
        width_size: int = 64,
        depth: int = 3,
        activation: Callable = jax.nn.relu,
        final_activation: Callable = _identity,
        use_bias: bool = True,
        use_final_bias: bool = True,
        boundary_mode: Optional[
            Literal["periodic", "dirichlet", "neumann"]
        ] = None,  # unused
        key: PRNGKeyArray,
    ):
        """
        A MLP for the conv format.

        Takes states of shape `(in_channels, #num_points)` with a leading
        `in_channels` axis and as many spatial axes as `num_spatial_dims`.
        Internally, the input is flattened and given to a classical MLP. The
        conv shape is restored in the end.

        Contrary to convolutional networks, the `num_points` must be supplied!

        **Arguments:**

        - `num_spatial_dims`: The number of spatial dimensions. For example
            traditional convolutions for image processing have this set to `2`.
        - `in_channels`: The number of input channels.
        - `out_channels`: The number of output channels.
        - `num_points`: The number of points in each spatial dimension. Must be
            supplied.
        - `width_size`: The width of the hidden layers. Default is `64`.
        - `depth`: The number of hidden layers. Default is `3`. The number of
            linear-affine transformations performed is `depth + 1`.
        - `activation`: The activation function to use in the hidden layers.
            Default is `jax.nn.relu`.
        - `final_activation`: The activation function to use in the final layer.
            Default is the identity function.
        - `use_bias`: If `True`, uses bias in the hidden layers. Default is
            `True`.
        - `use_final_bias`: If `True`, uses bias in the final layer. Default is
            `True`.
        - `boundary_mode`: Unused, just for compatibility with other architectures.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        self.num_spatial_dims = num_spatial_dims
        self.num_points = num_points
        self.in_channels = in_channels
        self.out_channels = out_channels

        self._in_shape = (in_channels,) + (num_points,) * num_spatial_dims
        self._out_shape = (out_channels,) + (num_points,) * num_spatial_dims
        flat_in_size = prod(self._in_shape)
        flat_out_size = prod(self._out_shape)

        self.flat_mlp = eqx.nn.MLP(
            in_size=flat_in_size,
            out_size=flat_out_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            use_bias=use_bias,
            use_final_bias=use_final_bias,
            key=key,
        )

    def __call__(self, x):
        if x.shape != self._in_shape:
            raise ValueError(
                f"Input shape {x.shape} does not match expected shape {self._in_shape}. For batched operation use jax.vmap"
            )
        x_flat = x.flatten()
        x_flat = self.flat_mlp(x_flat)
        x = x_flat.reshape(self._out_shape)
        return x

    @property
    def receptive_field(self) -> tuple[tuple[float, float], ...]:
        return ((self.num_points, self.num_points),) * self.num_spatial_dims
