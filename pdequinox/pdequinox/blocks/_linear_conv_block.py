from typing import Callable, Literal

from jaxtyping import PRNGKeyArray

from ..conv import PhysicsConv
from ._base_block import BlockFactory


class LinearConvBlock(PhysicsConv):
    def __init__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        use_bias: bool = True,
        zero_bias_init: bool = False,
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],
        key: PRNGKeyArray,
    ):
        super().__init__(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            boundary_mode=boundary_mode,
            use_bias=use_bias,
            zero_bias_init=zero_bias_init,
            key=key,
        )


class LinearConvBlockFactory(BlockFactory):
    kernel_size: int
    use_bias: bool

    def __init__(
        self,
        *,
        kernel_size: int = 3,
        use_bias: bool = True,
    ):
        """
        Factory for creating `LinearConvBlock` instances.

        **Arguments:**

        - `kernel_size`: The size of the convolutional kernel. Default is `3`.
        - `use_bias`: Whether to use bias in the convolutional layers. Default is
            `True`.
        """
        self.kernel_size = kernel_size
        self.use_bias = use_bias

    def __call__(
        self,
        num_spatial_dims: int,
        in_channels: int,
        out_channels: int,
        *,
        activation: Callable,  # unused
        boundary_mode: Literal["periodic", "dirichlet", "neumann"],
        key: PRNGKeyArray,
    ):
        return LinearConvBlock(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            boundary_mode=boundary_mode,
            use_bias=self.use_bias,
            key=key,
        )
