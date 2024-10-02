import jax.numpy as jnp
import jax.random as jr
from typing import List

class SineWavesIC:
    def __init__(
        self,
        L: float,
        wave_numbers: List[int],
        amplitudes: List[float],
        phases: List[float],
        offsets: List[float],
    ):
        self.L = L
        self.wave_numbers = wave_numbers
        self.amplitudes = amplitudes
        self.phases = phases
        self.offsets = offsets

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return sum(
            [
                self.amplitudes[i]
                * jnp.sin(
                    2 * jnp.pi / self.L * self.wave_numbers[i] * x - self.phases[i]
                )
                + self.offsets[i]
                for i in range(len(self.wave_numbers))
            ],
        )
    
class RandomSineWavesIC:
    def __init__(
        self,
        L: float,
        *,
        n_wave_numbers: int = 5,
        ampltiude_range = [-1.0, 1.0],
        phase_range = [-jnp.pi, jnp.pi],
        offset_range = [0.0, 0.0],
    ):
        self.L = L
        self.n_wave_numbers = n_wave_numbers
        self.ampltiude_range = ampltiude_range
        self.phase_range = phase_range
        self.offset_range = offset_range

    def __call__(self, key):
        wave_numbers = jnp.arange(1, self.n_wave_numbers + 1)
        a_key, p_key, o_key = jr.split(key, 3)
        amplitudes = jr.uniform(a_key, shape=(self.n_wave_numbers,), minval=self.ampltiude_range[0], maxval=self.ampltiude_range[1])
        phases = jr.uniform(p_key, shape=(self.n_wave_numbers,), minval=self.phase_range[0], maxval=self.phase_range[1])
        offsets = jr.uniform(o_key, shape=(self.n_wave_numbers,), minval=self.offset_range[0], maxval=self.offset_range[1])
        return SineWavesIC(self.L, wave_numbers, amplitudes, phases, offsets)


class DiscontinuityIC:
    def __init__(
        self,
        left_positions: List[float],
        right_positions: List[float],
        heights: List[float],
    ):
        self.left_positions = left_positions
        self.right_positions = right_positions
        self.heights = heights

    def _one_discontinuity(
        self,
        x,
        *,
        left_position: float,
        right_position: float,
        height: float,
    ):
        return jnp.where(
            (x >= left_position) & (x <= right_position),
            height,
            0.0,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return sum(
            [
                self._one_discontinuity(
                    x,
                    left_position=l,
                    right_position=r,
                    height=h,
                )
                for l, r, h in zip(
                    self.left_positions,
                    self.right_positions,
                    self.heights,
                )
            ],
        )
    
class RandomDiscontinuityIC:
    def __init__(
        self,
        L,
        *,
        n_discontinuities: int = 3,
        left_position_fraction_range = [0.0, 0.5],
        right_position_fraction_range = [0.5, 1.0],
        height_range = [-1.0, 1.0],
    ):
        self.L = L
        self.n_discontinuities = n_discontinuities
        self.left_position_fraction_range = left_position_fraction_range
        self.right_position_fraction_range = right_position_fraction_range
        self.height_range = height_range

    def __call__(self, key):
        l_key, r_key, h_key = jr.split(key, 3)
        left_positions = self.L * jr.uniform(l_key, shape=(self.n_discontinuities,), minval=self.left_position_fraction_range[0], maxval=self.left_position_fraction_range[1])
        right_positions = self.L * jr.uniform(r_key, shape=(self.n_discontinuities,), minval=self.right_position_fraction_range[0], maxval=self.right_position_fraction_range[1])
        heights = jr.uniform(h_key, shape=(self.n_discontinuities,), minval=self.height_range[0], maxval=self.height_range[1])
        return DiscontinuityIC(left_positions, right_positions, heights)

class GaussianRandomField:
    """
    Based on
    https://github.com/bsciolla/gaussian-random-fields/blob/master/gaussian_random_fields.py
    """

    def __init__(
        self,
        L,
        *,
        alpha: float,
        normalize: bool = True,
        key,
    ):
        self.L = L
        self.alpha = alpha
        self.normalize = normalize
        self.key = key

    def __call__(self, x):
        N = len(x)
        k = jnp.fft.rfftfreq(N, 1/N)
        n_real_wavenumbers = len(k)
        noise_hat = (
            jr.normal(self.key, shape=(n_real_wavenumbers,))
            + 1j * jr.normal(self.key, shape=(n_real_wavenumbers,))
        )
        amplitude = jnp.power(k**2 + 1e-10, - self.alpha / 2)
        amplitude = amplitude.at[0].set(0.0)
        noise_filtered_hat = noise_hat * amplitude
        noise_filtered = jnp.fft.irfft(noise_filtered_hat, N)

        if self.normalize:
            noise_filtered -= jnp.mean(noise_filtered, keepdims=True)
            noise_filtered /= jnp.std(noise_filtered, keepdims=True)

        return noise_filtered

class RandomGaussianRandomField:
    def __init__(
        self,
        L,
        *,
        alpha: float = 3.0,
        normalize: bool = True,
    ):
        self.L = L
        self.alpha = alpha
        self.normalize = normalize

    def __call__(self, key):
        return GaussianRandomField(
            self.L,
            alpha=self.alpha,
            normalize=self.normalize,
            key=key,
        )