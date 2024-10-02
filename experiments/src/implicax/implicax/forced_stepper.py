import equinox as eqx
from .base_stepper import PicardStepper

class ForcedStepper(eqx.Module):
    unforced_stepper: PicardStepper

    def __call__(self, u, f):
        return self.unforced_stepper(u + self.unforced_stepper.dt * f)

