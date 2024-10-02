from .utilities import *
from .burgers import Burgers, Burgers2d
from .navier_stokes import NavierStokes
from .navier_stokes_lid_driven import NavierStokesLidDriven
from .kuramoto_sivashisnky import KS
from .heat import Heat, Heat2d
from .forced_stepper import ForcedStepper
from .initial_conditions import (
    SineWavesIC,
    RandomSineWavesIC,
    DiscontinuityIC,
    RandomDiscontinuityIC,
    GaussianRandomField,
    RandomGaussianRandomField,
)