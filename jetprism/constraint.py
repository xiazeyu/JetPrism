# All values here should be close to 0.

from hepunits.units import MeV, GeV
from jetprism.schemas import EventNumpy
from particle.literals import photon, proton, pi_plus, pi_minus
import numpy as np


def momentum_conservation(event: EventNumpy) -> np.float64:

    return (event.q + event.p1).subtract(event.p2 + event.k1 + event.k2).mass


def energy_conservation(event: EventNumpy) -> np.float64:

    return np.float64((event.q.e + event.p1.e) - (event.p2.e + event.k1.e + event.k2.e))


def mass_conservation(event: EventNumpy) -> np.ndarray:

    return np.array([
        event.q.mass - (photon.mass * MeV) / GeV,
        event.p1.mass - (proton.mass * MeV) / GeV,
        event.p2.mass - (proton.mass * MeV) / GeV,
        event.k1.mass - (pi_plus.mass * MeV) / GeV,
        event.k2.mass - (pi_minus.mass * MeV) / GeV,
    ])


def zero_momentum(event: EventNumpy) -> np.ndarray:

    return np.array([event.q.py, event.p1.py, event.p2.px, event.p2.py])
