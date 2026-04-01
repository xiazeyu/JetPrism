from jetprism.schemas import EventNumpy
import numpy as np
import vector


def mpipi(event: EventNumpy) -> np.ndarray:
    """
    This function calculates the di-pion invariant mass.
    The distribution of this invariant mass value would show a peak at the mass of that parent particle.
    """
    return (event.k1 + event.k2).mass


def t(event: EventNumpy) -> np.ndarray:
    """
    This function calculates the Mandelstam variable t.
    The variable t is related to the scattering angle and the momentum exchanged during the interaction.
    """
    return (event.p1 - event.p2).mass2


def s(event: EventNumpy) -> np.ndarray:
    """
    This function calculates the Mandelstam variable s.
    This variable represents the square of the total center-of-mass energy available in the collision.
    It determines what new particles can be created.
    """
    return (event.q + event.p1).mass2


def s12(event: EventNumpy) -> np.ndarray:
    """
    This function calculates squared invariant mass of the k1 and k2 system.
    This is a Lorentz-invariant quantity representing the sub-energy of the (k1, k2) system.
    """
    return (event.k1 + event.k2).mass2


def cos_theta(event: EventNumpy) -> np.ndarray:
    """
    This function calculates cosine of the polar angle (θ) of particle k1.
    This describes the "up-down" direction of the particle's trajectory in a specific coordinate system.
    """
    z_axis = vector.obj(x=0, y=0, z=1)
    cos_theta = z_axis.dot(event.k1.to_3D()) / event.k1.mag
    return cos_theta


def phi(event: EventNumpy) -> np.ndarray:
    """
    This function calculates the azimuthal angle of particle k1.
    This describes the "left-right" or rotational orientation of the particle's trajectory around the z-axis.
    """
    angle = np.arctan2(event.k1.y, event.k1.x)
    return angle % (2 * np.pi)
