import math

import numpy as np


def cart2pol(cart: np.array) -> np.array:
    """
    :param cart: An array with two values. 0: x, 1: y
    :return: An array with two values. 0: length, 1: direction in rad
    """
    r = np.sqrt(cart[0] ** 2 + cart[1] ** 2)
    phi = norm_angle(np.arctan2(cart[1], cart[0]))
    return np.array([r, phi])


def pol2cart(pol: np.array) -> np.array:
    """
    :param pol: An array with two values. 0: length, 1: direction in rad
    :return: An array with two values. 0: x, 1: y
    """
    x = pol[0] * np.cos(pol[1])
    y = pol[0] * np.sin(pol[1])
    return np.array([x, y])


def f_pol(pol: np.array) -> str:
    """
    Format polar coordinates
    :param pol: An array with two values. 0: length, 1: direction in rad
    :return: A string easy readable
    """
    return f"[{pol[0]:.3f}, {pol[1]:.3f}r|{math.degrees(pol[1]):.3f}d]"


def norm_angle(phi: float) -> float:
    """
    Normalize an angle between 0 and 2 * pi
    :param phi: The angle
    :return:  normalized
    """
    if phi < 0.0:
        return phi + (math.pi * 2.0)
    return phi
