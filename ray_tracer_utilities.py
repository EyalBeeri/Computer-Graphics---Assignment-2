import numpy as np
from ray_tracer_constants import EPSILON
import math

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < EPSILON:
        return v
    return v / norm

def reflect(direction, normal):
    # Reflect a vector around a normal
    return direction - 2 * np.dot(direction, normal) * normal

def refract(direction, normal, n1, n2):
    # Using Snell's law: n1 sin(theta1) = n2 sin(theta2)
    # direction is incoming direction (normalized)
    # normal is surface normal (normalized), pointing outwards
    # n1, n2 are refractive indices
    cosi = -max(-1.0, min(1.0, np.dot(direction, normal)))
    eta = n1 / n2
    k = 1 - eta**2 * (1 - cosi**2)
    if k < 0:
        # total internal reflection
        return None
    return eta * direction + (eta * cosi - math.sqrt(k)) * normal

def checkerboard_color(rgb_color, x, y):
    scale_parameter = 0.5
    checkerboard = 0
    
    if x < 0:
        checkerboard += math.floor((0.5 - x) / scale_parameter)
    else:
        checkerboard += math.floor(x / scale_parameter)

    if y < 0:
        checkerboard += math.floor((0.5 - y) / scale_parameter)
    else:
        checkerboard += math.floor(y / scale_parameter)

    checkerboard = (checkerboard * 0.5) - int(checkerboard * 0.5)
    checkerboard *= 2
    if checkerboard > 0.5:
        return 0.5 * rgb_color
    return rgb_color