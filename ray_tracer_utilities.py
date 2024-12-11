import numpy as np
from ray_tracer_constants import *
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

def get_plane_local_coords(plane, hitPoint):
    # plane.normal is normalized
    n = plane.normal
    # Find a vector not parallel to n for constructing a coordinate system
    if abs(n[0]) > abs(n[1]) and abs(n[0]) > abs(n[2]):
        u = normalize(np.cross(n, np.array([0, 1, 0])))
    else:
        u = normalize(np.cross(n, np.array([1, 0, 0])))

    v = np.cross(n, u)
    # Project the hitPoint onto the plane coordinate system
    point_on_plane = plane.point_on_plane
    phit = hitPoint - point_on_plane
    local_x = np.dot(phit, u)
    local_y = np.dot(phit, v)
    return local_x, local_y


def checkerboard_color(rgbColor, x, y):
    scaleParameter = 0.5
    checkerboard = 0

    if x < 0:
        checkerboard += math.floor((0.5 - x) / scaleParameter)
    else:
        checkerboard += math.floor(x / scaleParameter)

    if y < 0:
        checkerboard += math.floor((0.5 - y) / scaleParameter)
    else:
        checkerboard += math.floor(y / scaleParameter)

    checkerboard = (checkerboard * 0.5) - int(checkerboard * 0.5)
    checkerboard *= 2
    if checkerboard > 0.5:
        return 0.5 * rgbColor
    return rgbColor