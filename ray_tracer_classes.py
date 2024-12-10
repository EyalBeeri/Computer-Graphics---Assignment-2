import numpy as np
from ray_tracer_utilities import *
from ray_tracer_constants import *

class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = normalize(direction)

class Intersection:
    def __init__(self, t=float('inf'), point=None, normal=None, obj=None):
        self.t = t
        self.point = point
        self.normal = normal
        self.obj = obj

class Material:
    # For normal objects: K_A, K_D from input c lines; K_S=(0.7,0.7,0.7), n from c line; K_R=0 if normal
    # For reflective: ignore c line (use reflection color = (1,1,1))
    # For transparent: ignore c line (use refraction color = from behind the object)
    def __init__(self, ambient=None, diffuse=None, shininess=10.0, reflective=False, transparent=False):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = np.array([0.7,0.7,0.7])
        self.shininess = shininess
        self.reflective = reflective
        self.transparent = transparent

class Object3D:
    def intersect(self, ray: Ray) -> Intersection:
        raise NotImplementedError()

    def get_material(self):
        return self.material

class Sphere(Object3D):
    def __init__(self, center, radius, material):
        self.center = np.array(center)
        self.radius = radius
        self.material = material

    def intersect(self, ray:Ray):
        L = self.center - ray.origin
        tca = np.dot(L, ray.direction)
        d2 = np.dot(L, L) - tca*tca
        if d2 > self.radius*self.radius:
            return Intersection()
        thc = math.sqrt(self.radius*self.radius - d2)
        t0 = tca - thc
        t1 = tca + thc
        # we want the closest positive t
        t = float('inf')
        if t0 > EPSILON and t1 > EPSILON:
            t = min(t0, t1)
        elif t1 > EPSILON:
            t = t1
        elif t0 > EPSILON:
            t = t0

        if t == float('inf'):
            return Intersection()

        point = ray.origin + t * ray.direction
        normal = normalize(point - self.center)
        return Intersection(t, point, normal, self)

class Plane(Object3D):
    def __init__(self, a,b,c,d, material):
        self.normal_un = np.array([a,b,c])
        self.d = d
        self.material = material
        self.normal = normalize(self.normal_un)
        # Find a point on the plane:
        # If c !=0:
        #    point_on_plane = (0,0,-d/c)
        # else try another coordinate:
        if abs(c) > 1e-9:
            self.point_on_plane = np.array([0.0,0.0,-self.d/c])
        elif abs(b) > 1e-9:
            self.point_on_plane = np.array([0.0,-self.d/b,0.0])
        else:
            self.point_on_plane = np.array([-self.d/a,0.0,0.0])
    def intersect(self, ray:Ray):
        denom = np.dot(self.normal_un, ray.direction)
        if abs(denom) < EPSILON:
            return Intersection()

        t = -(self.d + np.dot(self.normal_un, ray.origin)) / denom
        if t < EPSILON:
            return Intersection()
        point = ray.origin + t * ray.direction
        # normal is well-defined (plane normal)
        normal = self.normal
        return Intersection(t, point, normal, self)


# Light classes
class Light:
    pass

class DirectionalLight(Light):
    def __init__(self, direction, intensity):
        self.direction = normalize(direction)
        self.intensity = intensity

    def get_direction(self, point):
        # for directional light, direction is constant
        return -self.direction, float('inf') # distance infinite

class Spotlight(Light):
    def __init__(self, position, direction, intensity, cutoff):
        self.position = position
        self.direction = normalize(direction)
        self.intensity = intensity
        self.cutoff = cutoff  # cosine of angle

    def get_direction(self, point):
        L = self.position - point
        dist = np.linalg.norm(L)
        if dist < EPSILON:
            return None,0
        return normalize(L), dist