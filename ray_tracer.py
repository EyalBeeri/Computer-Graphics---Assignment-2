import sys
import math
import numpy as np
import cv2

EPSILON = 1e-5
MAX_RECURSION_DEPTH = 5

# -----------------------
# Utility functions
# -----------------------
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
    if abs(n[0]) > abs(n[1]):
        u = normalize(np.cross(n, np.array([0,1,0])))
    else:
        u = normalize(np.cross(n, np.array([1,0,0])))

    v = np.cross(n, u)
    # Project the hitPoint onto the plane coordinate system
    # Need a reference point on the plane. You can find one by setting x,y=0 and solving for z, 
    # or since we have the plane equation aX+bY+cZ+d=0, a point on plane could be:
    # If c != 0, then a point on plane = (0,0,-d/c)
    # We'll do this once and store it in the plane object.
    point_on_plane = plane.point_on_plane
    phit = hitPoint - point_on_plane
    local_x = np.dot(phit, u)
    local_y = np.dot(phit, v)
    return local_x, local_y


def checkerboard_color(rgbColor, local_x, local_y):
    scaleParameter = 0.5
    checkerboard = 0
    x = local_x
    y = local_y

    if x < 0:
        checkerboard += math.floor((0.5 - x) / scaleParameter)
    else:
        checkerboard += math.floor(x / scaleParameter)

    if y < 0:
        checkerboard += math.floor((0.5 - y) / scaleParameter)
    else:
        checkerboard += math.floor(y / scaleParameter)

    checkerboard = (checkerboard * 0.5) - int((checkerboard * 0.5))
    checkerboard *= 2
    if checkerboard > 0.5:
        return 0.5 * rgbColor
    return rgbColor


# -----------------------
# Classes
# -----------------------

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

# -----------------------
# Scene and Rendering
# -----------------------
class Scene:
    def __init__(self):
        self.eye = np.array([0,0,4])  # default
        self.ambient = np.array([0.1,0.1,0.1])
        self.objects = []
        self.lights = []
        self.spotlights = []
        self.multi_sampling = False
        self.image_width = 500
        self.image_height = 500

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]

        # We expect a certain order: e (eye), a (ambient), then sets of objects and lights
        # "d" lines define directional/spots,
        # "p" lines define spotlights position & cutoff,
        # "i" lines define their intensities,
        # "o"/"r"/"t" define objects,
        # "c" define their colors & shininess.

        materials = []
        objects_info = []
        directions = []
        positions = []
        intensities = []
        scene_objects = []

        # We'll first just store lines, then process after.
        eye_line = None
        ambient_line = None

        # We'll need to keep track of the order we encounter objects & lights
        object_lines = []
        color_lines = []
        d_lines = []
        p_lines = []
        i_lines = []
        # reflect/transparent objects might ignore some lines, but we still read them in order.

        for line in lines:
            parts = line.split()
            c = parts[0]
            if c == 'e':
                # eye
                # e x y z mode_flag
                x,y,z,_ = map(float, parts[1:5])
                self.eye = np.array([x,y,z])
                # The 4th param can be used for multi-sampling: if >0 then enable
                if len(parts) == 5 and float(parts[4]) > 0:
                    self.multi_sampling = True
            elif c == 'a':
                # a r g b 1.0
                r,g,b,_ = map(float, parts[1:5])
                self.ambient = np.array([r,g,b])
            elif c in ['o','r','t']:
                # objects
                # spheres: x y z radius
                # planes: a b c d (d <=0)
                # The type (sphere/plane) determined by sign of 4th value:
                # if 4th >0 => sphere
                # else plane
                object_lines.append((c, parts[1:5]))
            elif c == 'c':
                # color lines: r g b shininess
                color_lines.append(parts[1:5])
            elif c == 'd':
                # direction lines: x y z w
                # w=0 directional, w=1 spotlight
                d_lines.append(parts[1:5])
            elif c == 'p':
                # position lines for spotlights: x y z cutoff
                p_lines.append(parts[1:5])
            elif c == 'i':
                # intensity lines: r g b 1.0
                i_lines.append(parts[1:5])
            else:
                # Ignore unknown lines
                pass

        # now build objects
        # According to instructions: "c" order corresponds to objects in order they appear.
        # We must match object_lines with color_lines for normal objects.
        # For reflective "r" and transparent "t", we ignore their "c" line in terms of K_A, K_D 
        # and just store their shininess from c anyway? It's not fully clear. 
        # The instructions say to ignore "c" parameters for reflection/transparent objects except shininess?
        # We'll store the shininess from the c line anyway (the order is guaranteed? 
        # We'll assume there's a one-to-one "c" for each "o"/"r"/"t" line in order.

        # Build objects
        for (obj_def, col_def) in zip(object_lines, color_lines):
            ctype, vals = obj_def
            r,g,b,sh = map(float, col_def)
            shininess = sh

            if ctype == 'o':
                # normal object
                # Determine if sphere or plane by looking at 4th param
                v = list(map(float, vals))
                if v[3] > 0:
                    # sphere
                    center = np.array(v[0:3])
                    radius = v[3]
                    mat = Material(ambient=np.array([r,g,b]), diffuse=np.array([r,g,b]), shininess=shininess)
                    self.objects.append(Sphere(center, radius, mat))
                else:
                    # plane
                    a,b,c,d = v
                    mat = Material(ambient=np.array([r,g,b]), diffuse=np.array([r,g,b]), shininess=shininess)
                    self.objects.append(Plane(a,b,c,d, mat))
            elif ctype == 'r':
                # reflective
                v = list(map(float, vals))
                if v[3] > 0:
                    # sphere
                    center = np.array(v[0:3])
                    radius = v[3]
                    mat = Material(reflective=True, shininess=shininess)
                    self.objects.append(Sphere(center, radius, mat))
                else:
                    # plane
                    a,b,c,d = v
                    mat = Material(reflective=True, shininess=shininess)
                    self.objects.append(Plane(a,b,c,d, mat))
            elif ctype == 't':
                # transparent
                v = list(map(float, vals))
                # transparent only spheres
                center = np.array(v[0:3])
                radius = v[3]
                mat = Material(transparent=True, shininess=shininess)
                self.objects.append(Sphere(center, radius, mat))

        # Build lights
        # The order of "d" lines corresponds with "p" and "i" lines.
        # If w=0 in d line => directional light
        # If w=1 in d line => spotlight (must match with p line for position & cutoff)
        # i lines correspond to each d line

        # We must carefully match them:
        # Assume: number of d lines = number of i lines
        # number of p lines = number of spotlights only
        p_spot_count = 0
        for idx, dline in enumerate(d_lines):
            x,y,z,w = map(float, dline)
            ix,iy,iz,_ = map(float, i_lines[idx])
            intensity = np.array([ix,iy,iz])
            if abs(w - 1.0) < EPSILON:
                # spotlight
                px,py,pz,cutoff = map(float, p_lines[p_spot_count])
                p_spot_count+=1
                self.lights.append(Spotlight(np.array([px,py,pz]), np.array([x,y,z]), intensity, cutoff))
            else:
                # directional
                self.lights.append(DirectionalLight(np.array([x,y,z]), intensity))

    def trace_ray(self, ray, depth=0):
        if depth > MAX_RECURSION_DEPTH:
            return np.zeros(3)

        # Find nearest intersection
        closest = Intersection()
        for obj in self.objects:
            inter = obj.intersect(ray)
            if inter.t < closest.t:
                closest = inter

        if closest.t == float('inf'):
            # no hit
            return np.zeros(3)  # background black

        obj = closest.obj
        mat = obj.get_material()
        point = closest.point
        normal = closest.normal
        view_dir = normalize(self.eye - point)

        # If reflective or transparent, ignore object's own color in direct shading:
        # As per instructions, for reflective/transparent:
        #   I = K_R * I_R (I_R from recursive ray)
        # and K_R * I_R = reflection or refraction contribution only.
        # They say "ignore material parameter (Ambient, Diffuse, Specular)" for reflective/transparent.
        # That means if reflective:
        #    reflect the ray and get color from reflection
        # If transparent:
        #    refract the ray and get color from that direction

        # For normal objects:
        # I = I_A * K_A + sum over lights [ K_D(N·L) + K_S(V·R)^n ] * S_i * I_i

        # Also handle shadows:
        # For each light, shoot a shadow ray and see if blocked.

        color = np.zeros(3)

        if mat.reflective:
            # Reflective object
            refl_dir = reflect(ray.direction, normal)
            refl_ray = Ray(point + normal * EPSILON, refl_dir)
            color = self.trace_ray(refl_ray, depth+1)
            return np.clip(color,0,1)
        elif mat.transparent:
            # Transparent object (sphere only)
            # Refractive index: outside is air=1, inside sphere=1.5
            # Determine if we are entering or exiting the sphere
            outside = np.dot(ray.direction, normal) < 0
            n1 = 1.0 if outside else 1.5
            n2 = 1.5 if outside else 1.0
            ref_normal = normal if outside else -normal
            refr_dir = refract(ray.direction, ref_normal, n1, n2)
            if refr_dir is None:
                # total internal reflection
                refl_dir = reflect(ray.direction, normal)
                refl_ray = Ray(point + normal * EPSILON, refl_dir)
                color = self.trace_ray(refl_ray, depth+1)
            else:
                # refracted ray
                # Move the ray origin slightly inside or outside
                refr_ray = Ray(point - ref_normal * EPSILON, refr_dir)
                color = self.trace_ray(refr_ray, depth+1)
            return np.clip(color,0,1)
        else:
            # Normal object, use Phong model
            # Ambient:
            Ia = self.ambient * mat.ambient
            color += Ia

            # For planes: checkerboard pattern affects the diffuse color only
            if isinstance(obj, Plane):
                local_x, local_y = get_plane_local_coords(obj, point)
                mat_diffuse = checkerboard_color(mat.diffuse, local_x, local_y)
            else:
                mat_diffuse = mat.diffuse

            for light in self.lights:
                L, distToLight = light.get_direction(point)
                if L is None:
                    continue

                # Check spotlight cutoff if spotlight
                spotlight_factor = 1.0
                if isinstance(light, Spotlight):
                    # Check if angle is within cutoff
                    # angle = dot(L, light.direction)
                    angle = np.dot(-L, light.direction)
                    if angle < light.cutoff:
                        # outside cutoff angle
                        continue
                    else:
                        spotlight_factor = angle

                # Shadow check
                # For spotlight: ensure object is not behind the spotlight
                # Shoot a shadow ray from point+normal*EPSILON to light
                shadow_orig = point + normal*EPSILON
                shadow_ray = Ray(shadow_orig, L)
                shadow_hit = False

                # If directional light, infinite distance => if any object hits
                # If spotlight, must check if intersection is before reaching the light pos
                for o in self.objects:
                    sh_inter = o.intersect(shadow_ray)
                    if sh_inter.t < float('inf'):
                        # If directional light: any hit means shadow
                        # If spotlight: check if sh_inter.t < distToLight
                        if isinstance(light, Spotlight):
                            if sh_inter.t > EPSILON and sh_inter.t < distToLight:
                                shadow_hit = True
                                break
                        else:
                            # directional
                            if sh_inter.t > EPSILON:
                                shadow_hit = True
                                break

                if shadow_hit:
                    # no contribution from this light
                    continue

                # Compute Phong terms
                N = normal
                R = reflect(-L, N)
                NdotL = max(0, np.dot(N,L))
                RdotV = max(0, np.dot(R, view_dir))

                Id = mat_diffuse * NdotL
                Is = mat.specular * (RdotV ** mat.shininess)

                # Add them up with light intensity
                light_contrib = (Id + Is) * light.intensity * spotlight_factor
                color += light_contrib

            return np.clip(color,0,1)


    def render(self, filename):
        img = np.zeros((self.image_height, self.image_width, 3), dtype=np.float32)

        # The screen is at z=0, from x=-1 to 1 and y=-1 to 1
        # pixel size:
        px_size_x = 2.0 / self.image_width
        px_size_y = 2.0 / self.image_height

        for y in range(self.image_height):
            for x in range(self.image_width):
                if self.multi_sampling:
                    # 4 sample AA
                    # jitter within pixel
                    samples = []
                    for sx in [0.25, 0.75]:
                        for sy in [0.25, 0.75]:
                            pixel_x = -1 + (x+sx)*px_size_x
                            pixel_y = -1 + (y+sy)*px_size_y
                            pixel_point = np.array([pixel_x, pixel_y, 0])
                            direction = pixel_point - self.eye
                            ray = Ray(self.eye, direction)
                            color = self.trace_ray(ray,0)
                            samples.append(color)
                    final_color = np.mean(samples, axis=0)
                else:
                    pixel_x = -1 + (x+0.5)*px_size_x
                    pixel_y = -1 + (y+0.5)*px_size_y
                    pixel_point = np.array([pixel_x, pixel_y, 0])
                    direction = pixel_point - self.eye
                    ray = Ray(self.eye, direction)
                    final_color = self.trace_ray(ray,0)

                img[self.image_height - 1 - y, x] = final_color

        # Convert to BGR and 0-255
        img = np.clip(img,0,1)
        img = (img * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, img_bgr)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ray_tracer.py scene.txt output.png")
        sys.exit(1)

    scene_file = sys.argv[1]
    output_file = sys.argv[2]

    scene = Scene()
    scene.load_from_file(scene_file)
    scene.render(output_file)
    print("Rendered image saved to", output_file)
