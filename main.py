import numpy as np
import cv2

MAX_LEVEL = 5
EPSILON = 1e-6

def normalize(v):
    norm = np.linalg.norm(v)
    if norm < EPSILON:
        return v
    return v / norm

def reflect(direction, normal):
    # Reflection: r = d - 2(d·n)n
    return direction - 2 * np.dot(direction, normal) * normal

def refract(direction, normal, n1, n2):
    # Using Snell's law and standard refraction formula
    # direction, normal are normalized
    cos_i = -np.dot(normal, direction)
    eta = n1 / n2
    k = 1 - (eta**2) * (1 - cos_i**2)
    if k < 0:
        # total internal reflection
        return None
    return eta * direction + (eta * cos_i - np.sqrt(k)) * normal

class Ray:
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=np.float32)
        self.direction = normalize(np.array(direction, dtype=np.float32))

class Material:
    def __init__(self, ambient, diffuse, specular=(0.7,0.7,0.7), shininess=10, reflective=False, transparent=False):
        self.ambient = np.array(ambient, dtype=np.float32)
        self.diffuse = np.array(diffuse, dtype=np.float32)
        self.specular = np.array(specular, dtype=np.float32)
        self.shininess = shininess
        self.reflective = reflective
        self.transparent = transparent

class Intersection:
    def __init__(self, hit=False, distance=np.inf, point=None, normal=None, obj=None):
        self.hit = hit
        self.distance = distance
        self.point = point
        self.normal = normal
        self.obj = obj

class Object3D:
    def __init__(self, material):
        self.material = material

    def intersect(self, ray):
        # Returns an Intersection object
        raise NotImplementedError()

    def get_normal(self, point):
        raise NotImplementedError()

class Sphere(Object3D):
    def __init__(self, center, radius, material):
        super().__init__(material)
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius

    def intersect(self, ray):
        # (O + tD - C) · (O + tD - C) = r^2
        # Solve quadratic for t
        oc = ray.origin - self.center
        b = 2 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius**2
        discriminant = b**2 - 4*c
        if discriminant < 0:
            return Intersection()
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0

        # Choose the closest positive t
        t = None
        if t1 > EPSILON and t2 > EPSILON:
            t = min(t1,t2) if min(t1,t2) > EPSILON else max(t1,t2)
        elif t1 > EPSILON:
            t = t1
        elif t2 > EPSILON:
            t = t2
        
        if t is None:
            return Intersection()

        point = ray.origin + t*ray.direction
        normal = normalize(point - self.center)
        return Intersection(hit=True, distance=t, point=point, normal=normal, obj=self)

    def get_normal(self, point):
        return normalize(point - self.center)

class Plane(Object3D):
    def __init__(self, normal, d, material):
        super().__init__(material)
        # plane equation: a*x + b*y + c*z + d = 0
        self.normal = normalize(np.array(normal, dtype=np.float32))
        self.d = d

    def intersect(self, ray):
        denom = np.dot(self.normal, ray.direction)
        if abs(denom) < EPSILON:
            return Intersection()
        t = -(np.dot(self.normal, ray.origin) + self.d) / denom
        if t > EPSILON:
            point = ray.origin + t*ray.direction
            return Intersection(hit=True, distance=t, point=point, normal=self.normal, obj=self)
        return Intersection()

    def get_normal(self, point):
        return self.normal

class Light:
    def __init__(self, intensity):
        self.intensity = np.array(intensity, dtype=np.float32)

    def get_direction(self, hit_point):
        raise NotImplementedError()

    def get_intensity(self):
        return self.intensity

class DirectionalLight(Light):
    def __init__(self, direction, intensity):
        super().__init__(intensity)
        self.direction = normalize(np.array(direction, dtype=np.float32))

    def get_direction(self, hit_point):
        # Directional light is coming from a specific direction (like the sun)
        return -self.direction

class SpotLight(Light):
    def __init__(self, position, direction, cutoff_cos, intensity):
        super().__init__(intensity)
        self.position = np.array(position, dtype=np.float32)
        self.direction = normalize(np.array(direction, dtype=np.float32))
        self.cutoff_cos = cutoff_cos

    def get_direction(self, hit_point):
        return normalize(self.position - hit_point)

    def within_cutoff(self, hit_point):
        dir_to_point = normalize(hit_point - self.position)
        angle_cos = np.dot(dir_to_point, -self.direction)
        return angle_cos > self.cutoff_cos

class Scene:
    def __init__(self, objects=[], lights=[], ambient=(0,0,0), background=(0,0,0)):
        self.objects = objects
        self.lights = lights
        self.ambient = np.array(ambient, dtype=np.float32)
        self.background = np.array(background, dtype=np.float32)

    def find_intersection(self, ray):
        closest = Intersection()
        for obj in self.objects:
            inter = obj.intersect(ray)
            if inter.hit and inter.distance < closest.distance:
                closest = inter
        return closest

def calc_color(scene, in_ray, level=0):
    if level > MAX_LEVEL:
        return np.array([0,0,0], dtype=np.float32)

    hit = scene.find_intersection(in_ray)
    if not hit.hit:
        # background
        return scene.background.copy()

    material = hit.obj.material
    color = np.zeros(3, dtype=np.float32)
    
    # Emission assumed (0,0,0)
    # Ambient
    color += material.ambient * scene.ambient

    normal = hit.normal
    view_dir = -in_ray.direction

    for light in scene.lights:
        light_dir = light.get_direction(hit.point)
        in_light = True

        # Check spotlight cutoff if needed
        if isinstance(light, SpotLight):
            if not light.within_cutoff(hit.point):
                in_light = False

        # Shadow calculation
        if in_light:
            shadow_ray = Ray(hit.point + normal*EPSILON, light_dir)
            shadow_inter = scene.find_intersection(shadow_ray)
            if isinstance(light, SpotLight):
                dist_to_light = np.linalg.norm(light.position - hit.point)
                if shadow_inter.hit and shadow_inter.distance < dist_to_light - EPSILON:
                    in_light = False
            else:
                # directional
                if shadow_inter.hit:
                    in_light = False
        
        if in_light:
            # Diffuse
            diffuse_factor = np.dot(normal, light_dir)
            if diffuse_factor > 0:
                color += material.diffuse * light.get_intensity() * diffuse_factor
            # Specular
            reflect_dir = reflect(-light_dir, normal)
            spec_factor = np.dot(view_dir, reflect_dir)
            if spec_factor > 0:
                color += material.specular * light.get_intensity() * (spec_factor ** material.shininess)

    # Reflection / Refraction
    if level < MAX_LEVEL:
        if material.reflective:
            reflect_dir = reflect(in_ray.direction, normal)
            reflect_ray = Ray(hit.point + normal*EPSILON, reflect_dir)
            reflected_color = calc_color(scene, reflect_ray, level+1)
            color += reflected_color
        
        if material.transparent:
            n1 = 1.0
            n2 = 1.5
            n = normal.copy()
            if np.dot(in_ray.direction, n) > 0:
                # inside going out
                n1, n2 = n2, n1
                n = -n
            refract_dir = refract(in_ray.direction, n, n1, n2)
            if refract_dir is None:
                # total internal reflection
                reflect_dir = reflect(in_ray.direction, n)
                reflect_ray = Ray(hit.point + n*EPSILON, reflect_dir)
                color += calc_color(scene, reflect_ray, level+1)
            else:
                refract_ray = Ray(hit.point - n*EPSILON, refract_dir)
                refracted_color = calc_color(scene, refract_ray, level+1)
                color += refracted_color

    return np.clip(color,0,1)

def render(scene, eye, width=800, height=800, use_antialias=False):
    img = np.zeros((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            px = (2.0 * x / (width - 1)) - 1.0
            py = (2.0 * y / (height - 1)) - 1.0

            if use_antialias:
                # Example: 4 samples
                samples = []
                offsets = [(-0.25, -0.25), (-0.25, 0.25), (0.25, -0.25), (0.25, 0.25)]
                for ox, oy in offsets:
                    pxx = px + (ox/(width))
                    pyy = py + (oy/(height))
                    pixel_pos = np.array([pxx, pyy, 0], dtype=np.float32)
                    ray_dir = pixel_pos - eye
                    ray = Ray(eye, ray_dir)
                    samples.append(calc_color(scene, ray, 0))
                color = np.mean(samples, axis=0)
            else:
                pixel_pos = np.array([px, py, 0], dtype=np.float32)
                ray_dir = pixel_pos - eye
                ray = Ray(eye, ray_dir)
                color = calc_color(scene, ray, 0)

            img[height - 1 - y, x, :] = color

    img_bgr = (img*255).astype(np.uint8)
    return img_bgr

def load_scene(filename):
    eye = np.array([0,0,4], dtype=np.float32)
    use_antialias = False
    ambient = np.zeros(3, dtype=np.float32)
    background = np.array([0,0,0], dtype=np.float32)

    objects = []
    object_types = []  # store 'o', 'r', or 't' to know object's nature
    raw_obj_params = []  # store the parameters from file: (x,y,z,w)
    materials_data = []  # will be filled after we read "c" lines

    light_dirs = []
    spotlight_params = []  # for storing (position, cutoff) corresponding to spotlight directions
    light_intensities = []

    # We'll process in multiple passes or store data and link later
    # According to instructions: 
    #   "e" - sets eye pos (4th param may be for anti-alias mode)
    #   "a" - sets ambient
    #   "o/r/t x y z w" - objects
    #   "c r g b shininess" - materials for objects in order
    #   "d x y z w" - light directions (w=0 directional, w=1 spotlight)
    #   "p x y z cutoff" - spotlight position & cutoff
    #   "i r g b 1" - light intensity

    with open(filename, 'r') as f:
        lines = f.readlines()

    # First pass: read everything
    # We'll store lines in temporary lists to handle order
    object_lines = []
    material_lines = []
    d_lines = []
    p_lines = []
    i_lines = []

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        key = parts[0]

        if key == 'e':
            # e x y z w
            eye = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
            # If w != 1.0, might indicate something about anti-aliasing mode
            if float(parts[4]) != 1.0:
                use_antialias = True
        elif key == 'a':
            # a r g b 1.0
            ambient = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
            # 4th param (1.0) can be ignored as per instructions
        elif key in ('o', 'r', 't'):
            # object line
            object_lines.append(parts)
        elif key == 'c':
            # material line
            material_lines.append(parts)
        elif key == 'd':
            # direction line
            d_lines.append(parts)
        elif key == 'p':
            # spotlight position line
            p_lines.append(parts)
        elif key == 'i':
            # intensity line
            i_lines.append(parts)

    # Process objects
    # Each object line: "o/r/t x y z w"
    # If w>0 => sphere with radius w
    # If w<=0 => plane with equation a*x+b*y+c*z+d=0
    # The sign of w determines sphere/plane.
    # For reflective/transparent objects, ignore the c (color) at this stage, they get reflection/refraction only
    for line in object_lines:
        key = line[0]
        x, y, z, w_param = float(line[1]), float(line[2]), float(line[3]), float(line[4])
        reflective = (key == 'r')
        transparent = (key == 't')
        
        # Create a default material now; we will assign the real material after parsing "c" lines.
        # Just a dummy material to hold a place
        default_material = Material(ambient=(0,0,0), diffuse=(0,0,0), shininess=1, reflective=reflective, transparent=transparent)
        
        if w_param > 0:
            # Sphere
            radius = w_param
            obj = Sphere(center=(x,y,z), radius=radius, material=default_material)
        else:
            # Plane
            # equation: a*x + b*y + c*z + d=0 with (a,b,c) = normal *some factor, d = w_param
            # from instructions: "o/r/t x y z w" for plane means (a,b,c,d) = (x,y,z,w)
            normal = (x,y,z)
            d_val = w_param
            obj = Plane(normal=normal, d=d_val, material=default_material)
        
        objects.append(obj)

    # Assign materials to objects in order
    # Each "c" line corresponds to the next object in order
    # c r g b n
    for i, line in enumerate(material_lines):
        r, g, b, n = float(line[1]), float(line[2]), float(line[3]), float(line[4])
        # Objects were created in order, so object i gets this material (unless it's reflective/transparent)
        obj = objects[i]
        # If obj is reflective or transparent, ignore ambient/diffuse and just do reflection/refraction,
        # but from the instructions, we still set Ka, Kd for completeness. 
        # Also note Ks=0.7 fixed as per instructions
        obj.material.ambient = np.array([r,g,b], dtype=np.float32)
        obj.material.diffuse = np.array([r,g,b], dtype=np.float32)
        obj.material.specular = np.array([0.7,0.7,0.7], dtype=np.float32)
        obj.material.shininess = n

    # Process lights
    # For each d line: d x y z w
    #   if w=0 directional light; if w=1 spotlight
    # For each spotlight, we have a corresponding p line: p x y z cutoff
    # For each light, we have an i line: i r g b 1.0
    # The order of i lines corresponds to the order of d lines
    lights = []
    spotlight_index = 0
    p_line_index = 0
    for idx, dline in enumerate(d_lines):
        dx, dy, dz, dw = float(dline[1]), float(dline[2]), float(dline[3]), float(dline[4])
        # Intensity for this light:
        iline = i_lines[idx]
        ir, ig, ib = float(iline[1]), float(iline[2]), float(iline[3])

        intensity = (ir, ig, ib)
        if dw == 0.0:
            # Directional light
            light = DirectionalLight(direction=(dx,dy,dz), intensity=intensity)
            lights.append(light)
        else:
            # Spotlight
            pline = p_lines[p_line_index]
            p_line_index += 1
            px, py, pz, cutoff = float(pline[1]), float(pline[2]), float(pline[3]), float(pline[4])
            light = SpotLight(position=(px,py,pz), direction=(dx,dy,dz), cutoff_cos=cutoff, intensity=intensity)
            lights.append(light)

    # Build the scene
    scene = Scene(objects=objects, lights=lights, ambient=ambient, background=background)

    return scene, eye, use_antialias

if __name__ == "__main__":
    # Example usage with the given example txt file:
    scene, eye, use_antialias = load_scene("./res/scene1.txt")
    img = render(scene, eye, width=400, height=400, use_antialias=use_antialias)
    cv2.imwrite("rendered_scene.png", img)
