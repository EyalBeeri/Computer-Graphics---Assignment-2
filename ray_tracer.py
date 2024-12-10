import numpy as np
import cv2
from ray_tracer_classes import *

class Scene:
    def __init__(self):
        self.eye = []
        self.ambient = []
        self.objects = []
        self.lights = []
        self.spotlights = []
        self.multi_sampling = False
        self.image_width = 400
        self.image_height = 400

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
        if np.dot(normal, view_dir) < 0:
            normal = -normal

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


scene_num = 1
scene_file = f"./res/scene{scene_num}.txt"
output_file = f"./scene{scene_num}.png"

scene = Scene()
scene.load_from_file(scene_file)
scene.render(output_file)
print("Rendered image saved to", output_file)
