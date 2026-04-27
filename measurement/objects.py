import numpy as np

class ObsPt:
    def __init__(self, pos):
        self.pos = pos

    def brightness(self, dist, FOV_angle):
        return 255

class Sat(ObsPt):
    def __init__(self, pos, area, reflectivity):
        super().__init__(pos)
        self.area = area
        self.reflectivity = reflectivity
    
    def update_pos(self, pos):
        self.pos = pos
    
    def brightness(self, dist, FOV_angle, rng):
        area_covered = min(self.area / (dist * np.tan(FOV_angle))**2, 1) 
        noise = rng.normal(0, 1)
        return np.round(np.clip(area_covered * self.reflectivity * 255 + noise, 0, 255))

class Star(ObsPt):
    def __init__(self, direction, magnitude):
        # Project star direction from Earth onto sphere interesecting Alpha Centauri centered on Earth
        alpha_centauri_dist = 4.15e16
        pos = direction * alpha_centauri_dist
        super().__init__(pos)
        self.pixel_brightness = int(round(83 * 2.51**(-magnitude-4)))

    def brightness(self, dist, FOV_angle, rng):
        noise = rng.normal(0, 1)
        return np.round(np.clip(self.pixel_brightness + noise, 0, 255))