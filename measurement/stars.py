from ctapipe.utils import get_bright_stars
from astropy.time import Time
from astropy.coordinates import AltAz, SkyCoord
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from measurement import objects as obj

t = Time("J2024")
data = get_bright_stars(t)
bright_stars = [d for d in data if 83*2.51**(-d['Vmag']-4) >= 0.5]

STARS = []
for bright_star in bright_stars:
    cart = bright_star['ra_dec'].cartesian
    vec = np.array([cart.x, cart.y, cart.z])
    uvec = vec / np.linalg.norm(vec)
    STARS.append(obj.Star(uvec, bright_star['Vmag']))

if __name__ == "__main__":
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    stars_x = [star.pos[0] for star in STARS]
    stars_y = [star.pos[1] for star in STARS]
    stars_z = [star.pos[2] for star in STARS]
    ax.scatter(stars_x, stars_y, stars_z)
    plt.show()
