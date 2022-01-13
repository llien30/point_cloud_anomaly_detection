import random
from math import cos, pi, sin

import numpy as np


def make_sphere(N: int) -> None:
    points = []
    for i in range(N):
        theta = 2 * pi * random.random()
        phi = 2 * pi * random.random()
        x = sin(phi) * cos(theta)
        y = sin(phi) * sin(theta)
        z = cos(phi)
        points.append([x, y, z])

    points = np.array(points)
    np.save(f"sphere_{N}.npy", points)


def main() -> None:
    make_sphere(5120)


if __name__ == "__main__":
    main()
