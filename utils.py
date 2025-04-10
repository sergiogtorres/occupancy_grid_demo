import numpy as np


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def angles_in_range(angles, center, delta):
    angles = wrap_to_pi(angles)
    lower = wrap_to_pi(center - delta)
    upper = wrap_to_pi(center + delta)

    if lower <= upper:
        return (angles >= lower) & (angles <= upper)
    else:
        # Wraparound case
        return (angles >= lower) | (angles <= upper)