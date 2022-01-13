import numpy as np


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point)
    """
    return ((x - y) ** 2).sum(axis=1)


def fartherst_point_sampling(
    points: np.ndarray,
    num_sample_point: int,
    initial_idx=None,
    metrics=l2_norm,
    indices_dtype=np.int32,
    distances_dtype=np.float32,
) -> np.ndarray:
    assert points.ndim == 2, "input points shoud be 2-dim array (n_points, coord_dim)"

    num_point, coord_dim = points.shape
    indices = np.zeros((num_sample_point,), dtype=indices_dtype)
    distances = np.zeros((num_sample_point, num_point), dtype=distances_dtype)

    if initial_idx is None:
        indices[0] = np.random.randint(len(points))
    else:
        indices[0] = initial_idx

    farthest_point = points[indices[0]]

    min_distances = metrics(farthest_point[None, :], points)
    distances[0, :] = min_distances
    for i in range(1, num_sample_point):
        indices[i] = np.argmax(min_distances, axis=0)
        farthest_point = points[indices[i]]
        dist = metrics(farthest_point[None, :], points)
        distances[i, :] = dist
        min_distances = np.minimum(min_distances, dist)
    return indices
