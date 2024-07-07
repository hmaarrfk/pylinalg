import numpy as np
from numpy.lib.stride_tricks import as_strided
try:
    from numba import jit
except ImportError:
    jit = None


def aabb_to_sphere(aabb, /, *, out=None, dtype=None):
    """A sphere that envelops an Axis-Aligned Bounding Box.

    Parameters
    ----------
    aabb : ndarray, [2, 3]
        The axis-aligned bounding box.
    out : ndarray, optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    sphere : ndarray, [4]
        A sphere (x, y, z, radius).

    """

    aabb = np.asarray(aabb, dtype=float)

    if out is None:
        out = np.empty((*aabb.shape[:-2], 4), dtype=dtype)

    out[..., :3] = np.sum(aabb, axis=-2) / 2
    out[..., 3] = np.linalg.norm(np.diff(aabb, axis=-2), axis=-1) / 2

    return out


def aabb_transform(aabb, matrix): #, *, out=None, dtype=None):
    dtype = np.float32

    compute_dtype = np.float32 # np.promote_types(aabb.dtype, dtype)
    aabb = np.ascontiguousarray(np.asarray(aabb, dtype=compute_dtype))
    matrix = np.ascontiguousarray(
        np.asarray(matrix, dtype=compute_dtype).transpose((-1, -2))
    )

    out = np.empty_like(aabb, dtype=dtype)

    corners = np.full(
        aabb.shape[:-2] + (8, 4),
        fill_value=1.,
        dtype=compute_dtype,
    )

    corners[..., 0::2, 0] = aabb[..., 0, 0]
    corners[..., 1::2, 0] = aabb[..., 1, 0]

    corners[..., 0::4, 1] = aabb[..., 0, 1]
    corners[..., 1::4, 1] = aabb[..., 0, 1]
    corners[..., 2::4, 1] = aabb[..., 1, 1]
    corners[..., 3::4, 1] = aabb[..., 1, 1]

    corners[..., 0:4, 2] = aabb[..., 0, 2]
    corners[..., 4:8, 2] = aabb[..., 1, 2]

    corners = corners @ matrix
    out[0, :] = corners[..., :-1].min(axis=-2)
    out[1, :] = corners[..., :-1].max(axis=-2)
    return out


if jit is not None:
    # A stripped down version of the function that can be compiled with Numba
    @jit(nopython=True)
    def aabb_transform(aabb, matrix):
        dtype = np.float32
        compute_dtype = np.float32 # np.promote_types(aabb.dtype, dtype)

        aabb = np.ascontiguousarray(np.asarray(aabb, dtype=compute_dtype))
        matrix = np.ascontiguousarray(
            np.asarray(matrix, dtype=compute_dtype).transpose((-1, -2))
        )

        out = np.empty_like(aabb, dtype=dtype)

        corners = np.full(
            aabb.shape[:-2] + (8, 4),
            fill_value=1.,
            dtype=compute_dtype,
        )

        corners[..., 0::2, 0] = aabb[..., 0, 0]
        corners[..., 1::2, 0] = aabb[..., 1, 0]

        corners[..., 0::4, 1] = aabb[..., 0, 1]
        corners[..., 1::4, 1] = aabb[..., 0, 1]
        corners[..., 2::4, 1] = aabb[..., 1, 1]
        corners[..., 3::4, 1] = aabb[..., 1, 1]

        corners[..., 0:4, 2] = aabb[..., 0, 2]
        corners[..., 4:8, 2] = aabb[..., 1, 2]

        corners = corners @ matrix
        # out[0, :] = corners[..., :-1].min(axis=-2)
        # out[1, :] = corners[..., :-1].max(axis=-2)
        # return out
        # Numba doesn't support the axis argument for min and max,
        # so we implement it manually
        for i in range(out.shape[-1]):
            out[0, i] = np.min(corners[..., i])
            out[1, i] = np.max(corners[..., i])

        return out


def quat_to_axis_angle(quaternion, /, *, out=None, dtype=None):
    """Convert a quaternion to axis-angle representation.

    Parameters
    ----------
    quaternion : ndarray, [4]
        A quaternion describing the rotation.
    out : Tuple[ndarray, ...], optional
        A location into which the result is stored. If provided, it
        must have a shape that the inputs broadcast to. If not provided or
        None, a freshly-allocated array is returned. A tuple must have
        length equal to the number of outputs.
    dtype : data-type, optional
        Overrides the data type of the result.

    Returns
    -------
    axis : ndarray, [3]
        The axis around which the quaternion rotates in euclidean coordinates.
    angle : ndarray, [1]
        The angle (in rad) by which the quaternion rotates.

    Notes
    -----
    To use `out` with a single quaternion you need to provide a ndarray of shape
    ``(1,)`` for angle.

    """

    quaternion = np.asarray(quaternion)

    if out is None:
        quaternion = quaternion.astype(dtype)
        out = (
            quaternion[..., :3] / np.sqrt(1 - quaternion[..., 3] ** 2),
            2 * np.arccos(quaternion[..., 3]),
        )
    else:
        out[0][:] = quaternion[..., :3] / np.sqrt(1 - quaternion[..., 3] ** 2)
        out[1][:] = 2 * np.arccos(quaternion[..., 3])

    return out


__all__ = [
    name for name in globals() if name.startswith(("vec_", "mat_", "quat_", "aabb_"))
]
