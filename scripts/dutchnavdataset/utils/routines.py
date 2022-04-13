import numpy as np
from numba import njit


# =====================================================================================
# String routines
# =====================================================================================
@njit
def strtobool(s: str) -> bool:
    if s == "True":
        return True
    else:
        return False

# =============================================================================
# Numpy routines
# =============================================================================
def extract_windows_numpy(array, sub_window_size):
    start = 0

    max_time = len(array)

    sub_windows = (
        start
        # expand_dims are used to convert a 1D array to 2D array.
        + np.expand_dims(np.arange(sub_window_size), 0)
        + np.expand_dims(np.arange(max_time + 2), 0).T
    )

    array = np.pad(array, (sub_window_size, sub_window_size), "edge")
    return array[sub_windows[sub_window_size:-1]]  # return viable windows


def array_slice(array, llim, rlim):
    return np.where((array >= llim) & (array <= rlim))

# also works for pytorch tensors and lists
def normalize(array: list or np.ndarray, a=None, b=None):
    if isinstance(array, list):
        array = np.array(array)
    if a is None:
        a = array.min()
    if b is None:
        b = array.max()
    return (array - a) / (b - a)


def find_nearest(a, a0):
    "Index of element in nd array `a` closest to the scalar value `a0`"
    return np.abs(a - a0).argmin()


def frames_to_events_representation(frames, timestamps):
    # on events
    ind_on, x_on, y_on = np.where(frames == 1)
    p_on = np.ones_like(ind_on, dtype=np.bool_)
    t_on = np.ones_like(ind_on, dtype=np.float64)
    for i, ind in enumerate(ind_on):
        t_on[i] = timestamps[ind]  # set timestamps
    # off events
    ind_off, x_off, y_off = np.where(frames == -1)
    p_off = np.zeros_like(ind_off, dtype=np.bool_)
    t_off = np.ones_like(ind_off, dtype=np.float64)
    for i, ind in enumerate(ind_off):
        t_off[i] = timestamps[ind]  # set timestamps

    # stack data
    ind = np.hstack((ind_on, ind_off))
    t = np.hstack((t_on, t_off))
    x = np.hstack((x_on, x_off))
    y = np.hstack((y_on, y_off))
    p = np.hstack((p_on, p_off))

    # sort events in chronological order
    sort_indices = np.argsort(ind, kind="mergesort")
    t = t[sort_indices]
    x = x[sort_indices]
    y = y[sort_indices]
    p = p[sort_indices]

    return t, x, y, p


def events_to_frames_representation(t, x, y, p, window_dt, dim=None):
    # determine/get input dimension
    if dim is None:
        dim = (x.max() + 1, y.max() + 1)

    # initiate arrays
    t_tot = t.max() - t.min()
    frames = np.zeros((int(np.ceil(t_tot / window_dt)), *dim), dtype=np.int8)
    timestamps = np.zeros((int(np.ceil(t_tot / window_dt))), dtype=np.float64)

    # polarity
    p = np.where(p, 1, -1)
    # assign events to frames
    for i, time in enumerate(np.arange(t.min(), t.max(), window_dt)):
        spike_indices = np.where((t >= time) & (t < time + window_dt))
        frames[i, x[spike_indices], y[spike_indices]] = p[spike_indices]
        timestamps[i] = time

    return frames, timestamps


def eventImage(event_data, window=None):
    img = np.zeros(event_data.dim[::-1])
    if window is not None:
        x, y, p = event_data.x[window], event_data.y[window], event_data.p[window]
    else:
        x, y, p = event_data.x, event_data.y, event_data.p[window]
    x -= 1
    y -= 1
    p = np.where(p, 1, -1)
    uv = np.array([y, x]).T  # image vs np array axis convention

    addr = np.unique(
        uv,
        axis=0,
    )

    for xy in addr:
        img[tuple(xy)] = p[(uv == xy).all(1)].sum()  # sum the polarities

    # gray background, scale for max contrast
    img = (127 + img * (127 / max(abs(img.max()), abs(img.min())))).astype(np.uint8)

    return img


# =====================================================================================
# Event noise filtering
# =====================================================================================
# Delbruck, T. (2008). Frame-free dynamic digital vision. Intl. Symp. on Secure-Life Electronics, Advanced Electronics for Quality Life and Society, 21–26.
@njit
def filter_noise_numba(tms, x, y, p, dim, cutoff):
    B = np.ones((dim[0] + 2, dim[1] + 2), dtype=np.float64)
    filter_array = np.ones_like(p).astype(np.bool_)

    for i, (tw, xw, yw) in enumerate(zip(tms, x, y)):
        # compare event timestamp with last time it was written
        t0 = B[xw + 1, yw + 1]
        if (tw - t0) > cutoff:
            filter_array[i] = False

        # update direct neighborhood of pixel in buffer B
        B[xw : xw + 3, yw : yw + 3] = tw

    return filter_array


# =====================================================================================
# GPS routines
# =====================================================================================
@njit
def dist_Haversine(latitude0, longitude0, latitude1, longitude1):
    lat0 = latitude0 * np.pi / 180
    lat1 = latitude1 * np.pi / 180
    lon0 = longitude0 * np.pi / 180
    lon1 = longitude1 * np.pi / 180

    dlon = lon1 - lon0
    dlat = lat1 - lat0

    # Haversine formula:
    R = 6371000  # earth radius in m
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c

    return d


@njit
def time_min_dist(lat0, lon0, t, lat, lon):
    return t[dist_Haversine(lat0, lon0, lat, lon).argmin()]
