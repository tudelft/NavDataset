import os
from dataclasses import dataclass, field
from copy import deepcopy
import datetime

import numpy as np
import h5py
from tqdm import tqdm

import utils.routines as rout


@dataclass
class EventData:
    rosTime: list = field(default_factory=list)  # ROS time
    x: np.ndarray = np.array(
        [], dtype=np.int16
    )  # x position of pixel along image array
    y: np.ndarray = np.array(
        [], dtype=np.int16
    )  # y position of pixel along image array
    p: np.ndarray = np.array([], dtype=np.bool_)  # polarity of event
    t: np.ndarray = np.array([], dtype=np.float64)  # time of event
    units: dict = field(default_factory=dict)
    dim: tuple = (240, 180)

    def __post_init__(self):
        self.units = {
            "ns": -9,
            "mus": -6,
            "ms": -3,
            "s": 0,
            "custom": 0,
        }

    def __str__(self):
        dic = {
            "time [s]": self.t,
            "x [-]": self.x,
            "y [-]": self.y,
            "polarity [-]": self.p,
        }
        return str(dic)

    def to_h5(self, out_path, meta_data={}):
        with h5py.File(out_path, "w") as f:
            f.create_dataset("events/xs", data=self.x)
            f.create_dataset("events/ys", data=self.y)
            f.create_dataset("events/ts", data=self.t)
            f.create_dataset("events/ps", data=self.p)
            for key, val in meta_data.items():
                f["events"].attrs[key] = val
        return

    def to_txt(self, out_path, header: str = "time [s], x [-], y [-], polarity [-]"):
        ev_mat = np.hstack(
            (
                self.t.reshape(-1, 1),
                self.x.reshape(-1, 1),
                self.y.reshape(-1, 1),
                self.p.reshape(-1, 1),
            )
        )
        with open(out_path, "w") as f:
            np.savetxt(
                f,
                ev_mat,
                fmt="%.9f,%d,%d,%d",
                delimiter=",",
                header=header,
                comments="",
            )
        return

    def remove_flashes(self, thresh=10):
        # scale to the required timescale
        t = self.to_timeunit(sign_dig=5).t
        filter_array = np.ones_like(self.p, dtype=np.bool_)

        # filter out flashes
        u, n = np.unique(t, return_counts=True)
        filter_array[np.isin(t, u[n > thresh])] = False

        # discard noise
        self.t = self.t[filter_array]
        self.x = self.x[filter_array]
        self.y = self.y[filter_array]
        self.p = self.p[filter_array]

        return

    def filter_noise(self, dim, cutoff=5):
        # scale to the required timescale
        t = self.to_timeunit(unit="ms").t

        # filter noise
        filter_array = rout.filter_noise_numba(
            t, self.x, self.y, self.p, dim, cutoff=cutoff
        )

        # discard noise
        self.t = self.t[filter_array]
        self.x = self.x[filter_array]
        self.y = self.y[filter_array]
        self.p = self.p[filter_array]

        return

    def to_timeunit(self, unit="s", exp=0, sign_dig=0):
        # change custom scale
        if unit == "custom":
            self.units["custom"] = exp

        if unit not in self.units.keys():
            raise KeyError(
                f"{unit} is not a valid unit, valid units: {self.unit.keys()}"
            )
        else:
            # scale to desired timescale
            scale = 10 ** -self.units.get(unit)

        evdata = EventData(
            x=self.x,
            y=self.y,
            p=self.p,
            t=deepcopy(self.t),
            rosTime=self.rosTime,
            dim=self.dim,
        )
        evdata.t = np.round(scale * evdata.t, sign_dig)

        return evdata

    def to_timeunit_(self, unit="s", exp=0, sign_dig=0):
        # change custom scale
        if unit == "custom":
            self.units["custom"] = exp

        if unit not in self.units.keys():
            raise KeyError(
                f"{unit} is not a valid unit, valid units: {self.unit.keys()}"
            )
        else:
            # scale to desired timescale
            scale = 10 ** -self.units.get(unit)

        self.t = np.round(scale * self.t, sign_dig)

        return self

    def _reset_t0(self, reset_time=None, verbose=False):
        if verbose:
            print("Fixing the timestamps of events ...")
        cut = None
        if (self.t[-1] - self.t[0]) > 10 ** 7:
            # clean up time encoding (first few entries are equal to 0 sec)
            cut = np.where(np.diff(self.t) > 1)[0][0]

        # initialize time to zero
        if cut is not None:
            self.x = self.x[cut + 1 :]
            self.y = self.y[cut + 1 :]
            self.p = self.p[cut + 1 :]
            self.t = self.t[cut + 1 :]
            if reset_time is None:
                self.t -= self.t[0]
            else:
                self.t -= reset_time

        else:
            if reset_time is None:
                self.t -= self.t[0]
            else:
                self.t -= reset_time

        return self

    def _odd_even_fix(self, verbose=False):
        if verbose:
            print("Fixing odd/even address mismatch")
        # count number of unique events at each x-address
        x_addr, n = np.unique(self.x, return_counts=True)

        # loop through events by their x-event address
        for k, i in tqdm(
            enumerate(x_addr),
            desc="Fixing odd/even address mismatch",
            total=len(x_addr),
        ):
            # odd events (1, 3, ...)
            if i % 2 == 1:
                # find drop percentage to equalize it to the previous address
                p_drop = n[k - 1] / n[k]

                # select events at event address i
                valid = np.where(self.x == i)[0]
                # drop events as determined by p_drop
                #   multiply index j+1 (position in sequence of events with address i)
                #   with p_drop. If this changes by 1, an event is tagged as 'allowed'
                #   and stored.
                select = np.fromiter(
                    (
                        x
                        for j, x in enumerate(valid)
                        if round((j + 1) * p_drop) - round(j * p_drop)
                    ),
                    dtype=int,
                )

                # determine which entries should be dropped
                discard = np.setdiff1d(valid, select)
                # tag data that should be dropped
                self.t[discard] = -1

            # even events (0, 2, ...)
            else:
                pass

        # select data > -1 to obtain address-corrected dvs data
        # "t" corrected last
        for key in ["x", "y", "pol", "t"]:
            self.__dict__[key] = self.__dict__[key][self.t > -1]

        return


@dataclass
class GpsMsg:
    t: np.ndarray = np.array([])
    lat: np.ndarray = np.array([])
    lon: np.ndarray = np.array([])
    heading: np.ndarray = np.array([])
    gspeed: np.ndarray = np.array([])

    def __str__(self) -> str:
        dic = {
            "time [s]": self.t,
            "lat [deg]": self.lat,
            "lon [deg]": self.lon,
            "heading [deg]": self.heading,
            "ground speed [m/s]": self.gspeed,
        }
        return str(dic)


@dataclass
class NavPVT(GpsMsg):
    # https://docs.ros.org/en/kinetic/api/ublox_msgs/html/msg/NavPVT.html
    # inherited
    # t: np.ndarray
    # lat: np.ndarray
    # lon: np.ndarray
    # heading: np.ndarray
    # gspeed: np.ndarray

    iTOW: np.ndarray = np.array([])
    year: np.ndarray = np.array([])
    month: np.ndarray = np.array([])
    day: np.ndarray = np.array([])
    hour: np.ndarray = np.array([])
    minute: np.ndarray = np.array([])
    sec: np.ndarray = np.array([])
    valid: np.ndarray = np.array([])
    tAcc: np.ndarray = np.array([])
    nano: np.ndarray = np.array([])
    fixType: np.ndarray = np.array([])
    flags: np.ndarray = np.array([])
    flags2: np.ndarray = np.array([])
    numSV: np.ndarray = np.array([])
    height: np.ndarray = np.array([])
    hMSL: np.ndarray = np.array([])
    hAcc: np.ndarray = np.array([])
    vAcc: np.ndarray = np.array([])
    velN: np.ndarray = np.array([])
    velE: np.ndarray = np.array([])
    velD: np.ndarray = np.array([])
    sAcc: np.ndarray = np.array([])
    headAcc: np.ndarray = np.array([])
    pDOP: np.ndarray = np.array([])
    reserved1: np.ndarray = np.array([])
    headVeh: np.ndarray = np.array([])
    magDec: np.ndarray = np.array([])
    magAcc: np.ndarray = np.array([])

    def to_csv(self, track_name=None, time_slice=None, save_dir=None):
        # format
        # trackpoint, time, latitude, longitude, alt, speed, course, name, color

        if time_slice is not None:
            _navPVT = np.where((self.t >= time_slice[0]) & (self.t <= time_slice[1]))
        else:
            _navPVT = self.navPVT
        if track_name is None:
            track_name = f"track {_navPVT.day[0]}/{_navPVT.month[0]}/{_navPVT.year[0]} {str(_navPVT.hour[0]).zfill(2)}:{str(_navPVT.minute[0]).zfill(2)}"

        to_timestr = lambda hms_tuple: str(datetime.time(*hms_tuple))

        t = np.array(
            list(map(to_timestr, zip(_navPVT.hour, _navPVT.minute, _navPVT.sec)))
        )

        _navPVT.t = t
        _navPVT.track = np.array(_navPVT.shape[0] * ["T"])
        _navPVT.filler = np.array(_navPVT.shape[0] * [""])

        # waypoint
        middle = _navPVT.shape[0] // 2
        wp = f"W,,{_navPVT.lat[middle]},{_navPVT.lon[middle]},{_navPVT.height[middle]},,,{track_name},Waypoint\n"

        header = [
            "type",
            "time",
            "latitude",
            "longitude",
            "alt",
            "speed",
            "course",
            "name",
            "desc",
        ]

        GNSS_track = _navPVT[
            [
                "track",
                "time",
                "lat",
                "lon",
                "height",
                "gSpeed",
                "heading",
                "filler",
                "filler",
            ]
        ]
        GNSS_track.columns = header

        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "gps.csv")

        GNSS_track.to_csv(save_dir, index=False, index_label=False)
        with open(save_dir, "w") as f:
            f.write(wp)

        return
