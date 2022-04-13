# builtin libs
import os

# data processing
import numpy as np

# file formats
import h5py

# progress bar
from tqdm import tqdm

# image processing
import cv2
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny

# visualization
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

# local libraries
from data_types.data_types import EventData
from utils.routines import (
    normalize,
    time_min_dist,
    array_slice,
)


class DvsDataHandler:
    def __init__(self, verbose=False):
        """Supported formats:   - .bag
                                - .aedat4

        Args:
            dvs_data_dir (str): Directory to dvs data.
        """
        self.data_dir = ""

        # storing dvs event data
        self.event_data = EventData()
        # parameters for visualizing
        self.res = self.event_data.dim
        self.masked = False
        self.unwrapped = False
        self.cxy = (
            self.event_data.dim[0] // 2,
            self.event_data.dim[1] // 2,
        )  # center of image x&y
        self.r_in = 11
        self.r_out = int(
            np.sqrt(np.power(self.cxy, 2).sum())
        )  # distance to furthest point from cxy
        self.angle_res = 1
        self.verbose = verbose
        return

    def __str__(self):
        return f"{self.event_data}"

    # inspired by https://github.com/TimoStoff/events_contrast_maximization/blob/master/tools/read_events.py
    def read_h5(self, hdf_path: str = "", hdf5_file=None):
        assert (
            os.path.exists(hdf_path) or hdf5_file is not None
        ), "Either give a valid path to a .h5 file (given: {}), or an open h5 file (given: {})".format(
            hdf_path, hdf5_file
        )
        if hdf5_file is not None:
            f = hdf5_file
        else:
            f = h5py.File(hdf_path, "r")
            self.data_dir = hdf_path

        self.event_data.x = f["events/xs"][:]
        self.event_data.y = f["events/ys"][:]
        self.event_data.p = f["events/ps"][:]
        self.event_data.t = f["events/ts"][:]
        self.data_dir = f["events"].attrs["from_file"]
        self.res = f["events"].attrs["resolution"]
        self.event_data.dim = self.res
        self.masked = f["events"].attrs["masked"]
        self.unwrapped = f["events"].attrs["unwrapped"]

        if hdf5_file is None:
            f.close()
        return self

    def mask_events(self, demo=False, cxy=None, r_in=None, r_out=None):
        if self.verbose is True:
            print("Masking events ...")

        if cxy is None or r_out is None or demo is True:
            img = np.zeros(self.event_data.dim)
            # Mask center and outer region as this contains no useful omnidirectional data
            # detect image edges using canny edge detection
            # accumulate events into image
            # count number of unique events at each x-address
            print("Accumulating events ...")
            addr, n = np.unique(
                np.array([self.event_data.x, self.event_data.y]).T,
                axis=0,
                return_counts=True,
            )
            print("Done ...")
            for xy, count in zip(addr, n):
                img[tuple(xy)] = count
            img = (255 * normalize(img)).astype(np.uint8)

        if cxy is None or r_out is None:
            edges = canny(img, sigma=3, low_threshold=5, high_threshold=50)
            plt.imshow(edges)
            plt.plot()
            # Detect radii
            hough_radii = np.arange(50, 60, step=1)
            hough_res = hough_circle(edges, hough_radii)

            # Select the most prominent circle
            _, cx, cy, radius = hough_circle_peaks(
                hough_res, hough_radii, total_num_peaks=1
            )
            if self.verbose:
                print(
                    f"Determined frame center: ({cx[0]},{cy[0]}), inner radius: {self.r_in}, outer radius: {radius[0]}"
                )

            self.cxy, self.r_out = (cx[0], cy[0]), int(round(radius[0]))
        else:
            self.cxy = cxy
            self.r_out = r_out

        # construct mask containing solely data inside outer radius
        self.mask = np.zeros(self.event_data.dim, dtype=np.uint8)
        cv2.circle(
            self.mask,
            self.cxy[::-1],
            self.r_out,
            255,
            thickness=-1,
        )

        if r_in is not None:
            self.r_in = r_in

        # construct mask leaving out inner circle
        cv2.circle(self.mask, self.cxy[::-1], self.r_in, 0, thickness=-1)
        self.mask //= 255

        if demo is True:
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            ax1.imshow(img, cmap=plt.cm.gray)
            circle_out = plt.Circle(cxy[::-1], self.r_out, fill=False, color="r")
            circle_in = plt.Circle(cxy[::-1], self.r_in, fill=False, color="r")
            ax1.add_artist(circle_out)
            ax1.add_artist(circle_in)
            fig2, ax2 = plt.subplots(figsize=(3, 3))
            ax2.imshow(self.mask * img, cmap=plt.cm.gray)
            return fig1, fig2

        if demo is False:
            # mask the events
            select = np.where(self.mask[self.event_data.x, self.event_data.y] == 1)

            self.event_data.p = self.event_data.p[select]
            self.event_data.t = self.event_data.t[select]
            self.event_data.x = self.event_data.x[select]
            self.event_data.y = self.event_data.y[select]

            self.masked = True

            return self

        return

    def unwrap_events(self, prec=1, angle_res=None):
        if angle_res is not None:
            self.angle_res = angle_res
        if self.verbose:
            print("Unwrapping events ...")

        # Unwrap the events
        offset = np.array(self.cxy)  # offset of image centre

        # unwrap from origin of mask
        u = np.array([self.event_data.x, self.event_data.y]).T - offset
        (r, th) = cv2.cartToPolar(*u.T.astype(np.float), angleInDegrees=True)
        th = th.round().astype(int).ravel() % 360 // self.angle_res
        r = (r * prec).round().astype(int).ravel()

        # select events within bounds
        select = (r > self.r_in * prec) & (r < self.r_out * prec)
        # 'top' of omni vision band is furthest away
        # to convert to camera coordinates one has
        # to take the inverse as r is measured from the center of
        # the image
        r = (self.r_out - self.r_in) * prec - (r - self.r_in * prec)
        # shift 90 deg to the left such that the forward motion aligns
        # with the center of unwrapped view (-180 -90 0 90 180)
        th = (th - 90) % (360 // self.angle_res)

        self.event_data.x = th[select]  # store angle as x-position
        self.event_data.y = r[
            select
        ]  # store inverse distance to center (radius) as y-position
        self.event_data.t = self.event_data.t[select]
        self.event_data.p = self.event_data.p[select]

        # 3. Set boundaries of the output
        self.event_data.dim = (360 // self.angle_res, (self.r_out - self.r_in) * prec)
        self.res = self.event_data.dim
        self.unwrapped = True

        return self

    def noise_filter_events(self, thresh=10, cutoff=5):
        if self.verbose:
            print("Removing flashes ...")
        self.event_data.remove_flashes(thresh=thresh)
        if self.verbose:
            print("Removing noise ...")
        self.event_data.filter_noise(self.event_data.dim, cutoff)

        return

    def visualize_events(
        self,
        out_path=None,
        txt_path=None,
        h5_path=None,
        framerate=29.97,
        visualize=False,
        save=True,
        encoder="hevc_nvenc",
        res=None,
    ):
        # 1. Configure visualization depending on data
        if res is not None:
            self.res = res

        fig, ax = self._configure_dvsvisual(
            res=self.res,
        )
        window_dt = 1000 / framerate  # time window [ms]

        assert all([txt_path, h5_path]) is False, "Can only load .txt OR .h5 file"
        if txt_path is not None:
            assert os.path.exists(txt_path), "File does not exist!"
            # read event data from .txt file
            self.read_txt(txt_path)
        if h5_path is not None:
            assert os.path.exists(h5_path), "File does not exist!"
            # read event data from .h5 file
            self.read_h5(txt_path)

        tms = self.event_data.to_timeunit("ms").t.ravel()
        x = self.event_data.x.ravel()
        y = self.event_data.y.ravel()
        polarity = self.event_data.p.ravel()

        colors = np.where(polarity, "g", "r")

        # align camera frame with plt.scatter frame
        x = self.event_data.dim[0] - x

        # initialize first frame
        t0 = tms[0]
        window = ((tms >= t0) & (tms <= t0 + window_dt)).nonzero()
        line = ax.scatter(
            x[window],
            y[window],
            color=colors[window],
            marker="s",
            s=(72.0 / fig.dpi) ** 2,  # == exactly 1 pixel
            lw=0,
        )

        # progress bar
        bar = tqdm(desc="Writing frames", total=int((tms[-1] - tms[0]) / window_dt))

        def animate(i):
            window = ((tms >= i) & (tms <= i + window_dt)).nonzero()
            if window[0].size != 0:
                line.set_offsets(np.array([x[window], y[window]]).T)
                line.set_color(colors[window])
            bar.update(1)  # 1 frame written

        anim = FuncAnimation(
            fig,
            animate,
            interval=window_dt,
            frames=np.arange(t0 + window_dt, tms[-1], window_dt),
        )

        if visualize is True:
            plt.show()
        # save to bag_file parent directory by default
        if out_path is None:
            out_path = os.path.join(os.path.dirname(self.data_dir), "dvs.mp4")
        # make sure the base directory exists
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        if save is True:
            Writer = writers["ffmpeg"]
            writer = Writer(fps=framerate, codec=encoder, bitrate=framerate * 100)
            anim.save(out_path, writer=writer, savefig_kwargs={"facecolor": "white"})

        # cleanup
        plt.close(fig)
        bar.close()

    def _configure_dvsvisual(self, res=None):
        # initiate figure and figure axis
        fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=300, frameon=True)
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        ax.set_aspect(1)

        if res is not None:
            fig.set_size_inches((res[0] / fig.dpi, res[1] / fig.dpi))

        ax.set_xlim([0, self.event_data.dim[0]])
        ax.set_ylim([0, self.event_data.dim[1]])

        if self.unwrapped is True:
            # adjust axis and figure to fit mask
            ax.set_xlim([0, self.event_data.dim[0]])
            ax.set_ylim([0, self.event_data.dim[1]])
            fig.set_size_inches(
                360 / fig.dpi,
                (ax.get_ylim()[1] - ax.get_ylim()[0]) / fig.dpi,
                forward=True,
            )

        ax.set_axis_off()

        return fig, ax

class DutchNavDataset:
    def __init__(self):
        return

    @staticmethod
    def load_gps_h5(hdf5_file, gps0: tuple, gps1: tuple):
        f = hdf5_file
        gps_time = f["gps/rosTime"][:]
        gps_heading = f["gps/heading"][:]
        gps_gSpeed = f["gps/gSpeed"][:]
        gps_lat = f["gps/lat"][:]
        gps_lon = f["gps/lon"][:]
        t_start = time_min_dist(*gps0, gps_time, gps_lat, gps_lon)
        t_stop = time_min_dist(*gps1, gps_time, gps_lat, gps_lon)

        # select track part of GPS data where ground speed is at least 1 m/s
        # close to the start and finish point
        valid_speed = np.where(gps_gSpeed >= 1)
        gps_track = np.where((gps_time >= t_start) & (gps_time <= t_stop))
        gps_track_cap = (
            max(valid_speed[0][0], gps_track[0][0]),
            min(valid_speed[0][-1], gps_track[0][-1]),
        )
        gps_time = gps_time[gps_track_cap[0] : gps_track_cap[1]]
        gps_heading = gps_heading[gps_track_cap[0] : gps_track_cap[1]]

        return gps_time, gps_heading, t_start, t_stop

    @staticmethod
    def load_frames_h5(hdf5_file, images_name: str, t_start: float, t_stop: float):
        f = hdf5_file
        # prepare arrays
        images = np.zeros(
            (len(f[images_name]), *f[f"{images_name}/image000000000"].shape),
            dtype=np.uint8,
        )
        im_time = np.zeros((len(f[images_name])), dtype=np.float64)
        # read data into arrays
        for i, image in enumerate(f[images_name].values()):
            image.read_direct(images[i])
            im_time[i] = image.attrs["timestamp"]

        # select images within track
        im_track = np.where((im_time >= t_start) & (im_time <= t_stop))
        imgs = images[im_track]
        im_time = im_time[im_track]

        return imgs, im_time

    @staticmethod
    def load_events_h5(
        hdf5_file,
        t_start: float,
        t_stop: float,
        noise_filter: bool = False,
        mask: bool = False,
        unwrap: bool = False,
    ):
        # load dvs data
        dvs = DvsDataHandler()
        dvs.read_h5(hdf5_file=hdf5_file)
        if t_stop is not None:
            e_track = array_slice(dvs.event_data.t, t_start, t_stop)
            # select track data
            dvs.event_data.t = dvs.event_data.t[e_track]
            dvs.event_data.x = dvs.event_data.x[e_track]
            dvs.event_data.y = dvs.event_data.y[e_track]
            dvs.event_data.p = dvs.event_data.p[e_track]
        # filter, mask and unwrap
        if noise_filter is True:
            dvs.noise_filter_events()
        if mask is True:
            dvs.mask_events(cxy=(117, 74), r_out=67, r_in=45)
        if unwrap is True:
            dvs.unwrap_events(prec=3)

        return dvs.event_data
