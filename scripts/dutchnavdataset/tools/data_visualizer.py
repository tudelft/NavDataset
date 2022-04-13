import os
import sys
import numpy as np
import h5py
import smopy
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from tools.data_loader import DvsDataHandler


class DatasetVideoWriter:
    def __init__(self):
        return

    @staticmethod
    def _configure_visualization(start, stop, dpi):
        fig_size = (10, 8)
        fig = plt.figure(figsize=fig_size, dpi=dpi, constrained_layout=False)

        ax_imu = fig.subplots(
            nrows=3,
            ncols=2,
            sharex=True,
            gridspec_kw={"top": 0.95, "bottom": 0.65, "left": 0.07, "right": 0.95},
        )
        gs_gopro = fig.add_gridspec(
            nrows=1, ncols=2, top=0.56, bottom=0.28, left=0.07, right=0.95
        )
        gs_dvs = fig.add_gridspec(nrows=1, ncols=1, top=0.25, bottom=0.06)
        gs_time = fig.add_gridspec(nrows=1, ncols=1, top=0.05, bottom=0)

        fig.suptitle("DVS IMU", size="medium")

        ax_gps = fig.add_subplot(gs_gopro[0, 0])
        ax_gps.set_title("Travelled route", size="medium")

        ax_gopro = fig.add_subplot(gs_gopro[0, 1])
        ax_gopro.set_title("GoPro footage", size="medium")

        ax_dvs = fig.add_subplot(gs_dvs[:])
        ax_dvs.set_title("Unwrapped + filtered DVS footage", size="medium")
        ax_dvs.set_aspect(1)

        ax_time = fig.add_subplot(gs_time[:])

        for ax in (ax_gps, ax_gopro, ax_time):
            ax.axis("off")

        ax_dvs.set(
            xticks=[0, 90, 180, 270, 360],
            xticklabels=[
                "$-180^\circ$",
                "$-90^\circ$",
                "$0^\circ$",
                "$90^\circ$",
                "$180^\circ$",
            ],
            yticks=[],
        )
        ax_dvs.set_aspect(1)

        ax_imu[-1, 0].set_xlim([0, stop - start])
        ax_imu[-1, 0].set_xlabel("t [s]")
        ax_imu[-2, 0].set_ylabel("IMU linear accelerations [$m\cdot s^{-2}$]")
        ax_imu[-1, 1].set_xlim([0, stop - start])
        ax_imu[-1, 1].set_xlabel("t [s]")
        ax_imu[-2, 1].set_ylabel("IMU angular velocity [$deg/s$]")

        ax_time.plot([start, stop], [0, 0], "b", lw=12)
        ax_time.text(
            (stop - start) * 1.02,
            -0.03,
            f"{stop - start:.2f} s",
            ma="center",
        )

        return fig, (ax_gps, ax_dvs, ax_imu, ax_gopro, ax_time)

    def visualize(
        self,
        dataset_path,
        gopro_path,
        out_path=None,
        save=True,
        visualize=False,
        gopro_offset=0,
        framerate=None,
        start=None,
        stop=None,
        dpi=100,
        encoder="hevc_nvenc",
    ):
        # check for appropriate data
        assert os.path.exists(dataset_path) and dataset_path.endswith(
            ".h5"
        ), f"{dataset_path} does not exist/is not the supported format (.h5)"
        # load dataset
        dataset = h5py.File(dataset_path, "r")
        assert os.path.exists(gopro_path) and gopro_path.endswith(
            ".mp4"
        ), f"{gopro_path} does not exist/is not a (supported .mp4) video file"

        start_stop_warning = (
            "Select start/stop point within bounds of data (0.00 s - {:.2f} s)".format(
                dataset["gps/rosTime"][-1] - dataset["gps/rosTime"][0]
            )
        )
        if start is None:
            start = dataset["gps/rosTime"][0]
        else:
            assert start >= 0, start_stop_warning
            assert start <= (
                dataset["gps/rosTime"][-1] - dataset["gps/rosTime"][0]
            ), start_stop_warning
            start += dataset["gps/rosTime"][0]
        if stop is None:
            stop = dataset["gps/rosTime"][-1]
        else:
            assert stop >= 0, start_stop_warning
            assert stop <= (
                dataset["gps/rosTime"][-1] - dataset["gps/rosTime"][0]
            ), start_stop_warning
            stop += dataset["gps/rosTime"][0]
            assert stop > start, "Stop point must be after start point"

        fig, axs = self._configure_visualization(0, stop - start, dpi)
        # convert to ms for sim
        start *= 10 ** 3
        stop *= 10 ** 3

        # GPS data prep
        lat = dataset["gps/lat"][:]
        lon = dataset["gps/lon"][:]
        gps_t = dataset["gps/rosTime"][:]
        gps_t = np.round(10 ** 3 * gps_t, 0)

        route_sel = ((gps_t >= start) & (gps_t <= stop)).nonzero()
        lat_min = lat[route_sel].min()
        lat_max = lat[route_sel].max()
        lon_min = lon[route_sel].min()
        lon_max = lon[route_sel].max()
        box = (lat_min, lon_min, lat_max, lon_max)

        map = smopy.Map(box, z=18, margin=0.05)
        gps_x, gps_y = map.to_pixels(lat, lon)

        map.show_mpl(axs[0])
        line_gps = axs[0].scatter(
            gps_x[route_sel][0], gps_y[route_sel][0], marker="s", s=1, lw=0
        )

        # GoPro data prep
        cap = cv2.VideoCapture(gopro_path)
        gopro_framerate = cap.get(cv2.CAP_PROP_FPS)
        if framerate is None:
            framerate = gopro_framerate  # fps determined by 'fastest' member

        if not cap.isOpened():
            print("Could not open the reference " + gopro_path)
            sys.exit(-1)

        # skip frames by gopro_offset milliseconds
        # and required start pos
        # - : gopro is ahead
        # + : gopro is behind
        if gopro_offset < 0:
            cap.set(
                1,
                round((-gopro_offset + (start - gps_t[0])) / (1000 / gopro_framerate)),
            )
        elif gopro_offset > 0:
            cap.set(1, round((start - gps_t[0]) / (1000 / gopro_framerate)))
            start += gopro_offset
            stop += gopro_offset
        else:
            cap.set(1, round((start - gps_t[0]) / (1000 / gopro_framerate)))

        _, im = cap.read()
        frame = axs[3].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

        # DVS prep
        dvs = DvsDataHandler()
        dvs.read_h5(dataset.filename)
        dvs.mask_events(cxy=(117, 74), r_out=67, r_in=45)
        dvs.unwrap_events(prec=3)
        dvs.noise_filter_events(cutoff=30)
        tms = dvs.event_data.to_timeunit("ms").t.ravel()
        x = dvs.event_data.x.ravel()
        y = dvs.event_data.y.ravel()
        polarity = dvs.event_data.p.ravel()
        colors = np.where(polarity, "b", "r")

        y = dvs.event_data.dim[1] - y

        axs[1].set_xlim([0, dvs.event_data.dim[0]])
        axs[1].set_ylim([0, dvs.event_data.dim[1]])

        # initialize first frame
        t0 = start
        window_dt = 1000 / framerate
        window = ((tms >= t0) & (tms <= t0 + window_dt)).nonzero()
        line_dvs = axs[1].scatter(
            x[window],
            y[window],
            color=colors[window],
            marker="s",
            # s=(72.0 / fig.dpi) ** 2,  # is exactly 1 pixel
            s=2,
            lw=0,
        )

        # IMU data prep
        imu_t = dataset["imu/rosTime"][:]
        imu_t = np.round(10 ** 3 * imu_t, 0)
        route_sel = ((imu_t >= start) & (imu_t <= stop)).nonzero()
        a_x = dataset["imu/a_x"][:]
        a_y = dataset["imu/a_y"][:]
        a_z = dataset["imu/a_z"][:]
        rad2deg = 180 / np.pi
        w_x = dataset["imu/w_x"][:] * rad2deg
        w_y = dataset["imu/w_y"][:] * rad2deg
        w_z = dataset["imu/w_z"][:] * rad2deg
        line_imu_ax = axs[2][0, 0].plot(imu_t[0], a_x[0], lw=1, label="accX")[0]
        line_imu_ay = axs[2][1, 0].plot(imu_t[0], a_y[0], lw=1, label="accY")[0]
        line_imu_az = axs[2][2, 0].plot(imu_t[0], a_z[0], lw=1, label="accZ")[0]
        line_imu_wx = axs[2][0, 1].plot(imu_t[0], w_x[0], lw=1, label="p")[0]
        line_imu_wy = axs[2][1, 1].plot(imu_t[0], w_y[0], lw=1, label="q")[0]
        line_imu_wz = axs[2][2, 1].plot(imu_t[0], w_z[0], lw=1, label="r")[0]
        bounds = lambda arr: (arr.min(), arr.max())
        for limits, ax in zip(
            (
                bounds(a_x[route_sel]),
                bounds(w_x[route_sel]),
                bounds(a_y[route_sel]),
                bounds(w_y[route_sel]),
                bounds(a_z[route_sel]),
                bounds(w_z[route_sel]),
            ),
            axs[2].flat,
        ):
            ax.set_ylim(limits)
            ax.legend()

        # time bar prep
        time = axs[4].text(
            -0.08 * (stop - start) / 1000, -0.03, f"{start:.2f} s", ma="center"
        )
        time_indicator = axs[4].text(0, -0.01, "|", color="white")

        # progress bar
        if save is True:
            bar = tqdm(desc="Making video ...", total=int((stop - start) / window_dt))

        # animation func
        def animate(i):
            # update gps
            window = ((gps_t >= start) & (gps_t <= i + window_dt)).nonzero()
            if window[0].size != 0:
                line_gps.set_offsets(np.array([gps_x[window], gps_y[window]]).T)
                line_gps.set_color("r")

            # update time bar
            time_indicator.set_position(np.array([(i - start) / 1000, -0.01]).T)
            time.set_text(f"{(i - start) / 1000:.2f} s")

            # update dvs
            window = ((tms >= i) & (tms <= i + window_dt)).nonzero()
            if window[0].size != 0:
                line_dvs.set_offsets(np.array([x[window], y[window]]).T)
                line_dvs.set_color(colors[window])

            # update IMU
            window = ((imu_t >= start) & (imu_t <= i + window_dt)).nonzero()
            if window[0].size != 0:
                line_imu_ax.set_data(
                    np.array([(imu_t[window] - start) / 1000, a_x[window]])
                )
                line_imu_ay.set_data(
                    np.array([(imu_t[window] - start) / 1000, a_y[window]])
                )
                line_imu_az.set_data(
                    np.array([(imu_t[window] - start) / 1000, a_z[window]])
                )
                line_imu_wx.set_data(
                    np.array([(imu_t[window] - start) / 1000, w_x[window]])
                )
                line_imu_wy.set_data(
                    np.array([(imu_t[window] - start) / 1000, w_y[window]])
                )
                line_imu_wz.set_data(
                    np.array([(imu_t[window] - start) / 1000, w_z[window]])
                )

            # update video
            if framerate != gopro_framerate:
                cap.set(1, round((i - gps_t[0]) / (1000 / gopro_framerate)))
            ret, im = cap.read()
            if ret is False:
                print("End of stream received, exiting ...")
                cap.release()
                sys.exit(0)
            frame.set_array(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

            # update progress bar
            if save is True:
                bar.update(1)

            return (
                line_gps,
                line_dvs,
                line_imu_ax,
                line_imu_ay,
                line_imu_az,
                line_imu_wx,
                line_imu_wy,
                line_imu_wz,
                frame,
                time_indicator,
            )

        anim = FuncAnimation(
            fig,
            animate,
            interval=window_dt,
            frames=np.arange(t0 + window_dt, stop, window_dt),
            repeat=False,
            blit=True,
        )

        if visualize is True:
            plt.show()
        # save to current working directory by default
        if out_path is None:
            out_path = os.path.join(os.getcwd(), "DatasetMosaic.mp4")
        # make sure the base directory exists
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        if save is True:
            Writer = writers["ffmpeg"]
            writer = Writer(fps=framerate, codec=encoder, bitrate=framerate * 100)
            anim.save(out_path, writer=writer)
            print(f"Video written to: {out_path}")

        # cleanup
        plt.close(fig)

        return


def video_to_frames(video_path, out_dir, dim=(28, 8), overwrite=False):
    """Convert video stream to gray scale histogram-equalized images of requested resolution and store them in directory as separate numbered images."""

    if (
        os.path.exists(os.path.join(out_dir, "frame0.png")) is False
        or overwrite is True
    ):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # open video stream
        cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print("Could not open the reference " + video_path)
            sys.exit(-1)

        pbar = tqdm(desc="Writing frames ...", total=cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # read, process and write frame to output_dir
        framenum = -1  # Frame counter
        while True:
            ret, frame = cap.read()  # read single frame
            framenum += 1
            if not ret:
                if framenum == 0:
                    print(
                        "Can't receive frame! Try re-encoding the video files with ex. ffmpeg\n"
                    )
                print("Stream end. Exiting ...\n")
                cap.release()
                break
            # convert to greyscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # histogram equalize to increase contrast
            hist_eq = cv2.equalizeHist(gray)
            # resize image to desired resolution
            resized = cv2.resize(hist_eq, dim, interpolation=cv2.INTER_AREA)
            # write numbered images to output directory
            cv2.imwrite(os.path.join(out_dir, f"frame{framenum}.png"), resized)

            pbar.update(1)

        pbar.close()
        print(f"Wrote {framenum} image frames to {os.path.abspath(out_dir)}")

    return


def video_to_frames_h5(
    video_path,
    hdf5_packager,
    dim=(28, 8),
    start=0,
    offset=0,
    preprocess=False,
    overwrite=False,
):
    """Convert video stream to gray scale histogram-equalized images of requested resolution and store them in directory as separate numbered images."""

    # open video stream
    cap = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Could not open the reference " + video_path)
        sys.exit(-1)

    has_cuda = False
    equalizeHist = cv2.equalizeHist
    cvtColor = cv2.cvtColor
    resize = cv2.resize

    if cv2.getBuildInformation().find("NVIDIA CUDA") != -1:
        has_cuda = True
        equalizeHist = cv2.cuda.equalizeHist
        cvtColor = cv2.cuda.cvtColor
        resize = cv2.cuda.resize

    framerate = cap.get(cv2.CAP_PROP_FPS)
    if has_cuda is True:
        frame = cv2.cuda_GpuMat()

    # progress bar
    pbar = tqdm(desc="Writing frames ...", total=cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # read, process and write frame to output_dir
    framenum = -1  # Frame counter
    while True:
        ret, _frame = cap.read()  # read single frame
        framenum += 1
        if not ret:
            if framenum == 0:
                print(
                    "Can't receive frame! Try re-encoding the video files with ex. ffmpeg\n"
                )
            print("Stream end. Exiting ...\n")
            cap.release()
            break

        if has_cuda is True:
            # upload to gpu
            frame.upload(_frame)
        else:
            frame = _frame

        # convert to greyscale
        gray = cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if preprocess is True:
            # histogram equalize to increase contrast
            hist_eq = equalizeHist(gray)
            # resize image to desired resolution
            _resized = resize(hist_eq, dim, interpolation=cv2.INTER_AREA)

            custom_name = "images_histeq"
        else:
            # resize image to desired resolution
            _resized = resize(gray, dim, interpolation=cv2.INTER_AREA)

            custom_name = "images"

        if has_cuda is True:
            resized = _resized.download()
        else:
            resized = _resized

        # write numbered images to output directory
        hdf5_packager.package_image(
            resized,
            start + offset + framenum / framerate,
            framenum,
            custom_name=custom_name,
            overwrite=overwrite,
        )

        # update progress bar
        pbar.update(1)

    pbar.close()
    print(f"Wrote {framenum} image frames to {hdf5_packager.hdf5_path}")

    return
