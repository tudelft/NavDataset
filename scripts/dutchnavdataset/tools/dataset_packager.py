import os
from os.path import splitext
import h5py
import numpy as np


# inspired by https://github.com/TimoStoff/events_contrast_maximization
class hdf5_packager:
    """
    This class packages data to hdf5 files
    Data consisting of: DVS + IMU + GPS + GoPro footage
    """

    def __init__(self, output_path, max_buffer_size=1000000, overwrite_events=False):
        # fmt: off
        self.max_buffer_size = max_buffer_size
        print("CREATING FILE IN {}".format(output_path))
        self.hdf5_path = output_path
        # read write mode, creates new if non-existent
        self.hdf5_file = h5py.File(output_path, "a")
        data_kwargs = {"maxshape": (None,), "chunks": True}
        self.skip_events = False
        if self.hdf5_file.__contains__("events"):
            if overwrite_events is True:
                self.remove_data("events")
            else:
                self.skip_events = True

        self.event_xs = self.add_data("events/xs", (0,), dtype=np.dtype(np.int16), overwrite=overwrite_events, h5_kwargs=data_kwargs)
        self.event_ys = self.add_data("events/ys", (0,), dtype=np.dtype(np.int16), overwrite=overwrite_events, h5_kwargs=data_kwargs)
        self.event_ts = self.add_data("events/ts", (0,), dtype=np.dtype(np.float64), overwrite=overwrite_events, h5_kwargs=data_kwargs)
        self.event_ps = self.add_data("events/ps", (0,), dtype=np.dtype(np.bool_), overwrite=overwrite_events, h5_kwargs=data_kwargs)

        self.no_images = False
        self.no_imu = False
        self.no_gps = False
        # fmt: on

    def add_data(
        self,
        name,
        shape=None,
        dtype=None,
        data=None,
        overwrite=None,
        attrs: dict = {},
        h5_kwargs: dict = {},
    ):
        if self.hdf5_file.__contains__(name):
            if self.check_same(name, data) is True:
                return
            # overwrite=None: ask for each
            elif overwrite is None:
                ans = ""
                while ans not in ["y", "n"]:
                    ans = input(
                        "h5 file {} already contains topic {}, would you like to overwrite? (y|n)".format(
                            self.hdf5_path, name
                        )
                    )
                if ans == "y":
                    del self.hdf5_file[name]
                    dset = self.hdf5_file.create_dataset(
                        name, shape=shape, dtype=dtype, data=data, **h5_kwargs
                    )
                    for key, item in attrs.items():
                        dset.attrs[key] = item
                else:
                    return
            elif overwrite is False:
                return
            elif overwrite is True:
                self.remove_data(name)
                dset = self.hdf5_file.create_dataset(
                    name, shape=shape, dtype=dtype, data=data, **h5_kwargs
                )
                for key, item in attrs.items():
                    dset.attrs[key] = item
        else:
            dset = self.hdf5_file.create_dataset(
                name, shape=shape, dtype=dtype, data=data, **h5_kwargs
            )
            for key, item in attrs.items():
                dset.attrs[key] = item

        return dset

    def remove_data(self, names: str or "list[str]"):
        # remove selected topics
        if isinstance(names, list):
            for name in names:
                if self.hdf5_file.__contains__(name):
                    del self.hdf5_file[name]
        else:
            if self.hdf5_file.__contains__(names):
                del self.hdf5_file[names]
        return

    def check_same(self, name, data):
        check = np.all(self.hdf5_file[name][:] == data)
        return check

    def close(self):
        # copy files to new hdf5 file as file size remains
        # the same on modification, so copy data to new file
        # which lowers the amount of data used
        temp_file = splitext(self.hdf5_path)[0] + "_temp.h5"
        # copy data to new hdf5 file
        with h5py.File(temp_file, "w") as f_new:
            for a in self.hdf5_file.attrs:
                f_new.attrs[a] = self.hdf5_file.attrs[a]
            for d in self.hdf5_file:
                self.hdf5_file.copy(d, f_new)
        # remove original file and rename temp to original
        self.hdf5_file.close()
        os.remove(self.hdf5_path)
        os.rename(temp_file, self.hdf5_path)

    def append_to_dataset(self, dataset, data):
        # change size to accommodate data
        dataset.resize(dataset.shape[0] + len(data), axis=0)
        if len(data) == 0:
            return
        # add data
        dataset[-len(data) :] = data[:]

    def package_events(self, xs, ys, ts, ps):
        if self.skip_events is True:
            return
        self.append_to_dataset(self.event_xs, xs)
        self.append_to_dataset(self.event_ys, ys)
        self.append_to_dataset(self.event_ts, ts)
        self.append_to_dataset(self.event_ps, ps)

    def package_image(
        self, image, timestamp, img_idx, custom_name=None, overwrite: bool = False
    ):
        name = "images" if custom_name is None else custom_name
        # delete all gopro data if overwrite is necessary, only once
        if self.no_images is False and (
            self.hdf5_file.__contains__(name) and overwrite is True
        ):
            self.remove_data(name)
            self.no_images = True

        image_type = (
            "greyscale"
            if image.shape[-1] == 1 or len(image.shape) == 2
            else "color_bgr"
        )
        attrs = {"size": image.shape, "timestamp": timestamp, "type": image_type}

        self.add_data(
            "{}/image{:09d}".format(name, img_idx),
            data=image,
            dtype=np.dtype(np.uint8),
            overwrite=overwrite,
            attrs=attrs,
        )

    def package_imu(self, imu_data, overwrite: bool = False):
        # delete all imu data if overwrite is necessary, only once
        if self.no_imu is False and (
            self.hdf5_file.__contains__("imu") and overwrite is True
        ):
            self.remove_data("imu")
            self.no_imu = True
        # fmt: off
        # time
        self.add_data("imu/rosTime", data=imu_data.rosTime, dtype=np.dtype(np.float64), overwrite=overwrite)
        # linear acceleration
        self.add_data("imu/a_x", data=imu_data["a_x"].to_numpy(), dtype=np.dtype(np.float64), overwrite=overwrite)
        self.add_data("imu/a_y", data=imu_data["a_y"].to_numpy(), dtype=np.dtype(np.float64), overwrite=overwrite)
        self.add_data("imu/a_z", data=imu_data["a_z"].to_numpy(), dtype=np.dtype(np.float64), overwrite=overwrite)
        # angular velocity
        self.add_data("imu/w_x", data=imu_data["w_x"].to_numpy(), dtype=np.dtype(np.float64), overwrite=overwrite)
        self.add_data("imu/w_y", data=imu_data["w_y"].to_numpy(), dtype=np.dtype(np.float64), overwrite=overwrite)
        self.add_data("imu/w_z", data=imu_data["w_z"].to_numpy(), dtype=np.dtype(np.float64), overwrite=overwrite)
        # orientation
        self.add_data("imu/r_x", data=imu_data["r_x"].to_numpy(), dtype=np.dtype(np.float64), overwrite=overwrite)
        self.add_data("imu/r_y", data=imu_data["r_y"].to_numpy(), dtype=np.dtype(np.float64), overwrite=overwrite)
        self.add_data("imu/r_z", data=imu_data["r_z"].to_numpy(), dtype=np.dtype(np.float64), overwrite=overwrite)
        self.add_data("imu/r_w", data=imu_data["r_w"].to_numpy(), dtype=np.dtype(np.float64), overwrite=overwrite)
        # covariance matrices
        #                                       otherwise not recognized
        self.add_data("imu/k_a", data=np.array([d for d in imu_data["k_a"]]), dtype=np.dtype(np.float64), overwrite=overwrite)
        self.add_data("imu/k_w", data=np.array([d for d in imu_data["k_w"]]), dtype=np.dtype(np.float64), overwrite=overwrite)
        self.add_data("imu/k_r", data=np.array([d for d in imu_data["k_r"]]), dtype=np.dtype(np.float64), overwrite=overwrite)
        # fmt: on

    def package_gps(self, gps_data, overwrite: bool = False):
        # delete all gps data if overwrite is necessary, only once
        if self.no_gps is False and (
            self.hdf5_file.__contains__("gps") and overwrite is True
        ):
            self.remove_data("gps")
            self.no_gps = True

        # fmt: off
        self.add_data("gps/rosTime", data=gps_data["rosTime"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/iTOW", data=gps_data["iTOW"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/year", data=gps_data["year"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/month", data=gps_data["month"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/day", data=gps_data["day"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/hour", data=gps_data["hour"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/minute", data=gps_data["minute"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/sec", data=gps_data["sec"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/valid", data=gps_data["valid"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/tAcc", data=gps_data["tAcc"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/nano", data=gps_data["nano"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/fixType", data=gps_data["fixType"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/flags", data=gps_data["flags"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/flags2", data=gps_data["flags2"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/numSV", data=gps_data["numSV"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/lon", data=gps_data["lon"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/lat", data=gps_data["lat"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/height", data=gps_data["height"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/hMSL", data=gps_data["hMSL"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/hAcc", data=gps_data["hAcc"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/vAcc", data=gps_data["vAcc"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/velN", data=gps_data["velN"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/velE", data=gps_data["velE"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/velD", data=gps_data["velD"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/gSpeed", data=gps_data["gSpeed"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/heading", data=gps_data["heading"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/sAcc", data=gps_data["sAcc"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/headAcc", data=gps_data["headAcc"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/pDOP", data=gps_data["pDOP"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/reserved1", data=gps_data["reserved1"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/headVeh", data=gps_data["headVeh"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/magDec", data=gps_data["magDec"].to_numpy(), overwrite=overwrite)
        self.add_data("gps/magAcc", data=gps_data["magAcc"].to_numpy(), overwrite=overwrite)
        # fmt: on

    def add_metadata(self, metadata, topic=None):
        if topic is None:
            for key, val in metadata.items():
                self.hdf5_file.attrs[key] = val
        else:
            for key, val in metadata.items():
                self.hdf5_file[topic].attrs[key] = val
