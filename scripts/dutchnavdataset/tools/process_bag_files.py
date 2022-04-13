#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =====================================================================================
# Import libraries
# =====================================================================================
# built-in libraries
import os

# efficient progress bar
from tqdm import tqdm

# linear algebra and dataframes
import numpy as np
import pandas as pd

# ROS messages
import rosbag
import rospy
from sensor_msgs.msg import Imu
from dvs_msgs.msg import EventArray, Event

# event IO
from dv import AedatFile
import h5py
from tools.dataset_packager import hdf5_packager

# local libraries
from utils.routines import strtobool
from data_types.data_types import EventData, NavPVT


def read_radar_data(bag_dir):
    bag = rosbag.Bag(bag_dir)

    rx1_re = []
    rx1_im = []
    rx2_re = []
    rx2_im = []
    time = []

    for _, msg, t in bag.read_messages(topics=["raw_data"]):

        # read radar data
        rx1_re.append(msg.data_rx1_im)
        rx2_re.append(msg.data_rx2_re)

        rx1_im.append(msg.data_rx1_im)
        rx2_im.append(msg.data_rx2_im)

        time.append(t.to_time())

    bag.close()

    radar_data = {}
    radar_data["rx1_re"] = np.array(rx1_re)
    radar_data["rx1_im"] = np.array(rx1_im)
    radar_data["rx2_re"] = np.array(rx2_re)
    radar_data["rx2_im"] = np.array(rx2_im)
    radar_data["t"] = np.array(time)

    return radar_data


def read_imu_data(bag_dir):
    """[summary]

    Args:
        bag_dir ([type]): [description]

    Returns:
        [type]: [description]
    """

    bag = rosbag.Bag(bag_dir)

    a_x, a_y, a_z = [], [], []
    w_x, w_y, w_z = [], [], []
    r_x, r_y, r_z, r_w = [], [], [], []
    k_a, k_w, k_r = [], [], []

    time = []
    timestamp_float = lambda t: t.secs + t.nsecs * 10 ** (-9)

    for _, msg, t in bag.read_messages(topics=["raw_data", "data"]):
        # read imu data
        # linear acceleration
        a_x.append(msg.linear_acceleration.x)
        a_y.append(msg.linear_acceleration.y)
        a_z.append(msg.linear_acceleration.z)
        # angular velocity
        w_x.append(msg.angular_velocity.x)
        w_y.append(msg.angular_velocity.y)
        w_z.append(msg.angular_velocity.z)
        # orientation
        r_x.append(msg.orientation.x)
        r_y.append(msg.orientation.y)
        r_z.append(msg.orientation.z)
        r_w.append(msg.orientation.w)
        # covariance matrices
        k_a.append(np.array(msg.linear_acceleration_covariance).reshape(3, 3))
        k_w.append(np.array(msg.angular_velocity_covariance))
        k_r.append(np.array(msg.orientation_covariance))

        time.append(timestamp_float(t))

    bag.close()

    imu_data = pd.DataFrame()
    imu_data["a_x"] = np.array(a_x)
    imu_data["a_y"] = np.array(a_y)
    imu_data["a_z"] = np.array(a_z)
    imu_data["w_x"] = np.array(w_x)
    imu_data["w_y"] = np.array(w_y)
    imu_data["w_z"] = np.array(w_z)
    imu_data["r_x"] = np.array(r_x)
    imu_data["r_y"] = np.array(r_y)
    imu_data["r_z"] = np.array(r_z)
    imu_data["r_w"] = np.array(r_w)
    imu_data["k_a"] = k_a
    imu_data["k_w"] = k_w
    imu_data["k_r"] = k_r
    imu_data["t"] = np.array(time)

    return imu_data


def write_imu_data(imu_data, out_dir):
    # http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/Imu.html

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    topic = "raw_data"
    with rosbag.Bag(os.path.join(out_dir, "", "IMU_BAG.bag"), "w") as bag:
        for i, t in enumerate(imu_data["t"]):

            imu_msg = Imu()

            timestamp = rospy.Time.from_sec(t)  # from ms to sec
            imu_msg.header.stamp = timestamp

            # Populate the data elements for IMU
            imu_msg.orientation.x = 0.0
            imu_msg.orientation.y = 0.0
            imu_msg.orientation.z = 0.0
            imu_msg.orientation.w = 0.0
            imu_msg.orientation_covariance = [
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]

            imu_msg.angular_velocity.x = imu_data["w_x"][i]
            imu_msg.angular_velocity.y = imu_data["w_y"][i]
            imu_msg.angular_velocity.z = imu_data["w_z"][i]
            imu_msg.angular_velocity_covariance = [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]

            imu_msg.linear_acceleration.x = imu_data["a_x"][i]
            imu_msg.linear_acceleration.y = imu_data["a_y"][i]
            imu_msg.linear_acceleration.z = imu_data["a_z"][i]
            imu_msg.linear_acceleration_covariance = [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]

            bag.write(topic, imu_msg, timestamp)
    return

def read_NavPVT_bag(bag_dir) -> NavPVT:
    """Read NavPVT messages from a ROS .bag file.

    Args:
        bag_dir (str): path to the .bag file containing the gps data.
    """

    bag = rosbag.Bag(bag_dir)

    time = []
    iTOW = []
    year = []
    month = []
    day = []
    hour = []
    minute = []
    sec = []
    valid = []
    tAcc = []
    nano = []
    fixType = []
    flags = []
    flags2 = []
    numSV = []  # Number of SVs used in Nav Solution
    lon = []
    lat = []
    height = []  # Height above Ellipsoid [mm]
    hMSL = []  # Height above mean sea level [mm]
    hAcc = []  # Horizontal Accuracy Estimate [mm]
    vAcc = []  # Vertical Accuracy Estimate [mm]
    velN = []  # NED north velocity [mm/s]
    velE = []  # NED east velocity [mm/s]
    velD = []  # NED down velocity [mm/s]
    gSpeed = []  # Ground Speed (2-D) [mm/s]
    heading = []  # Heading of motion 2-D [deg / 1e-5]
    sAcc = []  # Speed Accuracy Estimate [mm/s]
    headAcc = []  # Heading Accuracy Estimate (both motion & vehicle) [deg / 1e-5]
    pDOP = []  # Position DOP [1 / 0.01]
    reserved1 = []
    headVeh = []  # Heading of vehicle (2-D) [deg / 1e-5]
    magDec = []  # Magnetic declination [deg / 1e-2]
    magAcc = []  # Magnetic declination accuracy [deg / 1e-2]

    topics = ["/ublox/navPVT", "data"]
    for _, msg, rosTime in bag.read_messages(topics=topics):
        # ROS time
        time.append(rosTime.to_sec())

        # Message specific
        iTOW.append(msg.iTOW)
        year.append(msg.year)
        month.append(msg.month)
        day.append(msg.day)
        hour.append(msg.hour)
        minute.append(msg.min)
        sec.append(msg.sec)
        valid.append(msg.valid)
        tAcc.append(msg.tAcc)
        nano.append(msg.nano)
        fixType.append(msg.fixType)
        flags.append(msg.flags)
        flags2.append(msg.flags2)
        numSV.append(msg.numSV)
        lon.append(msg.lon)
        lat.append(msg.lat)
        height.append(msg.height)
        hMSL.append(msg.hMSL)
        hAcc.append(msg.hAcc)
        vAcc.append(msg.vAcc)
        velN.append(msg.velN)
        velE.append(msg.velE)
        velD.append(msg.velD)
        gSpeed.append(msg.gSpeed)
        heading.append(msg.heading)
        sAcc.append(msg.sAcc)
        headAcc.append(msg.headAcc)
        pDOP.append(msg.pDOP)
        reserved1.append(msg.reserved1)
        headVeh.append(msg.headVeh)
        magDec.append(msg.magDec)
        magAcc.append(msg.magAcc)

    bag.close()

    # Store into NavPvt datatype
    # convert to SI units
    navPVT = NavPVT()

    navPVT.t = np.array(time)
    navPVT.iTOW = np.array(iTOW)
    navPVT.year = np.array(year)
    navPVT.month = np.array(month)
    navPVT.day = np.array(day)
    navPVT.hour = np.array(hour)
    navPVT.minute = np.array(minute)
    navPVT.sec = np.array(sec)
    navPVT.valid = np.array(valid)
    navPVT.tAcc = np.array(tAcc)
    navPVT.nano = np.array(nano)
    navPVT.fixType = np.array(fixType)
    navPVT.flags = np.array(flags)
    navPVT.flags2 = np.array(flags2)
    navPVT.numSV = np.array(numSV)
    navPVT.lon = np.array(lon) / 1e7
    navPVT.lat = np.array(lat) / 1e7
    navPVT.height = np.array(height) / 1e3
    navPVT.hMSL = np.array(hMSL) / 1e3
    navPVT.hAcc = np.array(hAcc) / 1e3
    navPVT.vAcc = np.array(vAcc) / 1e3
    navPVT.velN = np.array(velN) / 1e3
    navPVT.velE = np.array(velE) / 1e3
    navPVT.velD = np.array(velD) / 1e3
    navPVT.gSpeed = np.array(gSpeed) / 1e3
    navPVT.heading = np.array(heading) / 1e5
    navPVT.sAcc = np.array(sAcc) / 1e3
    navPVT.headAcc = np.array(headAcc) / 1e5
    navPVT.pDOP = np.array(pDOP) / 1e3
    navPVT.reserved1 = np.array(reserved1)
    navPVT.headVeh = np.array(headVeh) / 1e3
    navPVT.magDec = np.array(magDec) / 1e2
    navPVT.magAcc = np.array(magAcc) / 1e2

    return navPVT


class DvsDataReader:
    def __init__(self):
        return

    @staticmethod
    def read_aedat4(aedat4_path):
        with AedatFile(aedat4_path) as f:
            events = np.hstack([packet for packet in f["events"].numpy()])
        event_data = EventData()
        event_data.x = events["x"]
        event_data.y = events["y"]
        event_data.p = events["polarity"]
        event_data.t = (events["timestamp"] - events["timestamp"][0]) * 10 ** (-3)

        return event_data

    # inspired by https://github.com/TimoStoff/events_contrast_maximization/blob/master/tools/read_events.py
    @staticmethod
    def read_h5(
        hdf_path: str = "", hdf5_file=None
    ) -> tuple[EventData, bool, bool, str]:
        assert (
            os.path.exists(hdf_path) or hdf5_file is not None
        ), "Either give a valid path to a .h5 file (given: {}), or an open h5 file (given: {})".format(
            hdf_path, hdf5_file
        )
        if hdf5_file is not None:
            f = hdf5_file
        else:
            f = h5py.File(hdf_path, "r")

        event_data = EventData()
        event_data.x = f["events/xs"][:]
        event_data.y = f["events/ys"][:]
        event_data.p = f["events/ps"][:]
        event_data.t = f["events/ts"][:]
        event_data.dim = f["events"].attrs["resolution"]

        data_dir = f["events"].attrs["from_file"]
        masked = f["events"].attrs["masked"]
        unwrapped = f["events"].attrs["unwrapped"]

        if hdf5_file is None:
            f.close()

        return event_data, masked, unwrapped, data_dir

    @staticmethod
    def read_txt(txt_path, nrows=None) -> tuple[EventData, bool, bool, str]:
        data_dir = txt_path
        h_dict = {}  # header dictionary
        event_data = EventData()
        # read event data from .txt file
        print("Reading {} ...".format(txt_path))
        with open(txt_path, "r") as f:
            # header format: "from_file: /path, resolution: tuple, masked: bool, unwrapped: bool
            header = f.readline()[:-1].split(", ")
            for h in header:
                h_dict[h.split(": ", 2)[0]] = h.split(": ", 2)[1]

            event_data.dim = tuple(int(r) for r in h_dict["resolution"].split("x"))
            masked = strtobool(h_dict["masked"])
            unwrapped = strtobool(h_dict["unwrapped"])

            ev_mat = pd.read_csv(
                f,
                header=1,
                names=["t", "x", "y", "p"],
                dtype={"t": np.float, "x": np.int16, "y": np.int16, "p": bool},
                engine="c",
                nrows=nrows,
            )

        event_data.t = ev_mat.t.to_numpy()
        event_data.x = ev_mat.x.to_numpy()
        event_data.y = ev_mat.y.to_numpy()
        event_data.p = ev_mat.p.to_numpy()

        return event_data, masked, unwrapped, data_dir

    @staticmethod
    def read_bag(bag_path, num_msgs=None) -> EventData:
        bag = rosbag.Bag(bag_path)
        event_data = EventData()

        # get number of messages
        if num_msgs is None:
            num_msgs = bag.get_message_count(topic_filters=["data", "raw_data"])

        # get dimension of input events
        for _, msg, _ in bag.read_messages(topics=["data", "raw_data"]):
            event_data.dim = (msg.width, msg.height)
            break

        # loading as python object takes about 2.5x the memory
        # if small size (< 500MB), read directly into memory
        # else prompt to first convert to .txt/.h5 format
        file_size = os.path.getsize(bag_path)
        if file_size < 500_000_000:
            # initialize numpy arrays to store data
            xs, ys, ts, ps = [], [], [], []
            rosTime = []

            for count, (_, msg, time) in tqdm(
                enumerate(bag.read_messages(topics=["data", "raw_data"])),
                desc="Reading {} ...".format(bag_path),
                total=num_msgs,
            ):
                for e in msg.events:
                    xs.append(e.x)
                    ys.append(e.y)
                    ps.append(e.polarity)
                    ts.append(e.ts)
                    rosTime.append(time)

                # only read the asked amount of messages
                if count == num_msgs:
                    break

            bag.close()

            # convert rospy Time to time in seconds
            to_sec = np.vectorize(lambda t: t.secs + t.nsecs * 10 ** (-9))

            event_data.x = np.hstack(xs).astype(np.int16)
            event_data.y = np.hstack(ys).astype(np.int16)
            event_data.p = np.hstack(ps).astype(np.bool_)
            event_data.t = np.hstack(to_sec(np.hstack(ts)))
            event_data.rosTime = rosTime

            event_data._reset_t0()

            return event_data

        else:
            raise MemoryError("File size is too large, convert to txt/h5 format first.")

    @staticmethod
    def bag_to_txt(bag_path, out_path=None, fix_t0=False, num_msgs=None):
        event_data = EventData()
        # format ts string helper function
        timestamp_str = lambda ts: str(ts.secs) + "." + str(ts.nsecs).zfill(9)
        # set output directory
        if out_path is None:
            out_path = os.path.join(os.path.dirname(bag_path), "DVS.txt")
        # make sure the base directory exists
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))

        with rosbag.Bag(bag_path, "r") as bag:
            # get number of messages
            if num_msgs is None:
                num_msgs = bag.get_message_count(topic_filters=["data", "raw_data"])

            # get dimension of input events
            for _, msg, _ in bag.read_messages(topics=["data", "raw_data"]):
                event_data.dim = (msg.width, msg.height)
                break

            # write events to csv .txt file
            with open(out_path, "w") as f:
                # write header
                f.write(
                    f"from_file: {bag_path}, resolution: {event_data.dim[0]}x{event_data.dim[1]}, masked: False, unwrapped: False\n"
                )
                f.write("time [s], x [-], y [-], polarity [-]\n")
                # write events
                for _, msg, _ in tqdm(
                    bag.read_messages(topics=["data", "raw_data"]),
                    total=num_msgs,
                    desc="Writing to txt file",
                ):
                    for e in msg.events:
                        f.write(timestamp_str(e.ts) + ",")
                        f.write(str(e.x) + ",")
                        f.write(str(e.y) + ",")
                        f.write(("1" if e.polarity else "0") + "\n")

        if fix_t0 is True:
            event_data, masked, unwrapped, _ = DvsDataReader.read_txt(out_path)
            event_data._reset_t0()
            header = f"from_file: {bag_path}, resolution: {event_data.dim[0]}x{event_data.dim[1]}, masked: {masked}, unwrapped: {unwrapped}\ntime [s], x [-], y [-], polarity [-]"
            event_data.to_txt(out_path, header=header)
        return

    @staticmethod
    def bag_to_h5(
        bag_path,
        out_path=None,
        event_topic="data",
        packager=None,
        fix_t0=False,
        sensor_size=None,
        verbose=False,
    ):
        if packager is None:
            if out_path is None:
                out_path = os.path.join(os.path.dirname(bag_path), "DVS.h5")
            ep = hdf5_packager(out_path)
        else:
            ep = packager
            if packager.skip_events is True:
                return

        with rosbag.Bag(bag_path, "r") as bag:
            # check if topic contains data
            assert (
                bag.get_message_count(event_topic) > 0
            ), "No messages in topic {}".format(event_topic)

            # get dimension of input events
            for _, msg, _ in bag.read_messages(topics=(event_topic)):
                res = (msg.width, msg.height)
                break
            # Extract events to h5
            xs, ys, ts, ps = [], [], [], []
            max_buffer_size = 75_000_000  # about 10GB
            timestamp_float = lambda t: t.secs + t.nsecs * 10 ** (-9)

            for _, msg, _ in tqdm(
                bag.read_messages(topics=(event_topic)),
                desc="reading {} ...".format(bag_path),
                total=bag.get_message_count(topic_filters=(event_topic)),
            ):
                for e in msg.events:
                    xs.append(e.x)
                    ys.append(e.y)
                    ts.append(timestamp_float(e.ts))
                    ps.append(1 if e.polarity else 0)

                if len(xs) > max_buffer_size:
                    if (
                        sensor_size is None
                        or sensor_size[0] < max(ys)
                        or sensor_size[1] < max(xs)
                    ):
                        sensor_size = [max(xs), max(ys)]

                    ep.package_events(xs, ys, ts, ps)
                    del xs[:]
                    del ys[:]
                    del ts[:]
                    del ps[:]
            if (
                sensor_size is None
                or sensor_size[0] < max(ys)
                or sensor_size[1] < max(xs)
            ):
                sensor_size = [max(xs), max(ys)]

            ep.package_events(xs, ys, ts, ps)
            del xs[:]
            del ys[:]
            del ts[:]
            del ps[:]
        if verbose is True:
            print("Detect sensor size {}".format(sensor_size))

        # add meta data
        meta_data = {
            "from_file": bag_path,
            "resolution": res,
            "masked": False,
            "unwrapped": False,
        }
        ep.add_metadata(meta_data, "events")

        if packager is None:
            ep.close()

        if fix_t0 is True:
            event_data, masked, unwrapped, _ = DvsDataReader.read_h5(out_path)
            event_data = event_data._reset_t0()
            meta_data = {
                "from_file": bag_path,
                "resolution": res,
                "masked": masked,
                "unwrapped": unwrapped,
            }
            event_data.to_h5(out_path, meta_data=meta_data)

        return


def write_dvs2bag(event_data: EventData, out_dir: str, topic: str = "raw_data"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    events_msg = EventArray()
    events_msg.header.stamp = event_data.t[0]
    events_msg.width = event_data.dim[0]
    events_msg.height = event_data.dim[1]

    with rosbag.Bag(os.path.join(out_dir, "DVS.bag"), "wb") as bag:
        events = []
        for x, y, ts, pol in zip(
            event_data.x, event_data.y, event_data.t, event_data.pol
        ):
            event = Event()
            event.x = x
            event.y = y
            event.ts = ts
            event.polarity = pol

            events.append(event)

        events_msg.events = events
        bag.write(topic, events_msg, event_data.t[0])

    return
