# =====================================================================================
# This program processes the raw data (.bag files) of the DutchNavDataset
# =====================================================================================
import os
import subprocess as sp
import shlex
import yaml

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../dutchnavdataset")))
from tools.process_bag_files import DvsDataReader, read_NavPVT_bag, read_imu_data
from tools.data_loader import DvsDataHandler
from tools.dataset_packager import hdf5_packager
from tools.data_visualizer import video_to_frames_h5

BASE_DIR = ""  # path to the directory of the unprocessed dataset
BASE_TARGET_DIR = None  # path to where the processed data should be saved to
ROUTE_INFO_YAML = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset_information/RouteInformation.yaml"))  # path to the RouteInformation.yaml file
create_dvs_videos = False  # create videos of the dvs data

with open(ROUTE_INFO_YAML, "r") as f:
    d = yaml.load(f, Loader=yaml.FullLoader)

# - means gopro is ahead
# + means gopro is lagging
gopro_offsets = {
    "GX010026.MP4": 0,
    "GX010027.MP4": 0,
    "GX010028.MP4": -7,
    "GX010029.MP4": 0,
    "GX010031.MP4": -1.5,
    "GX010032.MP4": 0,
    "GX010033.MP4": -4,
    "GX010034.MP4": -2,
    "GX010035.MP4": -2,
    "GX010037.MP4": -15,
    "GX010038.MP4": -3,
    "GX010039.MP4": -2.5,
}

for route in d["route_list"]:
    for date_dir in d["routes"][route]["directory"]:
        dir = os.path.join(BASE_DIR, date_dir)

        # make directory for saving processed files
        if BASE_TARGET_DIR is None:
            target_dir = os.path.join(BASE_DIR + "_Processed", date_dir)
        else:
            target_dir = os.path.join(BASE_TARGET_DIR, date_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # initiate event packager
        hdf5_path = os.path.join(target_dir, "dataset.h5")
        ep = hdf5_packager(hdf5_path)

        # loop through directory
        for f in sorted(os.listdir(dir)):
            raw_file = os.path.join(dir, f)
            target = os.path.join(target_dir, os.path.basename(raw_file).split(".")[0])

            print("Processing: {}".format(raw_file))

            # GPS data
            if raw_file.endswith("GPS.bag"):
                # for .kmz file conversion (visualization in Google Earth)
                # https://www.gpsvisualizer.com/map_input?form=google
                navPVT = read_NavPVT_bag(raw_file).to_csv(save_dir=target + ".csv")
                # package to hdf5 file
                ep.package_gps(navPVT)

            # IMU data
            if raw_file.endswith("IMU.bag") and not raw_file.endswith("RS_IMU.bag"):
                imu_data = read_imu_data(raw_file)
                ep.package_imu(imu_data)

            # DVS data
            if raw_file.endswith("DVS.bag"):
                # read bag file
                DvsDataReader.bag_to_h5(raw_file, packager=ep)
                dvs = DvsDataHandler()

                if create_dvs_videos is True:
                    dvs.read_h5(hdf5_path)
                    # mask events and visualize
                    dvs.mask_events(cxy=(117, 74), r_out=67, r_in=45)
                    dvs.visualize_events(
                        out_path=target + "_masked.mp4", framerate=60
                    )
                    # unwrap events and visualize
                    dvs.unwrap_events(prec=3)
                    dvs.visualize_events(
                        out_path=target + "_unwrapped.mp4", framerate=60
                    )
                    # filter and visualize events
                    dvs.noise_filter_events()
                    dvs.visualize_events(
                        out_path=target + "_filtered.mp4", framerate=60
                    )

            # GoPro data
            if raw_file.endswith(".MP4"):
                # Recode gopro files (otherwise all frames are not extracted by OpenCV) and remove audio
                overwrite_arg = "-y"
                out = sp.Popen(
                    shlex.split(
                        f"ffmpeg -i {raw_file} -c copy -an {'-y' if overwrite_arg is True else '-n'} {target + '.mp4'}"
                    ),
                )
                # Wait for sub-process to finish
                out.wait()
                # Terminate the sub-process
                out.terminate()

                # convert recoded gopro footage to individual frames
                # rescaled frames for generating events
                video_to_frames_h5(
                    target + ".mp4",
                    ep,
                    start=navPVT.t[0],
                    offset=gopro_offsets.get(os.path.basename(raw_file)),
                    overwrite=overwrite_arg,
                )
                # rescaled and histogram equalized for input to frame-based nets
                video_to_frames_h5(
                    target + ".mp4",
                    ep,
                    start=navPVT.t[0],
                    offset=gopro_offsets.get(os.path.basename(raw_file)),
                    preprocess=True,
                    overwrite=overwrite_arg,
                )

        ep.close()
