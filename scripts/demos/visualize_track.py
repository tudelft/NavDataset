import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../dutchnavdataset")))
from tools.data_visualizer import DatasetVideoWriter

# fill in dataset directory
DATASET_DIR = "/home/janv/Documents/Thesis/Dataset"
dataset_path = os.path.join(DATASET_DIR, "InsectNavDataset_Processed/2021-04-08_172710/dataset.h5")
# find gopro footage
gopro_path = os.path.join(
    os.path.dirname(dataset_path),
    next(
        filter(
            lambda s: s.startswith("GX") and s.endswith(".mp4"),
            os.listdir(os.path.dirname(dataset_path)),
        )
    ),
)

DataVis = DatasetVideoWriter()
DataVis.visualize(
    dataset_path,
    gopro_path,
    gopro_offset=0,
    out_path="/home/janv/Documents/temp/mosaic2.mp4",
    save=True,
    visualize=False,
    start=551,
    stop=None,
)
