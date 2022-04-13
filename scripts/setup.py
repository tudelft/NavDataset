from setuptools import setup, find_packages


setup(
    name="dutchnavdataset",
    version="0.1",
    description="Framework for loading and manipulating data with the DutchNav dataset.",
    # This field corresponds to the "Home-Page" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#home-page-optional
    url="https://github.com/tudelft/NavDataset",
    author="Jan Verheyen",
    author_email="jan.verheyen@protonmail.com",
    classifiers=[
        "Development Status :: 3 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    package_dir={"": "dutchnavdataset"},
    packages=find_packages(where="dutchnavdataset"),
    python_requires=">=3.7, <4",
    install_requires=[
        "numpy>=1.18,<1.22",
        "numba",  # speed up numpy
        "pandas",
        "h5py",  # HDF5 handling
        "dv",  # aedat4 file format
        "matplotlib",  # plotting
        "scikit-image",  # image processing (canny edge detection hugh transform)
        "opencv-python",  # image processing (all the rest)
        "smopy",  # loading openstreetmaps maps
        "tqdm",  # progress bar
    ],
    project_urls={
        "Bug Reports": "https://github.com/tudelft/NavDataset/issues",
        "Dataset": "https://data.4tu.nl/articles/dataset/",
    },
)
