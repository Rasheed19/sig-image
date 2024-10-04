import logging
import os
import pickle
from typing import Any

import requests
import yaml


def get_rcparams():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 11

    FONT = {"family": "serif", "serif": ["Times"], "size": MEDIUM_SIZE}

    rc_params = {
        "axes.titlesize": MEDIUM_SIZE,
        "axes.labelsize": SMALL_SIZE,
        "xtick.labelsize": SMALL_SIZE,
        "ytick.labelsize": SMALL_SIZE,
        "legend.fontsize": SMALL_SIZE,
        "figure.titlesize": BIGGER_SIZE,
        "font.family": FONT["family"],
        "font.serif": FONT["serif"],
        "font.size": FONT["size"],
        "text.usetex": True,
        "axes.grid": False,
        # "axes.spines.top": False,
        # "axes.spines.right": False,
        # "axes.spines.left": True,
        # "axes.spines.bottom": True,
        # "xtick.bottom": True,
        # "ytick.left": True,
        "figure.constrained_layout.use": True,
    }

    return rc_params


def read_data(fname: str, path: str) -> Any:
    """
    Function that reads .pkl file from a
    a given folder.

    Args:
    ----
        fname (str):  file name
        path (str): path to folder

    Returns:
    -------
            loaded file.
    """
    # Load pickle data
    with open(os.path.join(path, fname), "rb") as fp:
        loaded_file = pickle.load(fp)

    return loaded_file


def dump_data(data: Any, fname: str, path: str) -> None:
    """
    Function that dumps a pickled data into
    a specified path

    Args:
    ----
        data (Any): data to be pickled
        fname (str):  file name
        path (str): path to folder

    Returns:
    -------
            None
    """
    with open(os.path.join(path, fname), "wb") as fp:
        pickle.dump(data, fp)

    return None


def load_yaml_file(path: str) -> dict[Any, Any]:
    with open(path, "r") as file:
        data = yaml.safe_load(file)

    return data


class CustomFormatter(logging.Formatter):
    purple = "\x1b[1;35m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s: [%(name)s] %(message)s"
    # "[%(asctime)s] (%(name)s) %(levelname)s: %(message)s"
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: purple + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(root_logger: str) -> logging.Logger:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())

    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler],
    )

    return logging.getLogger(root_logger)


def download_file(url: str, file_name: str, destination_folder: str = "data") -> None:
    response = requests.get(url)
    with open(f"./{destination_folder}/{file_name}", "wb") as file:
        file.write(response.content)
