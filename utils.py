import logging
import os
import time

import yaml

LOG_DIR = os.path.join("logs")
LOG_FORMAT = "%(levelname)s %(name)s %(asctime)s - %(message)s"

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)


def get_logger(logger_name, log_path, log_level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    format = logging.Formatter(LOG_FORMAT)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(format)

    logger.addHandler(file_handler)

    return logger


def load_yaml(field, path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)
    return config[field]


def get_output_paths():
    if is_dryrun():
        output_dir = load_yaml("dryrun_output_dir")
    else:
        output_dir = load_yaml("output_dir")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_filenames = load_yaml("output_filenames")

    paths = {}
    for key, filename in output_filenames.items():
        paths[key] = os.path.join(output_dir, filename)

    return paths


def timing(method):
    def timed(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()

        execution_time = end - start
        if execution_time < 0.001:
            print(
                f"{method.__name__} took {round(execution_time * 1000, 3)} "
                f"milliseconds"
            )
        else:
            print(f"{method.__name__} took {round(execution_time, 3)} seconds")

        return result

    return timed


def is_dryrun():
    mode = load_yaml("dryrun_mode")
    if mode == 1:
        return True
    return False
