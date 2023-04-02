import logging

from conf.base import dir


def get_log(exp_numbers: str):
    log_path = dir.joinpath("logs")

    exp_folder = log_path.joinpath(exp_numbers)
    exp_folder.mkdir(exist_ok=True, parents=True)
    filepath = exp_folder.joinpath("train.log")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s: %(message)s")

    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
