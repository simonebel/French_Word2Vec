from conf.base import dir

from utils.log import get_log


def init_experiment(cfg):
    log_path = dir.joinpath("logs")
    log_path.mkdir(parents=True, exist_ok=True)

    dirs = [path.name for path in log_path.iterdir()]
    exp_number = max([int(dir) for dir in dirs]) + 1 if dirs else 0

    logger = get_log(str(exp_number))

    return logger, exp_number
