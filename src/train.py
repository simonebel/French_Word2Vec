import json
import multiprocessing as mp
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path

import gensim.models
from gensim import utils
from gensim.test.utils import datapath
from omegaconf import OmegaConf

from data.load import load_data
from utils.utils import init_experiment


def main(cfg):
    logger, exp_number = init_experiment(cfg)

    logger.info("Loading datasets")
    corpus = load_data(cfg.datasets.datasets, cfg.data_dir)

    logger.info("Start Training")
    params = cfg.params
    model = gensim.models.Word2Vec(
        sentences=corpus,
        min_count=params.min_count,
        vector_size=params.vector_size,
        workers=mp.cpu_count(),
        window=params.window,
        epochs=params.epochs,
        seed=params.seed,
        sg=params.sg,
    )
    ckpt_dir = Path(cfg.data_dir, "ckpt", str(exp_number))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Train finished, save model into {str(ckpt_dir)}")

    model.save(
        str(ckpt_dir.joinpath(f"fr_w2v_fl_{params.vector_size}_w{params.window}"))
    )

    with open(ckpt_dir.joinpath("config.json"), "w") as f:
        json.dump(dict(params), f)


def set_cfg(args: Namespace):
    """
    Set the configuration object
    """
    cfg = OmegaConf.create(vars(args))
    cfg.datasets = OmegaConf.load("conf/dataset.yaml")
    cfg.params = OmegaConf.load("conf/params.yaml")
    return cfg


if __name__ == "__main__":
    parser = ArgumentParser(
        description="This script is used to generate the list of ids of the insee publication to scrap. "
    )

    parser.add_argument(
        "--log_path", type=str, default="./log/", help="Experiment log path"
    )

    parser.add_argument(
        "--dev",
        type=int,
        help="Whether to run the script in dev mode",
    )

    parser.add_argument(
        "--data-dir",
        help="The path to the data directory",
    )

    args = parser.parse_args()
    cfg = set_cfg(args)

    main(cfg)
