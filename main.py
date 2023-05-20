import os
import time
import datetime
import pickle
import yaml
import threading
import logging
import sys

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.server import Server
from src.utils import launch_tensor_board

if __name__ == "__main__":
    # read configuration file
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'

    with open(config_path) as c:
        configs = list(yaml.load_all(c, Loader=yaml.FullLoader))
    global_config = configs[0]["global_config"]
    data_config = configs[1]["data_config"]
    fed_config = configs[2]["fed_config"]
    init_config = configs[3]["init_config"]
    model_config = configs[4]["model_config"]
    log_config = configs[5]["log_config"]

    # modify log_path to contain current time
    log_config["log_path"] = os.path.join(log_config["log_path"],
                                          str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))

    # initiate TensorBoard for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_config["log_path"], filename_suffix="FL")
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([log_config["log_path"], log_config["tb_port"], log_config["tb_host"]])
    ).start()
    time.sleep(3.0)

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_config["log_path"], log_config["log_name"]),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")

    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message);
    logging.info(message)

    for config in configs:
        print(config);
        logging.info(config)
    print()

    # initialize federated learning
    central_server = Server(writer, model_config, global_config, data_config, init_config, fed_config)
    central_server.setup()

    # do federated learning
    central_server.fit()

    # save resulting losses and metrics
    with open(os.path.join(log_config["log_path"], "result.pkl"), "wb") as f:
        pickle.dump(central_server.results, f)

    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message)
    logging.info(message)
    time.sleep(3)
    print(f'global_config= {global_config}')
    print(f'data_config= {data_config}')
    print(f'log_config= {log_config}')
    central_server.print_best_results()

