import os
import socket
from typing import Text

import comet_ml
import toml
from allennlp.training import TrainerCallback


@TrainerCallback.register("log_to_comet")
class LogToComet(TrainerCallback):
    def __init__(self, project_name: Text = None):
        self._project_name = project_name
        # model_config_file = os.environ.get("MODEL_CONFIG_FILE")

        self._experiment = comet_ml.Experiment(project_name=self._project_name)
        # slurm_log_file = os.environ.get("SLURM_LOG_FILE")
        # if slurm_log_file is not None:
        # self._experiment.log_asset(slurm_log_file, overwrite=True)
        # model_config_file = os.environ.get("MODEL_CONFIG_FILE")
        # if model_config_file is not None:
        #     self._experiment.log_asset(model_config_file)
        #     with open(model_config_file) as f:
        #         self._conf = toml.load(f)
        #     for key, val in self._conf["params"].items():
        #         self._experiment.log_parameter(key, val)
        #     self._experiment.add_tag(self._conf["name"])
        # self._experiment.log_other("hostname", socket.gethostname())

    def on_end(self, trainer, metrics, epoch, is_master):
        self._experiment.add_tag("COMPLETED")

    def on_epoch(self, trainer, metrics, epoch, is_master):
        if epoch >= 0:
            for key, val in metrics.items():
                self._experiment.log_metric(f"{key}", val, epoch=epoch + 1)
