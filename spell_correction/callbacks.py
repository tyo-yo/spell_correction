import json
from pathlib import Path
from typing import Text

import comet_ml
from allennlp.models.archival import CONFIG_NAME
from allennlp.training import TrainerCallback
from flatten_dict import flatten


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
        if epoch == -1:
            with (trainer._serialization_dir / Path(CONFIG_NAME)).open(
                encoding="utf-8"
            ) as f:
                config_dict = json.load(f)
            for key, val in flatten(config_dict).items():
                self._experiment.log_parameter(key, val)
        elif epoch >= 0:
            for key, val in metrics.items():
                self._experiment.log_metric(f"{key}", val, epoch=epoch + 1)
