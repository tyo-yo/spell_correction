import json
from pathlib import Path
from typing import List, Dict, Any

import comet_ml
from allennlp.data.dataloader import TensorDict
from allennlp.models.archival import CONFIG_NAME
from allennlp.training import TrainerCallback
from flatten_dict import flatten

@TrainerCallback.register("log_to_comet")
class LogToComet(TrainerCallback):
    def __init__(
        self,
        project_name: str = None,
        upload_serialization_dir: bool = True,
        log_interval: int = 100,
        log_batch_output: bool = True,
        send_notification: bool = True
    ):
        self._project_name = project_name
        self._experiment = comet_ml.Experiment(project_name=self._project_name)
        self.upload_serialization_dir = upload_serialization_dir
        self.log_interval = log_interval
        self.send_notification = send_notification

    def on_batch(
        self,
        trainer: "GradientDescentTrainer",
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_master: bool
    ):
        if self.log_interval <= 0:
            return
        elif batch_number % self.log_interval == 0:
            for key, val in batch_metrics.items():
                self._experiment.log_metric(
                    f"{key}",
                    val,
                    epoch=epoch + 1,
                    step=batch_number
                )

    def on_end(self, trainer, metrics, epoch, is_master):
        self._experiment.add_tag("COMPLETED")
        if self.upload_serialization_dir:
            self._experiment.log_model("serialization_dir", trainer._serialization_dir)
        if self.send_notification:
            self._experiment.send_notification('Training Finished!', status='COMPLETED')

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
