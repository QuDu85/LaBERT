import logging
import os

import torch


class Checkpointer(object):

    def __init__(self, model, optimizer=None, scheduler=None,
                 save_dir="", save_to_disk=None, logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger("Checkpointer")
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return
        if not self.save_to_disk:
            return

        data = {}
        if self.model is not None:
            data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, f=None, model_only=False):
        if f is not None:
            if os.path.exists(f):
                self.logger.info("Loading checkpoint from {}".format(f))
                checkpoint = torch.load(f, map_location=torch.device("cpu"))

                if "model" in checkpoint and self.model:
                    model_dict = self.model.state_dict()
                    # 1. filter out unnecessary keys
                    pretrained_dict = {k: v for k, v in checkpoint.pop("model").items() if k in model_dict}
                    # 2. overwrite entries in the existing state dict
                    model_dict.update(pretrained_dict)
                    # 3. load the new state dict
                    self.model.load_state_dict(pretrained_dict, strict=False)
                    # self.model.load_state_dict(checkpoint.pop("model"), strict=False)
                if model_only:
                    checkpoint.pop("optimizer", None)
                    checkpoint.pop("scheduler", None)
                    return checkpoint

                if "optimizer" in checkpoint and self.optimizer:
                    self.logger.info("Loading optimizer from {}".format(f))
                    try:
                        self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
                    except ValueError:
                        print('Failed to load optimizer due to param difference')
                if "scheduler" in checkpoint and self.scheduler:
                    self.logger.info("Loading scheduler from {}".format(f))
                    self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

                return checkpoint
            else:
                self.logger.info("No checkpoint found.")
        else:
            self.logger.info("No checkpoint provided.")
        return {}
