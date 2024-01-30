import torch
import math
import numpy as np
import torch.optim as optim
import torch.autograd as autograd
import torchvision
from torchvision.utils import save_image
from typing import Callable, Optional
from pytorch_lightning.core.optimizer import LightningOptimizer

from src.objectives.simclr import SimCLR, TaU_SimCLR
from src.systems.base import PretrainSystem
from torch.optim.optimizer import Optimizer
from src.scheduler.lars import LARSWrapper


class SimCLRSystem(PretrainSystem):

    def get_loss(self, batch, train=True, **kwargs):
        _, image1, image2, _ = batch
        outputs1 = self.forward(image1)
        outputs2 = self.forward(image2)
        loss = SimCLR(outputs1, outputs2, t=self.config.loss.t).get_loss()
        return loss

    def get_lr_schedule(self):
        batch_size = self.config.optimizer.batch_size
        iters_per_epoch = len(self.train_dataset) // batch_size
        start_lr = self.config.optimizer.start_lr
        final_lr = self.config.optimizer.final_lr
        learning_rate = self.config.optimizer.learning_rate
        warmup_epochs = self.config.optimizer.warmup_epochs
        max_epochs = self.config.num_epochs

        warmup_lr_schedule = np.linspace(start_lr, learning_rate, iters_per_epoch * warmup_epochs)
        iters = np.arange(iters_per_epoch * (max_epochs - warmup_epochs))
        cosine_lr_schedule = np.array([
            final_lr + 0.5 * (learning_rate - final_lr) *
            (1 + math.cos(math.pi * t / (iters_per_epoch * (max_epochs - warmup_epochs))))
            for t in iters
        ])
        lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        return lr_schedule

    def configure_optimizers(self):
        self.lr_schedule = self.get_lr_schedule()  # make lr schedule
        weight_decay = self.config.optimizer.weight_decay
        exclude_bn_bias = self.config.optimizer.exclude_bn_bias
        learning_rate = self.config.optimizer.learning_rate

        if exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=weight_decay)
        else:
            params = self.parameters()

        if self.config.optimizer.name == 'sgd':
            optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        elif self.config.optimizer.name == 'adam':
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        else:
            raise Exception(f'Optimizer {self.config.optimizer.name} not supported.')

        optimizer = LARSWrapper(optimizer, eta=0.001, clip=False)
        return [optimizer], []

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer: Optimizer = None,
        optimizer_idx: int = None,
        optimizer_closure: Optional[Callable] = None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
    ) -> None:
        # warm-up + decay schedule placed here since LARSWrapper is not optimizer class
        # adjust LR of optim contained within LARSWrapper
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr_schedule[self.trainer.global_step]

        if not isinstance(optimizer, LightningOptimizer):
            optimizer = LightningOptimizer.to_lightning_optimizer(optimizer, self.trainer)
        optimizer.step(closure=optimizer_closure)

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=['bias', 'bn']):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{
            'params': params,
            'weight_decay': weight_decay
        }, {
            'params': excluded_params,
            'weight_decay': 0.,
        }]
    
    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, train=True)
        metrics = {'train_loss': loss, 'learning_rate': self.lr_schedule[self.trainer.global_step]}
        self.log_dict(metrics)
        return loss


class TaU_SimCLRSystem(SimCLRSystem):

    def get_loss(self, batch, train=True):
        _, image1, image2, _ = batch
        loc1, temp1 = self.forward(image1)
        loc2, temp2 = self.forward(image2)
        loss = TaU_SimCLR(loc1, temp1, loc2, temp2,
                          t=self.config.loss.t,
                          eps=self.config.loss.eps,
                          simclr_mask=self.config.loss.simclr_mask).get_loss()
        return loss


