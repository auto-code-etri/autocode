import glob
import logging
import math
import os
import random
from typing import *

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as torch_utils
import yaml
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AdamW, get_cosine_schedule_with_warmup

from utils import AverageMeter


class Trainer:
    def __init__(self, hparams, loaders, model, resultwriter, pad_idx):
        self.hparams = hparams
        self.rank: int = self.hparams.rank
        self.main_process: bool = self.rank in [-1, 0]
        self.nprocs: int = torch.cuda.device_count()
        self.scaler = torch.cuda.amp.GradScaler() if self.hparams.amp else None
        if self.hparams.distributed:
            assert torch.cuda.is_available()
            self.device = f"cuda:{self.rank}"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        self.model = model.to(self.device, non_blocking=True)
        if self.hparams.distributed:
            self.model = DDP(self.model, device_ids=[self.rank])
        elif self.nprocs > 1:
            self.model = nn.DataParallel(self.model)
        self.max_grad_norm = self.hparams.max_grad_norm
        self.gradient_accumulation_step = self.hparams.gradient_accumulation_step

        # dataloader and distributed sampler
        self.train_loader, self.valid_loader = loaders
        self.train_sampler = self.train_loader.sampler

        # optimizer, scheduler
        self.step_total = (
            len(self.train_loader)
            // self.gradient_accumulation_step
            * self.hparams.epoch
        )
        self.optimizer, self.scheduler = self.configure_optimizers()

        # metric
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        # model saving options
        self.global_step = 0
        self.eval_step = (
            int(self.step_total * hparams.eval_ratio)
            if hparams.eval_ratio > 0
            else self.step_total // hparams.epoch
        )
        if self.main_process:
            self.version = 0
            while True:
                self.save_path = os.path.join(
                    hparams.ckpt_path, f"version-{self.version}"
                )
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                    break
                else:
                    self.version += 1
            self.summarywriter = SummaryWriter(self.save_path)
            self.global_val_loss = float("inf")
            with open(
                os.path.join(self.save_path, "hparams.yaml"), "w", encoding="utf8"
            ) as outfile:
                yaml.dump(
                    hparams, outfile, default_flow_style=False, allow_unicode=True
                )

            # experiment logging options
            self.best_result = {"version": self.version}
            self.log_step = hparams.log_step
            logging.basicConfig(
                filename=os.path.join(self.save_path, "experiment.log"),
                level=logging.INFO,
                format="%(asctime)s > %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S %Z",
            )
            logging.info(
                f"[SCHEDULER] Total_step: {self.step_total} | Warmup step: {self.warmup_steps} | Accumulation step: {self.gradient_accumulation_step}"
            )

    def configure_optimizers(self):
        # optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # lr warmup scheduler
        self.warmup_steps = math.ceil(self.step_total * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.step_total,
        )

        return optimizer, scheduler

    def get_parameter_names(self, model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in self.get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result

    def save_checkpoint(self, epoch: int, val_loss: float, model: nn.Module) -> None:
        logging.info(
            f"      Val loss decreased ({self.global_val_loss:.4f} → {val_loss:.4f}). Saving model ..."
        )
        new_path = os.path.join(
            self.save_path, f"best_model_step_{self.global_step}_loss_{val_loss:.4f}.pt"
        )

        for filename in glob.glob(os.path.join(self.save_path, "*.pt")):
            os.remove(filename)  # remove old checkpoint
        torch.save(model.state_dict(), new_path)
        self.global_val_loss = val_loss

    def fit(self) -> dict:
        # this zero gradient update is needed to avoid a warning message in warmup setting
        self.optimizer.zero_grad()
        self.optimizer.step()

        for epoch in tqdm(
            range(self.hparams.epoch), desc="epoch", disable=not self.main_process
        ):
            if self.hparams.distributed:
                self.train_sampler.set_epoch(epoch)
            self._train_epoch(epoch)

        if self.main_process:
            self.summarywriter.close()
        return self.best_result if self.main_process else None

    def _train_epoch(self, epoch: int) -> None:
        train_loss = AverageMeter()

        for step, batch in tqdm(
            enumerate(self.train_loader),
            desc="trn_steps",
            total=len(self.train_loader),
            disable=not self.main_process,
        ):
            """
            Dimensions:
                src: [batch_size, max_len]
                tgt_input: [batch_size, max_len]
                tgt_label: [batch_size, max_len]
                src_mask: [batch_size, 1, max_len]
                tgt_mask: [batch_size, 1, max_len, max_len]
            """
            self.model.train()
            src, tgt_input, tgt_label, src_mask, tgt_mask = map(
                lambda x: x.to(self.device, non_blocking=True), batch
            )

            # compute loss
            if self.hparams.amp:
                with torch.cuda.amp.autocast():
                    logit = self.model(src, tgt_input, src_mask, tgt_mask)
                    loss = self.criterion(
                        logit.contiguous().view(-1, logit.shape[-1]),
                        tgt_label.contiguous().view(-1),
                    )
                    loss *= tgt_mask
            else:
                logit = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(
                    logit.contiguous().view(-1, logit.shape[-1]),
                    tgt_label.contiguous().view(-1),
                )
                loss *= tgt_mask
            loss = loss.sum() / self.gradient_accumulation_step

            # update
            if self.hparams.amp:
                self.scaler.scale(loss).backward()
                if (step + 1) % self.gradient_accumulation_step == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch_utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scheduler.step()
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:
                loss.backward()
                if (step + 1) % self.gradient_accumulation_step == 0:
                    torch_utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

            train_loss.update(loss.item())
            if (step + 1) % self.gradient_accumulation_step != 0:
                continue

            # validate and logging
            if (self.global_step + 1) % self.eval_step == 0:
                val_loss = self.validate(epoch)
                if self.main_process:
                    self.summarywriter.add_scalars(
                        "loss/step", {"val": val_loss}, self.global_step
                    )
                    logging.info(
                        f"[VAL] global step: {self.global_step} | val loss: {val_loss:.4f}"
                    )
                    if val_loss < self.global_val_loss:
                        self.save_checkpoint(epoch, val_loss, self.model)

            # train logging
            if self.main_process:
                if self.global_step != 0 and self.global_step % self.log_step == 0:
                    logging.info(
                        f"[TRN] Version: {self.version} | Epoch: {epoch} | Global step: {self.global_step} | Train loss: {loss.item():.3f} | LR: {self.optimizer.param_groups[0]['lr']:.5f}"
                    )
                    self.summarywriter.add_scalars(
                        "loss/step", {"train": train_loss.avg}, self.global_step
                    )
                    self.summarywriter.add_scalars(
                        "lr",
                        {"lr": self.optimizer.param_groups[0]["lr"]},
                        self.global_step,
                    )

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        val_loss = AverageMeter()

        self.model.eval()
        for step, batch in tqdm(
            enumerate(self.valid_loader),
            desc="val_steps",
            total=len(self.valid_loader),
            disable=not self.main_process,
        ):
            src, tgt_input, tgt_label, src_mask, tgt_mask = map(
                lambda x: x.to(self.device, non_blocking=True), batch
            )

            # compute loss
            logit = self.model(src, tgt_input, src_mask, tgt_mask)
            loss = self.criterion(
                logit.contiguous().view(-1, logit.shape[-1]),
                tgt_label.contiguous().view(-1),
            )
            val_loss.update(loss.item())

        return val_loss.avg

    @torch.no_grad()
    def test(self, test_loader, state_dict) -> dict:
        test_loss = AverageMeter()

        self.model.load_state_dict(state_dict)
        self.model.eval()
        for step, batch in tqdm(
            enumerate(test_loader), desc="tst_steps", total=len(test_loader)
        ):
            src, tgt_input, tgt_label, src_mask, tgt_mask = map(
                lambda x: x.to(self.device, non_blocking=True), batch
            )

            # compute loss (should use '.module' here when using ddp)
            # https://discuss.pytorch.org/t/torch-distributed-barrier-hangs-in-ddp/114522/8
            logit = self.model.module(src, tgt_input, src_mask, tgt_mask)
            loss = self.criterion(
                logit.contiguous().view(-1, logit.shape[-1]),
                tgt_label.contiguous().view(-1),
            )
            test_loss.update(loss.item())

        logging.info(f"[TST] Test Loss: {test_loss.avg:.4f}")
        return {"test_loss": test_loss.avg}
