import torch

import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from scipy.special import softmax
from runners.base_runner import BaseRunner, reduce_tensor, gather_tensor
from utils.metrics import calc_ece


class CnnRunner(BaseRunner):
    def __init__(
        self,
        loader,
        model,
        optim,
        lr_scheduler,
        num_epoch,
        loss_with_weight,
        val_metric,
        test_metric,
        logger,
        model_path,
        rank,
    ):
        self.num_epoch = num_epoch
        self.epoch = 0
        self.loss_with_weight = loss_with_weight
        self.val_metric = val_metric
        self.test_metric = test_metric
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.best_score = 0.0
        self.save_kwargs = {}
        self.world_size = torch.distributed.get_world_size()
        super().__init__(loader, model, logger, model_path, rank)
        self.load()
        self.num_classes = self.model.module.num_classes

    def _calc_loss(self, img, label):
        output = self.model(img.cuda(non_blocking=True))
        label = label.cuda(non_blocking=True)
        loss_ = 0
        for loss, w in self.loss_with_weight:
            _loss = w * loss(output, label)
            loss_ += _loss
        return loss_

    def _train_a_batch(self, batch):
        loss = self._calc_loss(*batch)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        _loss = reduce_tensor(loss, True).item()
        return _loss

    @torch.no_grad()
    def _valid_a_batch(self, img, label, with_output=False):
        output = self.model(img.cuda(non_blocking=True))
        label = label.cuda(non_blocking=True)
        result = self.val_metric(output, label)
        if with_output:
            result = [result, output]
        return result

    def train(self):
        self.log("Start to train", "debug")
        for epoch in range(self.epoch, self.num_epoch):
            self.model.train()
            loader = self.loader.load("train")
            # if self.rank == 0:
            #     t_iter = tqdm(loader, total=len(loader), desc=f"[Train {epoch}]")
            # else:
            t_iter = loader
            losses = 0
            for i, batch in enumerate(t_iter):
                loss = self._train_a_batch(batch)
                losses += loss
                # if self.rank == 0:
                # t_iter.set_postfix(loss=f"{loss:.4} / {losses/(i+1):.4}")

            self.lr_scheduler.step()
            acc_val = self.val(epoch, "val")
            acc_test = self.val(epoch, "test")
            self.log(
                f"Epoch:{epoch}\tLoss:{losses/i:.6f}\tACC_val:{acc_val:.2f}\tACC_test:{acc_test:.2f}",
                "info",
            )

    def val(self, epoch, data="val"):
        loader = self.loader.load(data)
        v_iter = loader

        metrics = np.zeros([self.num_classes, 4])
        self.model.eval()
        outputs = []
        labels = []
        for img, label in v_iter:
            _metric, output = self._valid_a_batch(img, label, with_output=True)
            outputs += gather_tensor(output)
            labels += gather_tensor(label.cuda())
            metrics += reduce_tensor(_metric, False).cpu().numpy()
        dice = [(i[0] * 2) / (i[2] + i[3] + 2 * i[0]) for i in metrics]
        output = torch.cat(outputs, 0).cpu().numpy()
        label = torch.cat(labels, 0).cpu().numpy()
        acc = (output.argmax(-1) == label).mean() * 100
        if self.rank == 0 and data == "test":
            self.save(epoch, acc, **self.save_kwargs)
        return acc

    def test(self, ckp="best.pth"):
        # self.load("model.pth")
        self.load(ckp)
        loader = self.loader.load("test")
        # if self.rank == 0:
        #     t_iter = tqdm(loader, total=len(loader))
        # else:
        t_iter = loader

        metrics = np.zeros([self.model.module.num_classes, 4])
        outputs = []
        labels = []
        self.model.eval()
        for i, (img, label) in enumerate(t_iter):
            _metric, output = self._valid_a_batch(img, label, with_output=True)
            outputs += gather_tensor(output)
            labels += gather_tensor(label.cuda())
            metrics += reduce_tensor(_metric, False).cpu().numpy()
        dice = [(i[0] * 2) / (i[2] + i[3] + 2 * i[0]) * 100 for i in metrics]
        output = torch.cat(outputs, 0).cpu().numpy()
        label = torch.cat(labels, 0).cpu().numpy()
        acc = (output.argmax(-1) == label).mean() * 100
        ece = calc_ece(softmax(output, -1), label) * 100
        self.log(f"[Test] Score: {acc:.2f}, {ece:.2f}", "info")
        self.log(np.round(dice), "info")
        output_df = pd.DataFrame(np.concatenate([output, label[:, None]], 1))
        output_df.to_csv(f"{self.model_path}/output.csv")
        return acc

    @torch.no_grad()
    def infer(self, infer_type="none"):
        self.load("best.pth")
        loader = self.loader.load("infer")
        i_iter = tqdm(loader, total=int(len(loader)))

        folder = Path(f"{self.model_path}/output")
        folder.mkdir(parents=True, exist_ok=True)
        self._infer_normal(i_iter, folder)

    @torch.no_grad()
    def _infer_normal(self, loader, folder):
        output_dict = {"name": [], "output": [], "x": [], "y": []}
        self.model.eval()
        for img, path, x, y in loader:
            output = self.model(img.cuda())
            output_dict["name"] += path
            output_dict["x"] += x
            output_dict["y"] += y
            output_dict["output"] += output.tolist()
        output_df = pd.DataFrame(output_dict)
        output_df.to_csv(f"{self.model_path}/infer.csv")

    def save(self, epoch, metric, file_name="model", **kwargs):
        torch.save(
            {
                "epoch": epoch,
                "param": self.model.state_dict(),
                "optimizer": self.optim.state_dict(),
                "score": metric,
                "best": self.best_score,
                "lr_schdlr": self.lr_scheduler.state_dict(),
                **kwargs,
            },
            f"{self.model_path}/{file_name}.pth",
        )

        cond = metric >= self.best_score
        if cond:
            # self.log(f"{self.best_score} -------------------> {metric}", "debug")
            self.best_score = metric
            shutil.copy2(
                f"{self.model_path}/{file_name}.pth", f"{self.model_path}/best.pth"
            )
            print("-------------------------------------------------")
            # self.log(f"Model has saved at {epoch} epoch.", "debug")

    def load(self, file_name="model.pth"):
        self.log(self.model_path, "debug")
        if (self.model_path / file_name).exists():
            self.log(f"Loading {self.model_path} File", "debug")
            ckpoint = torch.load(
                f"{self.model_path}/{file_name}", map_location="cpu")

            for key, value in ckpoint.items():
                if key == "param":
                    self.model.load_state_dict(value)
                elif key == "optimizer":
                    self.optim.load_state_dict(value)
                elif key == "lr_schdlr":
                    self.lr_scheduler.load_state_dict(value)
                elif key == "epoch":
                    self.epoch = value + 1
                elif key == "best":
                    self.best_score = value
                else:
                    self.__dict__[key] = value

            self.log(
                f"Model Type : {file_name}, epoch : {self.epoch}", "debug")
        else:
            self.log("Failed to load, not existing file", "debug")

    def get_lr(self):
        return self.lr_scheduler.optimizer.param_groups[0]["lr"]
