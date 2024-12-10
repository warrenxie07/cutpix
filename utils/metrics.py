import torch
import numpy as np
import contextlib
from torch import nn


class NbsLoss(torch.nn.Module):
    def __init__(
        self, reduction="mean", base_loss=torch.nn.CrossEntropyLoss(reduction="none")
    ):
        super().__init__()
        self.reduction = reduction
        self.base_loss = base_loss

    def forward(self, input, target, w=None):
        out = self.base_loss(input, target)
        if w is not None:
            out = out * w
        if self.reduction == "mean":
            return out.mean()
        elif self.reduction == "sum":
            return out.sum()
        else:
            return out


class DiceLoss:
    def __init__(self, n_labels=5, smooth=1e-10, eps=1e-8):
        self.n_labels = n_labels
        self.smooth = smooth
        self.eps = eps

    def __call__(self, input_, target_):
        # print(input_.shape, target_.shape)
        # assert (input_.dim() == 5 and target_.dim() == 5)

        # masking
        dices_per_label = []
        for l in range(0, self.n_labels):
            dices = []
            for n in range(input_.size(0)):
                label_p = target_[n].eq(l + 1).float()
                prob_l = input_[n, l, :, :, :]
                prob_l = torch.clamp(prob_l, self.eps, 1 - self.eps)

                overlap = (prob_l * label_p).sum() + self.smooth
                pixel_sum = (label_p ** 2).sum() + (prob_l ** 2).sum()
                jacc = 1.0 - torch.clamp(overlap / (pixel_sum - overlap), 0, 1)
                dices.append(jacc.view(-1))

            dices_per_label.append(torch.mean(torch.cat(dices)).unsqueeze(0))
        if len(dices_per_label) > 1:
            dices_per_label = torch.cat(dices_per_label)
        else:
            dices_per_label = dices_per_label[0]
        return torch.mean(dices_per_label)


class Accuracy(torch.nn.Module):
    def __init__(self, reduction="mean", nlabels=5):
        super().__init__()
        self.reduction = reduction
        self.nlabels = nlabels

    def forward(self, input, target):
        if self.nlabels == 1:
            pred = input.sigmoid().gt(0.5).type_as(target)
        else:
            pred = input.argmax(1)
        acc = pred == target
        if self.reduction == "mean":
            acc = acc.float().mean()
        elif self.reduction == "sum":
            acc = acc.float().sum()
        return acc


class ConfusionMatrix(torch.nn.Module):
    def __init__(self, nlabels=5):
        super().__init__()
        self.nlabels = nlabels

    def forward(self, input, target):
        if self.nlabels == 1:
            pred = input.sigmoid().gt(0.5).type_as(target)
        else:
            pred = input.argmax(1)

        cm = torch.zeros([self.nlabels, 4]).cuda()
        for l in range(self.nlabels):
            if self.nlabels == 1:
                _pred = pred.eq(1).float()
                _label = target.eq(l).float()
            else:
                _pred = pred.eq(l).float()
                _label = target.eq(l).float()

            _cm = _pred * 2 - _label
            tp = _cm.eq(1).float().sum()
            tn = _cm.eq(0).float().sum()
            fp = _cm.eq(2).float().sum()
            fn = _cm.eq(-1).float().sum()

            for j, j_ in zip(cm[l], [tp, tn, fp, fn]):
                j += j_

        return cm


class Dice(torch.nn.Module):
    def __init__(self, reduction="mean", nlabels=5, eps=0.001, **kargs):
        super().__init__()
        self.reduction = reduction
        self.nlabels = nlabels
        self.eps = eps
        self.kargs = kargs

    def forward(self, input, target):
        if self.nlabels == 1:
            pred = input.sigmoid().gt(0.5).type_as(target)
        else:
            pred = input.argmax(1)

        dices = []
        for l in range(self.nlabels):
            if self.nlabels == 1:
                _pred = pred.eq(1).float()
                _label = target
            else:
                _pred = pred.eq(l).float()
                _label = target.eq(l).float()

            cm = _pred * 2 - _label
            dims = list(set(range(target.dim())) - set([0]))
            tp = cm.eq(1).float().sum(dim=dims)
            fp = cm.eq(2).float().sum(dim=dims)
            fn = cm.eq(-1).float().sum(dim=dims)
            dice = (2 * tp + self.eps) / (fn + fp + 2 * tp + self.eps)
            dices += [dice[None, ...]]

        dices = torch.cat(dices, dim=0)
        if "weighted" in self.reduction:
            weights = np.array(
                self.kargs["weights"].split(","), dtype=np.float32)
            assert (
                len(weights) == self.nlabels
            ), "The number of 'weights' should be same with 'nlabels'."
            dices = dices * torch.tensor(weights)[..., None].cuda()
        elif "index" in self.reduction:
            idx = self.kargs["index"]
            assert self.nlabels > idx, "The 'index' should be less than 'nlabels'."
            dices = dices[idx]

        if "mean" in self.reduction:
            dices = dices.mean()
        elif "sum" in self.reduction:
            dices = dices.sum()
        return dices


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, "track_running_stats"):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


class VATLoss(nn.Module):
    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()

            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction="batchmean")

        return lds


class CrossEntropyWithSoftLabel(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):
        log_probs = self.logsoftmax(input)
        loss = (-target * log_probs).sum(dim=1)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


def calc_ece(softmax, label, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    labels = torch.tensor(label)

    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels)

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * \
            softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin -
                             accuracy_in_bin) * prop_in_bin

    # print("ECE {0:.2f} ".format(ece.item() * 100))

    return ece.item()


if __name__ == "__main__":
    Acc = Accuracy()
    # a = torch.rand(8, 5, 64, 256, 256).float()
    # b = torch.randint(5, [8, 64, 256, 256])
    a = torch.rand(1, 3, 5)
    b = torch.randint(3, (1, 5))
    print(a)
    print(a.argmax(1))
    print(b)
    # print(Acc(a, b))

    dice = Dice(reduction="weighted_mean", nlabels=3, weights="1,1,1")
    # dice = Dice(reduction='index', index=0)
    # dice = Dice()
    print(dice(a, b))
