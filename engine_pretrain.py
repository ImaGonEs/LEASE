import contextlib
import math
import sys
import time
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

import torch.distributed as dist

import numpy as np


class IterationTimer:
    """Measures per-iteration wall time over a fixed window, logs once, then becomes a no-op."""

    def __init__(self, warmup: int = 10, measure: int = 1000):
        self.warmup = warmup
        self.measure = measure
        self._times = []
        self._start = None
        self._done = False

    def start(self):
        if self._done:
            return
        torch.cuda.synchronize()
        self._start = time.perf_counter()

    def stop(self, log_writer=None, step: int = None):
        if self._done:
            return
        torch.cuda.synchronize()
        self._times.append(time.perf_counter() - self._start)
        if len(self._times) >= self.warmup + self.measure:
            self._report(log_writer, step)
            self._done = True

    def _report(self, log_writer, step):
        t = np.array(self._times[self.warmup:]) * 1000  # ms
        print(f"[IterTimer] mean: {t.mean():.2f} ms  std: {t.std():.2f} ms")
        if log_writer is not None and step is not None:
            log_writer.add_scalar("stats/iter_time_ms", t.mean(), step)


def train_one_epoch_two_losses_tk(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('rec_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('nn_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_loss', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('train_acc', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    timer = IterationTimer(warmup=10, measure=1000)

    for data_iter_step, (vq_idx, labels, *_) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # ── Optimization 3 (cont): signal cudagraph step boundary ──
        # Required for cudagraph replay under reduce-overhead mode.
        torch.compiler.cudagraph_mark_step_begin()

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        timer.start()

        with torch.cuda.amp.autocast():
            loss, rec_loss, nn_loss, _, train_acc, class_loss, latents = model(vq_idx, labels, epoch)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.grad_clip, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        timer.stop(log_writer=log_writer, step=data_iter_step)

        metric_logger.update(loss=loss_value)
        metric_logger.update(rec_loss=rec_loss.item())
        metric_logger.update(nn_loss=nn_loss.item())
        metric_logger.update(train_acc=train_acc.item())
        metric_logger.update(class_loss=class_loss.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        rec_loss_value_reduce = misc.all_reduce_mean(rec_loss.item())
        nn_loss_value_reduce = misc.all_reduce_mean(nn_loss.item())
        train_acc_value_reduce = misc.all_reduce_mean(train_acc.item())
        class_loss_value_reduce = misc.all_reduce_mean(class_loss.item())

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('rec_loss', rec_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('nn_loss', nn_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_acc', train_acc_value_reduce, epoch_1000x)
            log_writer.add_scalar('class_loss', class_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_lease(
    model: torch.nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,   # kept for signature compatibility
    epoch: int,
    loss_scaler,
    log_writer=None,
    args=None,
):
    """
    model(vq_idx, dino_idx, labels, epoch) -> (
            total_loss, rec_loss_vq, mask_full_vq, accuracy, classifier_loss, x_with_cls
        )

    Logs:
      * loss (total), rec_loss_vq
      * info_nce_loss (model.debug["info_nce_loss"])
      * masked_dino_ce (model.debug["masked_dino_ce"])
      * class_loss (returned by model)
      * train_acc_cls
      * frac_masked (mean of mask_full[:, 1:])
      * frac_dropped (=0.5 fixed by design: keep 128 of 256)
      * feat_norm (||x|| mean over tokens != CLS)
    """
    model.train(True)

    def _tensor_or_zero(x, dev):
        if isinstance(x, torch.Tensor):
            return x.to(dev)
        try:
            return torch.tensor(float(x), device=dev)
        except Exception:
            return torch.tensor(0.0, device=dev)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr",                misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss",              misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("rec_loss_vq",       misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("info_nce_loss",     misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("masked_dino_ce",    misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_loss",        misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("train_acc_cls",     misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("frac_masked",       misc.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("frac_dropped",      misc.SmoothedValue(window_size=1, fmt="{value:.4f}"))
    metric_logger.add_meter("feat_norm",         misc.SmoothedValue(window_size=1, fmt="{value:.4f}"))

    header = f"Epoch: [{epoch}]"
    print_freq = 20
    accum_iter = getattr(args, "accum_iter", 1)

    optimizer.zero_grad(set_to_none=True)

    if log_writer is not None:
        print(f"log_dir: {log_writer.log_dir}")

    model_device = next(model.parameters()).device

    timer = IterationTimer(warmup=10, measure=1000)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        torch.compiler.cudagraph_mark_step_begin()
        vq_idx, labels, dino_idx, *extras = batch

        if vq_idx.device != model_device:
            vq_idx = vq_idx.to(model_device, non_blocking=(vq_idx.device.type == "cpu"))
        if dino_idx.device != model_device:
            dino_idx = dino_idx.to(model_device, non_blocking=(dino_idx.device.type == "cpu"))
        if labels.device != model_device:
            labels = labels.to(model_device, non_blocking=(labels.device.type == "cpu"))
        labels = labels.long().view(-1)

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        timer.start()

        use_amp = torch.cuda.is_available()
        amp_ctx = torch.cuda.amp.autocast() if use_amp else contextlib.nullcontext()
        with amp_ctx:
            (
                total_loss,
                rec_loss_vq,
                mask_full_vq,
                acc_main,
                classifier_loss,
                x_enc,
            ) = model(vq_idx, dino_idx, labels, epoch)

        inner = model.module if hasattr(model, "module") else model
        dbg = getattr(inner, "debug", {}) or {}
        info_nce_loss  = _tensor_or_zero(dbg.get("info_nce_loss", 0.0), model_device)
        masked_dino_ce = _tensor_or_zero(dbg.get("masked_dino_ce", 0.0), model_device)

        with torch.no_grad():
            frac_masked = mask_full_vq[:, 1:].float().mean() if isinstance(mask_full_vq, torch.Tensor) else torch.tensor(0.0, device=model_device)
            frac_dropped = torch.tensor(0.5, device=model_device)
            feat_norm = x_enc[:, 1:, :].norm(dim=-1).mean() if isinstance(x_enc, torch.Tensor) else torch.tensor(0.0, device=model_device)

        loss_value = float(total_loss.item())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Non-finite loss: {loss_value}")

        total_loss = total_loss / accum_iter
        loss_scaler(
            total_loss,
            optimizer,
            clip_grad=getattr(args, "grad_clip", None),
            parameters=model.parameters(),
            update_grad=((data_iter_step + 1) % accum_iter == 0),
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad(set_to_none=True)

        timer.stop(log_writer=log_writer, step=data_iter_step)

        metric_logger.update(loss=loss_value)
        metric_logger.update(rec_loss_vq=float(rec_loss_vq.item()))
        metric_logger.update(info_nce_loss=float(info_nce_loss.item() if isinstance(info_nce_loss, torch.Tensor) else info_nce_loss))
        metric_logger.update(masked_dino_ce=float(masked_dino_ce.item() if isinstance(masked_dino_ce, torch.Tensor) else masked_dino_ce))
        metric_logger.update(class_loss=float(classifier_loss.item() if isinstance(classifier_loss, torch.Tensor) else classifier_loss))
        metric_logger.update(train_acc_cls=float(acc_main.item()))
        metric_logger.update(frac_masked=float(frac_masked.item()))
        metric_logger.update(frac_dropped=float(frac_dropped.item()))
        metric_logger.update(feat_norm=float(feat_norm.item()))

        loss_value_reduce     = misc.all_reduce_mean(loss_value)
        rec_loss_vq_reduce    = misc.all_reduce_mean(float(rec_loss_vq.item()))
        info_nce_loss_reduce  = misc.all_reduce_mean(float(info_nce_loss.item() if isinstance(info_nce_loss, torch.Tensor) else info_nce_loss))
        masked_dino_ce_reduce = misc.all_reduce_mean(float(masked_dino_ce.item() if isinstance(masked_dino_ce, torch.Tensor) else masked_dino_ce))
        class_loss_reduce     = misc.all_reduce_mean(float(classifier_loss.item() if isinstance(classifier_loss, torch.Tensor) else classifier_loss))
        train_acc_cls_reduce  = misc.all_reduce_mean(float(acc_main.item()))
        frac_masked_reduce    = misc.all_reduce_mean(float(frac_masked.item()))
        frac_dropped_reduce   = misc.all_reduce_mean(float(frac_dropped.item()))
        feat_norm_reduce      = misc.all_reduce_mean(float(feat_norm.item()))

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            step_gs = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("train/loss",           loss_value_reduce,     step_gs)
            log_writer.add_scalar("train/rec_loss_vq",    rec_loss_vq_reduce,    step_gs)
            log_writer.add_scalar("train/info_nce_loss",  info_nce_loss_reduce,  step_gs)
            log_writer.add_scalar("train/masked_dino_ce", masked_dino_ce_reduce, step_gs)
            log_writer.add_scalar("train/class_loss",     class_loss_reduce,     step_gs)
            log_writer.add_scalar("train/acc_cls",        train_acc_cls_reduce,  step_gs)
            log_writer.add_scalar("train/lr",             lr,                    step_gs)
            log_writer.add_scalar("mask/frac_masked",     frac_masked_reduce,    step_gs)
            log_writer.add_scalar("mask/frac_dropped",    frac_dropped_reduce,   step_gs)
            log_writer.add_scalar("stats/feat_norm_mean", feat_norm_reduce,      step_gs)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
