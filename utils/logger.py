from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import sys, time

class AbstractMeter(object):
    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def update(self, value: float, n: int = 1):
        pass
    
    def __str__(self):
        fmtstr = "{name} {value" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    

class AverageMeter(AbstractMeter):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        super().__init__(name, fmt)
        self.reset()

    def reset(self) -> None:
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

class MaxMeter(AbstractMeter):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        super().__init__(name, fmt)
        self.reset()

    def reset(self) -> None:
        self.value = 0
        self.max_val = -987654321

    def update(self, value: float) -> None:
        self.value = value
        self.max_val = max(self.max_val, self.value)


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                  stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                  'ipykernel' in sys.modules or
                                  'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None, silent=False):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                        current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if not silent:
                if self._dynamic_display:
                    sys.stdout.write('\b' * prev_total_width)
                    sys.stdout.write('\r')
                else:
                    sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            if not silent:
                sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                    (eta % 3600) // 60,
                                                    eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            if not silent:
                sys.stdout.write(info)
                sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'
                if not silent:
                    sys.stdout.write(info)
                    sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None,silent=False):
        self.update(self._seen_so_far + n, values,silent=silent)
        
def build_meters(type: str) -> Dict[str, AbstractMeter]:
    meters = {}
    if type == "average":
        # Train Loss metrics
        meters["Train/Total_Loss"] = AverageMeter(name="Train/Total_Loss")
        meters["Train/Obj_Cls_Loss"] = AverageMeter(name="Train/Obj_Cls_Loss")
        meters["Train/Rel_Cls_Loss"] = AverageMeter(name="Train/Rel_Cls_Loss")
        meters["Train/Contrastive_Loss"] = AverageMeter(name="Train/Contrastive_Loss")
        # Train evaluation metrics
        meters["Train/Obj_R1"] = AverageMeter(name="Train/Obj_R1")
        meters["Train/Obj_R5"] = AverageMeter(name="Train/Obj_R3")
        meters["Train/Obj_R10"] = AverageMeter(name="Train/Obj_R5")
        meters["Train/Pred_R1"] = AverageMeter(name="Train/Pred_R1")
        meters["Train/Pred_R3"] = AverageMeter(name="Train/Pred_R3")
        meters["Train/Pred_R5"] = AverageMeter(name="Train/Pred_R5")
        meters['Validation/Total_Loss'] = AverageMeter(name='Validation/Total_Loss')
        # Validation evaluation metrics
        # meters["Validation/Acc@1/obj_cls"] = AverageMeter(name="Validation/Acc@1/obj_cls")
        # meters["Validation/Acc@5/obj_cls"] = AverageMeter(name="Validation/Acc@5/obj_cls")
        # meters["Validation/Acc@10/obj_cls"] = AverageMeter(name="Validation/Acc@10/obj_cls")
        # meters["Validation/Acc@1/rel_cls_acc"] = AverageMeter(name="Validation/Acc@1/rel_cls_acc")
        # meters["Validation/Acc@5/rel_cls_acc"] = AverageMeter(name="Validation/Acc@5/rel_cls_acc")
        # meters["Validation/Acc@10/rel_cls_acc"] = AverageMeter(name="Validation/Acc@10/rel_cls_acc")
        # meters["Validation/Acc@1/rel_cls_acc_mean"] = AverageMeter(name="Validation/Acc@1/rel_cls_acc_mean")
        # meters["Validation/Acc@5/rel_cls_acc_mean"] = AverageMeter(name="Validation/Acc@5/rel_cls_acc_mean")
        # meters["Validation/Acc@10/rel_cls_acc_mean"] = AverageMeter(name="Validation/Acc@10/rel_cls_acc_mean")
        # meters["Validation/Acc@50/triplet_acc"] = AverageMeter(name="Validation/Acc@50/triplet_acc")
        # meters["Validation/Acc@100/triplet_acc"] = AverageMeter(name="Validation/Acc@100/triplet_acc")
        # meters["Validation/mRecall@50"] = AverageMeter(name="Validation/mRecall@50")
        # meters["Validation/mRecall@100"] = AverageMeter(name="Validation/mRecall@100")
        
    elif type == "max":
        # Train Loss metrics
        meters["Train/Total_Loss"] = MaxMeter(name="Train/Total_Loss")
        meters["Train/Obj_Cls_Loss"] = MaxMeter(name="Train/Obj_Cls_Loss")
        meters["Train/Rel_Cls_Loss"] = MaxMeter(name="Train/Rel_Cls_Loss")
        meters["Train/Contrastive_Loss"] = MaxMeter(name="Train/Contrastive_Loss")
        # Train evaluation metrics
        meters["Train/Obj_R1"] = MaxMeter(name="Train/Obj_R1")
        meters["Train/Obj_R5"] = MaxMeter(name="Train/Obj_R3")
        meters["Train/Obj_R10"] = MaxMeter(name="Train/Obj_R5")
        meters["Train/Pred_R1"] = MaxMeter(name="Train/Pred_R1")
        meters["Train/Pred_R3"] = MaxMeter(name="Train/Pred_R3")
        meters["Train/Pred_R5"] = MaxMeter(name="Train/Pred_R5")
        # Validation evaluation metrics
        # meters["Validation/Acc@1/obj_cls"] = MaxMeter(name="Validation/Acc@1/obj_cls")
        # meters["Validation/Acc@5/obj_cls"] = MaxMeter(name="Validation/Acc@5/obj_cls")
        # meters["Validation/Acc@10/obj_cls"] = MaxMeter(name="Validation/Acc@10/obj_cls")
        # meters["Validation/Acc@1/rel_cls_acc"] = MaxMeter(name="Validation/Acc@1/rel_cls_acc")
        # meters["Validation/Acc@5/rel_cls_acc"] = MaxMeter(name="Validation/Acc@5/rel_cls_acc")
        # meters["Validation/Acc@10/rel_cls_acc"] = MaxMeter(name="Validation/Acc@10/rel_cls_acc")
        # meters["Validation/Acc@1/rel_cls_acc_mean"] = MaxMeter(name="Validation/Acc@1/rel_cls_acc_mean")
        # meters["Validation/Acc@5/rel_cls_acc_mean"] = MaxMeter(name="Validation/Acc@5/rel_cls_acc_mean")
        # meters["Validation/Acc@10/rel_cls_acc_mean"] = MaxMeter(name="Validation/Acc@10/rel_cls_acc_mean")
        # meters["Validation/Acc@50/triplet_acc"] = MaxMeter(name="Validation/Acc@50/triplet_acc")
        # meters["Validation/Acc@100/triplet_acc"] = MaxMeter(name="Validation/Acc@100/triplet_acc")
        # meters["Validation/mRecall@50"] = MaxMeter(name="Validation/mRecall@50")
        # meters["Validation/mRecall@100"] = MaxMeter(name="Validation/mRecall@100")
        
    else:
        raise NotImplementedError
    return meters