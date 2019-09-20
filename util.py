import torch.optim.lr_scheduler as lrs


def make_scheduler(my_optimizer):
    # learning rate decay per N epochs
    lr_decay = 200
    # learning rate decay factor for step decay
    gamma = 0.5
    scheduler = lrs.StepLR(my_optimizer, step_size=lr_decay, gamma=gamma)
    return scheduler


class LrMultiStep(object):
    def __init__(self, optimizer, milestones, lr_mults, last_iter=-1):
        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestones, lr_mults)
        self.milestones = milestones
        self.lr_mults = lr_mults
        self.optimizer = optimizer
        self.last_iter = last_iter
        for i, group in enumerate(optimizer.param_groups):
            if 'lr' not in group:
                raise KeyError("param 'lr' is not specified"
                               " in param_groups[{}] when resuming an optimizer".format(i))

    def _get_lr(self):
        try:
            pos = self.milestones.index(self.last_iter)
        except ValueError:
            return list(map(lambda group: group['lr'], self.optimizer.param_groups))
        except:
            raise Exception("don't know what error! wtf")
        return list(map(lambda group: group['lr'] * self.lr_mults[pos], self.optimizer.param_groups))

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = this_iter + 1
        self.last_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr
