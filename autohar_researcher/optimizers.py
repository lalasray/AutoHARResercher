"""Reference RMSProp rule used by Lasagne: epsilon inside the square root."""
import torch


class LasagneRMSprop(torch.optim.Optimizer):
    def __init__(self, params, lr=.01, alpha=.9, eps=1e-6, weight_decay=0.):
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if group['weight_decay']:
                    gradient = gradient.add(parameter, alpha=group['weight_decay'])
                state = self.state[parameter]
                if not state:
                    state['square_avg'] = torch.zeros_like(parameter)
                accumulator = state['square_avg']
                accumulator.mul_(group['alpha']).addcmul_(gradient, gradient, value=1 - group['alpha'])
                parameter.addcdiv_(gradient, (accumulator + group['eps']).sqrt(), value=-group['lr'])
        return loss
