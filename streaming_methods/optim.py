import torch



class NormSGD(torch.optim.Optimizer):

    def __init__(self, params, lr=0.3):
        defaults = dict(lr=lr)
        super(NormSGD, self).__init__(params, defaults)

    def step(self):
        grad_norm = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                grad_norm += p.grad.abs().sum().item()

        M = grad_norm
        #EPS = 1e-8
        #M = max(grad_norm, EPS)
        step_size = group["lr"] if M < 1 else group["lr"] / M

        for group in self.param_groups:
            for p in group["params"]:
                p.data.add_(p.grad, alpha=-step_size)


class Norm2SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=0.3):
        defaults = dict(lr=lr)
        super(Norm2SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        grad_norm_sq = 0.0

        # Compute L2 norm of the full gradient
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad_norm_sq += p.grad.pow(2).sum().item()

        grad_norm = grad_norm_sq ** 0.5

        for group in self.param_groups:
            lr = group["lr"]
            step_size = min(lr, lr / (grad_norm + 1e-8))

            for p in group["params"]:
                if p.grad is None:
                    continue
                p.add_(p.grad, alpha=-step_size)




class ObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD, self).__init__(params, defaults)
    def step(self, delta, reset=False):
        z_sum = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                e = state["eligibility_trace"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                e = state["eligibility_trace"]
                p.data.add_(delta * e, alpha=-step_size)
                if reset:
                    e.zero_()

    def _reset(self):

        # reset traces once done=True
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["eligibility_trace"] = torch.zeros_like(p.data)


class AdaptiveObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, beta2=beta2, eps=eps)
        self.counter = 0
        super(AdaptiveObGD, self).__init__(params, defaults)
    def step(self, delta, reset=False):
        z_sum = 0.0
        self.counter += 1
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                e, v = state["eligibility_trace"], state["v"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)

                v.mul_(group["beta2"]).addcmul_(delta*e, delta*e, value=1.0 - group["beta2"])
                v_hat = v / (1.0 - group["beta2"] ** self.counter)
                z_sum += (e / (v_hat + group["eps"]).sqrt()).abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                v, e = state["v"], state["eligibility_trace"]
                v_hat = v / (1.0 - group["beta2"] ** self.counter)
                p.data.addcdiv_(delta * e, (v_hat + group["eps"]).sqrt(), value=-step_size)
                if reset:
                    e.zero_()
