import torch
import torch.nn.functional as F
import torch
from spikingjelly.clock_driven import neuron

class GaussianSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x > 0).float()  

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            x, = ctx.saved_tensors
            alpha = ctx.alpha
            grad_x = grad_output * alpha * torch.exp(-alpha * (x ** 2))
        return grad_x, None


class GaussianSurrogateFunction(neuron.surrogate.SurrogateFunctionBase):
    def __init__(self, alpha=1.0, spiking=True):
        super().__init__(alpha, spiking)
        assert alpha > 0, 'alpha must be greater than 0'
        self.alpha = alpha
        self.spiking = spiking
        if spiking:
            self.f = self.spiking_function
        else:
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.alpha)

    @staticmethod
    def spiking_function(x, alpha):
        return GaussianSurrogate.apply(x, alpha)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (1 / (1 + torch.exp(-alpha * x)))
