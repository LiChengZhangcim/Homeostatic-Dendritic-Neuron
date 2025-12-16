import torch
from typing import Callable
from spikingjelly.clock_driven import neuron, base
from Surrogate_gradient import GaussianSurrogateFunction
from utils import absId

class LIFVO2(neuron.BaseNode):
    def __init__(self,
                 Rh: float = 100e3, Rs: float = 1.5e3,
                 v_threshold: float = 2., v_reset: float = 0.5,
                 Cmem: float = 200e-9,
                 Vdd: float = 5.,
                 input_scaling: float = 250e-6,
                 dt: float = 1e-3, refractory: int = 5,
                 surrogate_function: Callable = neuron.surrogate.Sigmoid(),
                 detach_reset: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.Rh = Rh
        self.Rs = Rs
        self.Cmem = Cmem
        self.tau = Cmem * (Rh + Rs)
        self.factor = torch.exp(torch.tensor(-dt / self.tau))
        self.refractory = refractory
        self.dt = dt

        self.Vdd = Vdd

        self.input_scaling = input_scaling

        self.v = 0.
        self.register_memory('va', 0.)
        self.register_memory('spike', 0)
        self.register_memory('spike_countdown', None)
        self.register_memory('is_refractory', 0)

    def neuronal_charge(self, x: torch.Tensor):
        x = torch.relu(x)  
        v = self.factor * self.v + (1 - self.factor) * (self.Rh + self.Rs) * (self.input_scaling * x)
        self.v = torch.where(self.is_refractory, self.v, v)

    def neuronal_fire(self):
       
        spike = self.surrogate_function((self.v - self.v_threshold) / self.v_threshold)

        with torch.no_grad():
            if self.spike_countdown is None:
                shape = list(spike.shape)
                shape.append(self.refractory)
                self.spike_countdown = torch.zeros(shape, device=spike.device)

        spike = torch.where(self.is_refractory, torch.zeros_like(spike), spike)
        self.spike = spike
        if spike.dim() == 2:
            self.spike_countdown = torch.cat([self.spike_countdown[:, :, 1:], torch.unsqueeze(spike, dim=-1)], dim=-1)
        elif spike.dim() == 1:
            self.spike_countdown = torch.cat([self.spike_countdown[:, 1:], torch.unsqueeze(spike, dim=-1)], dim=-1)
        return self.spike

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        self.v = (1. - spike_d) * self.v + spike_d * self.v_reset

    def forward(self, x: torch.Tensor):
        if x.dim() == 3 :
            x = x.permute(2, 0, 1) 
        elif x.dim() == 2:
            x = x.permute(1, 0)
        with torch.no_grad():
            if not isinstance(self.spike, torch.Tensor):
                self.spike = torch.zeros(x[0].shape, device=x.device)
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros(x[0].shape, device=x.device)
            if not isinstance(self.is_refractory, torch.Tensor):
                self.is_refractory = torch.zeros(x[0].shape, device=x.device, dtype=torch.bool)
        new_spike = []
        for i in range(x.size(0)):
            self.neuronal_reset(self.spike)
            self.neuronal_charge(x[i])
            new_spike.append(self.neuronal_fire())
            self.is_refractory = torch.gt(torch.amax(self.spike_countdown[..., -self.refractory:], dim=-1), 0)
        if x.dim() == 3 :
            out = torch.stack(new_spike, dim=0).permute(1, 2, 0)
        elif x.dim() == 2:
            out = torch.stack(new_spike, dim=0).permute(1, 0)
        return out

class ALIFVO2(neuron.BaseNode):
    def __init__(self,
                 Rh: float = 100e3, Rs: float = 1.5e3, Ra: float = 100e3,
                 v_threshold: float = 2., v_reset: float = 0.5,
                 Cmem: float = 200e-9, Ca: float = 7e-6,
                 vtn: float = 0.745, vtp: float = 0.973, kappa_n: float = 29e-6, kappa_p: float = 18e-6,
                 wl_ratio_n: float = 6., wl_ratio_p: float = 4.,
                 Vdd: float = 5.,
                 input_scaling: float = 250e-6,
                 dt: float = 1e-3, refractory: int = 5,
                 surrogate_function: Callable = neuron.surrogate.Sigmoid(),
                 detach_reset: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.Rh = Rh
        self.Rs = Rs
        self.Cmem = Cmem
        self.tau = Cmem * (Rh + Rs)
        self.factor = torch.exp(torch.tensor(-dt / self.tau))
        self.refractory = refractory
        self.dt = dt

        self.Ra = Ra
        self.Ca = Ca  
        self.tau_va = Ca * Ra
        self.factor_va = torch.exp(torch.tensor(-dt / self.tau_va))

        self.Vdd = Vdd
        self.vtn = vtn
        self.vtp = vtp
        self.kappa_n = kappa_n
        self.kappa_p = kappa_p
        self.wl_ratio_n = wl_ratio_n
        self.wl_ratio_p = wl_ratio_p

        self.input_scaling = input_scaling

        self.v = 0.
        self.register_memory('va', 0.)
        self.register_memory('spike', 0)
        self.register_memory('spike_countdown', None)
        self.register_memory('is_refractory', 0)

    def neuronal_charge(self, x: torch.Tensor):
        x = torch.relu(x)  
        Il = absId(kappa=self.kappa_n, w_over_l=self.wl_ratio_n, vgs=self.va,
                   vth=self.vtn, vds=self.v)  
        v = self.factor * self.v + (1 - self.factor) * (self.Rh + self.Rs) * (self.input_scaling * x - Il)
        self.v = torch.where(self.is_refractory, self.v, v)

    def neuronal_fire(self):
        
        spike = self.surrogate_function((self.v - self.v_threshold) / self.v_threshold)

        with torch.no_grad():
            if self.spike_countdown is None:
                shape = list(spike.shape)
                shape.append(self.refractory)
                self.spike_countdown = torch.zeros(shape, device=spike.device)

        spike = torch.where(self.is_refractory, torch.zeros_like(spike), spike)
        self.spike = spike
        if spike.dim() == 2:
            self.spike_countdown = torch.cat([self.spike_countdown[:, :, 1:], torch.unsqueeze(spike, dim=-1)], dim=-1)
        elif spike.dim() == 1:
            self.spike_countdown = torch.cat([self.spike_countdown[:, 1:], torch.unsqueeze(spike, dim=-1)], dim=-1)
        return self.spike

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        self.v = (1. - spike_d) * self.v + spike_d * self.v_reset

    def neuronal_adapt(self, spike):
        Ia = absId(kappa=self.kappa_p, w_over_l=self.wl_ratio_p, vgs=(self.Vdd * spike),
                   vth=self.vtp, vds=(self.Vdd - self.va))  
        self.va = self.factor_va * self.va + (1 - self.factor_va) * self.Ra * Ia

    def forward(self, x: torch.Tensor):
        if x.dim() == 3 :
            x = x.permute(2, 0, 1) 
        elif x.dim() == 2:
            x = x.permute(1, 0)
            
        with torch.no_grad():
            if not isinstance(self.spike, torch.Tensor):
                self.spike = torch.zeros(x[0].shape, device=x.device)
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros(x[0].shape, device=x.device)
            if not isinstance(self.va, torch.Tensor):
                self.va = torch.zeros(x[0].shape, device=x.device)
            if not isinstance(self.is_refractory, torch.Tensor):
                self.is_refractory = torch.zeros(x[0].shape, device=x.device, dtype=torch.bool)

        new_spike = []
        for i in range(x.size(0)):
            self.neuronal_reset(self.spike)
            self.neuronal_adapt(self.spike)
            self.neuronal_charge(x[i])
            new_spike.append(self.neuronal_fire())
            self.is_refractory = torch.gt(torch.amax(self.spike_countdown[..., -self.refractory:], dim=-1), 0)
        if x.dim() == 3 :
            out = torch.stack(new_spike, dim=0).permute(1, 2, 0)
        elif x.dim() == 2:
            out = torch.stack(new_spike, dim=0).permute(1, 0)
        return out

class HSLIFVO2(neuron.BaseNode):
    def __init__(self,
                 Rh: float = 100e3, Rs: float = 1.5e3, Ra: float = 100e3,
                 v_threshold: float = 2., v_reset: float = 0.5,
                 Cmem: float = 200e-9, Ca: float = 7e-6,
                 vtn: float = 0.745, vtp: float = 0.973, kappa_n: float = 29e-6, kappa_p: float = 18e-6,
                 wl_ratio_n: float = 6., wl_ratio_p: float = 4.,
                 Vdd: float = 5.,
                 input_scaling: float = 250e-6,
                 dt: float = 1e-3, refractory: int = 5,
                 surrogate_function: Callable = neuron.surrogate.Sigmoid(),
                 detach_reset: bool = False):
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)
        self.Rh = Rh
        self.Rs = Rs
        self.Cmem = Cmem
        self.tau = Cmem * (Rh + Rs)
        self.factor = torch.exp(torch.tensor(-dt / self.tau))
        self.refractory = refractory
        self.dt = dt

        self.Ra = Ra
        self.Ca = Ca  
        self.tau_va = Ca * Ra
        self.factor_va = torch.exp(torch.tensor(-dt / self.tau_va))

        self.Vdd = Vdd
        self.vtn = vtn
        self.vtp = vtp
        self.kappa_n = kappa_n
        self.kappa_p = kappa_p
        self.wl_ratio_n = wl_ratio_n
        self.wl_ratio_p = wl_ratio_p

        self.input_scaling = input_scaling

        self.v = 0.
        self.register_memory('va', 0.)
        self.register_memory('spike', 0)
        self.register_memory('spike_countdown', None)
        self.register_memory('is_refractory', 0)

    def balanced_neuronal_charge(self, x: torch.Tensor):
        x = torch.relu(x)  

        Ia = absId(kappa=self.kappa_p, w_over_l=self.wl_ratio_p, vgs=(self.Vdd * self.spike),
                   vth=self.vtp, vds=(self.Vdd - self.va))  
        self.va = self.factor_va * self.va + (1 - self.factor_va) * self.Ra * Ia

        Il = absId(kappa=self.kappa_n, w_over_l=self.wl_ratio_n, vgs=self.va,
                   vth=self.vtn, vds=self.v) 
        I2 = absId(kappa=self.kappa_n, w_over_l=self.wl_ratio_n, vgs=self.Vdd - self.va,
                   vth=self.vtn, vds=self.Vdd - self.v) 
        v = self.factor * self.v + (1 - self.factor) * (self.Rh + self.Rs) * (self.input_scaling * x - Il + I2)
        self.v = torch.where(self.is_refractory, self.v, v)

    def neuronal_fire(self):
        spike = self.surrogate_function((self.v - self.v_threshold) / self.v_threshold)

        with torch.no_grad():
            if self.spike_countdown is None:
                shape = list(spike.shape)
                shape.append(self.refractory)
                self.spike_countdown = torch.zeros(shape, device=spike.device)

        spike = torch.where(self.is_refractory, torch.zeros_like(spike), spike)
        self.spike = spike
        if spike.dim() == 2:
            self.spike_countdown = torch.cat([self.spike_countdown[:, :, 1:], torch.unsqueeze(spike, dim=-1)], dim=-1)
        elif spike.dim() == 1:
            self.spike_countdown = torch.cat([self.spike_countdown[:, 1:], torch.unsqueeze(spike, dim=-1)], dim=-1)
        self.is_refractory = torch.gt(torch.amax(self.spike_countdown[..., -self.refractory:], dim=-1), 0)
        return self.spike

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        self.v = (1. - spike_d) * self.v + spike_d * self.v_reset


    def forward(self, x: torch.Tensor):
        if x.dim() == 3 :
            x = x.permute(2, 0, 1)
        elif x.dim() == 2:
            x = x.permute(1, 0)
            
        with torch.no_grad():
            if not isinstance(self.spike, torch.Tensor):
                self.spike = torch.zeros(x[0].shape, device=x.device)
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros(x[0].shape, device=x.device)
            if not isinstance(self.va, torch.Tensor):
                self.va = torch.zeros(x[0].shape, device=x.device)
            if not isinstance(self.is_refractory, torch.Tensor):
                self.is_refractory = torch.zeros(x[0].shape, device=x.device, dtype=torch.bool)
        
        new_spike = []
        for i in range(x.size(0)):
            self.neuronal_reset(self.spike)
            self.balanced_neuronal_charge(x[i])
            new_spike.append(self.neuronal_fire())
        if x.dim() == 3 :
            out = torch.stack(new_spike, dim=0).permute(1, 2, 0)
        elif x.dim() == 2:
            out = torch.stack(new_spike, dim=0).permute(1, 0)
        return out
