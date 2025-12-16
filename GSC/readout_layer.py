import torch
import torch.nn as nn
class readout_integrator(nn.Module):
    def __init__(self,input_dim,output_dim,
                 Rh=99e3, Rs=1e3, dt=1e-3,input_scaling=10e-6,
                 Cmem_initializer='uniform', low_Cmem=0, high_Cmem=4,
                 Cmem_10_picofarads=10e-9,
                 device='cuda',bias=True):
        super(readout_integrator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.dense = nn.Linear(input_dim,output_dim,bias=bias)
        self.Rh = Rh
        self.Rs = Rs
        self.dt = dt
        self.input_scaling = input_scaling
        self.C_mem = nn.Parameter(torch.Tensor(self.output_dim))
        if Cmem_initializer == 'uniform':
            nn.init.uniform_(self.C_mem,low_Cmem,high_Cmem)
        elif Cmem_initializer == 'constant':
            nn.init.constant_(self.C_mem,low_Cmem)
        self.Cmem_10_picofarads = Cmem_10_picofarads

    def init_neuron_state(self,batch_size):
        self.mem = (torch.rand(batch_size,self.output_dim)).to(self.device)

    def integrator(self, inputs, mem, tau_m):
        alpha = torch.sigmoid(tau_m)
        mem = mem *alpha +  (1-alpha) * (inputs*self.input_scaling) * (self.Rh + self.Rs)
        return mem
    
    def forward(self,input_spike):
        #synaptic inputs
        tau_m = self.C_mem * self.Cmem_10_picofarads * (self.Rh+self.Rs) / self.dt
        d_input = self.dense(input_spike.float())
        # neuron model without spiking
        self.mem = self.integrator(d_input,self.mem,tau_m)
        return self.mem
