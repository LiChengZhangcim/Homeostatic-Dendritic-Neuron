import torch
import torch.nn as nn
from gradient import ActFun_adp

class LIF_VO2_snn(nn.Module):
    def __init__(self,input_dim,output_dim,
                 Rh=99e3, Rs=1e3, dt=1e-3,
                 v_threshold=2, v_reset=1,
                 input_scaling=10e-6,
                 Cmem_initializer = 'uniform',low_Cmem = 0,high_Cmem = 4,
                 Cmem_10_picofarads=10e-9,
                 branch = 1,
                 device='cuda',bias=True):
        super(LIF_VO2_snn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Rh = Rh
        self.Rs = Rs
        self.dt = dt
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.input_scaling = input_scaling
        self.C_mem = nn.Parameter(torch.Tensor(self.output_dim))
        if Cmem_initializer == 'uniform':
            nn.init.uniform_(self.C_mem,low_Cmem,high_Cmem)
        elif Cmem_initializer == 'constant':
            nn.init.constant_(self.C_mem,low_Cmem)
        self.Cmem_10_picofarads = Cmem_10_picofarads
        self.device = device
        self.dense = nn.Linear(input_dim,output_dim, bias=bias)
        self.ActFunAdp = ActFun_adp.apply

    #init
    def init_neuron_state(self,batch_size):
        #mambrane potential
        self.mem = torch.zeros(batch_size,self.output_dim).to(self.device)
        self.spike = torch.zeros(batch_size,self.output_dim).to(self.device)
        #threshold reset voltage
        self.v_thred = (torch.ones(batch_size,self.output_dim)*self.v_threshold).to(self.device)
        self.v_res = (torch.ones(batch_size,self.output_dim)*self.v_reset).to(self.device)

    def LIF_VO2(self, inputs, mem, spike, tau_m): 
        alpha = torch.sigmoid(tau_m)
        mem = mem*alpha*(1-spike) + (1-alpha)*(inputs*self.input_scaling)*(self.Rh + self.Rs) + self.v_reset*alpha*spike
        inputs_ = mem - self.v_thred
        spike = self.ActFunAdp(inputs_)  
        return mem, spike
    
    def forward(self,input_spike):
        # timing factor
        tau_m = self.C_mem * self.Cmem_10_picofarads * (self.Rh+self.Rs) / self.dt
        k_input = input_spike.float()
        self.d_input = self.dense(k_input)
        #update membrane potential and generate spikes
        self.mem,self.spike = self.LIF_VO2(self.d_input,self.mem,self.spike,tau_m)
        return self.mem,self.spike
  

class HSLIF_VO2_snn(nn.Module):
    def __init__(self,input_dim,output_dim,
                 Rh=99e3, Rs=1e3, dt=1e-3,
                 v_threshold=2, v_reset=1,
                 input_scaling=10e-6,
                 Cmem_initializer = 'uniform',low_Cmem = 0,high_Cmem = 4,
                 Cmem_10_picofarads=10e-9,
                 tau_ninitializer = 'uniform',low_n = 0,high_n = 4,
                 branch = 4,
                 device='cuda',bias=True):
        super(HSLIF_VO2_snn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Rh = Rh
        self.Rs = Rs
        self.dt = dt
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.input_scaling = input_scaling
        self.C_mem = nn.Parameter(torch.Tensor(self.output_dim))
        if Cmem_initializer == 'uniform':
            nn.init.uniform_(self.C_mem,low_Cmem,high_Cmem)
        elif Cmem_initializer == 'constant':
            nn.init.constant_(self.C_mem,low_Cmem)
        self.Cmem_10_picofarads = Cmem_10_picofarads
        self.tau_n = nn.Parameter(torch.Tensor(self.output_dim,branch))
        if tau_ninitializer == 'uniform':
            nn.init.uniform_(self.tau_n,low_n,high_n)
        elif tau_ninitializer == 'constant':
            nn.init.constant_(self.tau_n,low_n)
        self.branch = branch
        self.device = device
        self.pad = ((input_dim)//branch*branch+branch-(input_dim)) % branch
        self.dense = nn.Linear(input_dim+self.pad,output_dim*branch, bias=bias)
        self.ActFunAdp = ActFun_adp.apply

    #init
    def init_neuron_state(self,batch_size):
        #mambrane potential
        self.mem = torch.zeros(batch_size,self.output_dim).to(self.device)
        self.spike = torch.zeros(batch_size,self.output_dim).to(self.device)
        if self.branch == 1:
            self.d_input = torch.zeros(batch_size,self.output_dim,self.branch).to(self.device)
        else:
            self.d_input = torch.zeros(batch_size,self.output_dim,self.branch).to(self.device)
        #threshold reset voltage
        self.v_thred = (torch.ones(batch_size,self.output_dim)*self.v_threshold).to(self.device)
        self.v_res = (torch.ones(batch_size,self.output_dim)*self.v_reset).to(self.device)

    def HSLIF_VO2(self, inputs, mem, spike, tau_m): 
        alpha = torch.sigmoid(tau_m)
        mem = mem*alpha*(1-spike) + (1-alpha)*(inputs*self.input_scaling)*(self.Rh + self.Rs) + self.v_reset*alpha*spike
        inputs_ = mem - self.v_thred
        spike = self.ActFunAdp(inputs_)  
        return mem, spike
    
    def forward(self,input_spike):
        # timing factor
        tau_m = self.C_mem * self.Cmem_10_picofarads * (self.Rh+self.Rs) / self.dt
        beta = torch.sigmoid(self.tau_n).to(self.device)
        padding = torch.zeros(input_spike.size(0),self.pad).to(self.device)
        k_input = torch.cat((input_spike.float(),padding),1)
        self.d_input = beta*self.d_input+(1-beta)*self.dense(k_input).reshape(-1,self.output_dim,self.branch)
        l_input = (self.d_input).sum(dim=2,keepdim=False)
        #update membrane potential and generate spikes
        self.mem,self.spike = self.HSLIF_VO2(l_input,self.mem,self.spike,tau_m)
        return self.mem,self.spike

