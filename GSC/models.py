import torch.nn as nn
import torch.nn.functional as F
from rsnn_neuron import *
from snn_neuron import *
from readout_layer  import *

class SNN(nn.Module):
    def __init__(self, is_bias=True, device='cuda', neuron=LIF_VO2_dendr_snn):
        super(SNN, self).__init__()
        n = 200
        device = device
        self.dense_1 = neuron(40*3,n,branch = 8,bias=is_bias, device=device)
        self.dense_2 = readout_integrator(n,35,bias=is_bias,device=device)

    def forward(self,input):
        b,channel,seq_length,input_dim = input.shape
        self.dense_2.init_neuron_state(b)
        self.dense_1.init_neuron_state(b)
        output = 0
        input_s = input
        for i in range(seq_length):
            input_x = input_s[:,:,i,:].reshape(b,channel*input_dim)
            mem_layer1,spike_layer1 = self.dense_1.forward(input_x)
            mem_layer2= self.dense_2.forward(spike_layer1)
            output += mem_layer2
        output = F.log_softmax(output/seq_length,dim=1)
        return output
