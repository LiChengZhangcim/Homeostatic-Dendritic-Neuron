import torch
import torch.nn as nn
from Surrogate_gradient import GaussianSurrogateFunction
from data_process_CRWU import load_data
from spikingjelly.clock_driven import neuron, functional
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset 
from Surrogate_gradient import GaussianSurrogateFunction
from neuro import LIFVO2,ALIFVO2,HSLIFVO2
from STFT_conv import ConvSTFT
import os

class SharedWeightsLinear(nn.Module):
    def __init__(self, input_features, output_features):
        super(SharedWeightsLinear, self).__init__()
        self.linear = nn.Linear(input_features, output_features, bias=False)
    def forward(self, x):
        batch_size, features, lens = x.shape
        x_reshaped = x.reshape(-1, features)
        out = self.linear(x_reshaped)
        out = out.reshape(batch_size, -1, lens)
        return out

class Net(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            ConvSTFT(win_len=16, win_inc=16, fft_len=16, win_type='hamming', feature_type='complex', fix=True),
            LIFVO2(surrogate_function = GaussianSurrogateFunction())
        )
        self.fc = nn.Sequential(
            SharedWeightsLinear(18, 32),
            HSLIFVO2(surrogate_function = GaussianSurrogateFunction()))
        self.fc2 = nn.Linear(32*64, out_channel)
        
    def forward(self, x):
        x = self.layer1(x) 
        x = self.fc(x)
        x = x.reshape(x.size(0), -1)
        output = self.fc2(x)
        return output
        

def train_Model(model,train_loader,optimizer,epoch):
    model.train()
    trained_samples = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data) 
        loss_fn = nn.MSELoss(reduce=True, size_average=True)
        loss = loss_fn(output.float(), target.float()) 
        loss.backward(loss.clone().detach())
        optimizer.step()
        trained_samples += len(data)
        print("\rTrain epoch %d: %d/%d, " %
              (epoch, trained_samples, len(train_loader.dataset),), end='')
        pred = output.argmax(dim=1, keepdim=True)
        real = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(real.view_as(pred)).sum().item()
        functional.reset_net(model)
    train_acc = correct / len(train_loader.dataset)
    print("Train acc: " , train_acc)


def test_Model(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)  #logits
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            functional.reset_net(model)
    test_loss /= len(test_loader.dataset)
    print('Test: accuracy: {}/{} ({:.2f}%) \n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)

if __name__ == "__main__":
    epochs = 100
    batch_size = 16
    torch.manual_seed(0)
    save_dir = "CWRU_models_shareweight"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_path = os.path.join(save_dir, "CRWU_snn_shareweight_model_paras.pth")
    
    train_dataset,train_label,val_dataset,val_label,test_dataset,test_label = load_data(num = 100,length = 1024,hp = [0,1,2,3],fault_diameter = [0.007,0.014,0.021],split_rate = [0.7,0.1,0.2])
    
    train_dataset = torch.tensor(train_dataset)
    train_label = torch.tensor(train_label)
    test_dataset = torch.tensor(test_dataset)
    test_label = torch.tensor(test_label)
    val_dataset = torch.tensor(val_dataset)
    val_label = torch.tensor(val_label)
    
    train_dataset = train_dataset.unsqueeze(1)
    test_dataset  = test_dataset.unsqueeze(1)
    val_dataset  = val_dataset.unsqueeze(1)
    train_dataset = train_dataset.to(torch.float32)
    test_dataset  = test_dataset.to(torch.float32)
    val_dataset  = val_dataset.to(torch.float32)
    
    
    train_id = TensorDataset(train_dataset, train_label) 
    test_id  = TensorDataset(test_dataset, test_label)
    val_id  = TensorDataset(val_dataset, val_label)
    train_loader = DataLoader(dataset=train_id, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_id,  batch_size=batch_size, shuffle=False)
    val_loader  = DataLoader(dataset=val_id,  batch_size=batch_size, shuffle=False)
    
    model = Net()
    optimizer = torch.optim.Adadelta(model.parameters())
    model_history = []

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
        test_Model(model, val_loader)

    for epoch in range(1, epochs + 1):
        train_Model(model, train_loader, optimizer, epoch)  
        test_Model(model, test_loader)
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    print("\n=== Final Validation Results ===")
    test_Model(model, val_loader)