import os
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from models import *
from data import *
class SpeechCommandModelTrainer:
    def __init__(self, model, train_loader, test_loader, val_loader, device, lr=0.01, weight_decay=0.0001, step_size=40, gamma=0.1):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device
        self.model = model.to(self.device)
        if hasattr(self.model, 'dense_3'):
            self.base_params = [
                            self.model.dense_1.dense.weight,
                            self.model.dense_1.dense.bias,
                            self.model.dense_2.dense.weight,
                            self.model.dense_2.dense.bias,
                            self.model.dense_3.dense.weight,
                            self.model.dense_3.dense.bias,
                            self.model.dense_4.dense.weight,
                            self.model.dense_4.dense.bias,
                            ]
            params = [{'params': self.base_params, 'lr': lr},  
                        {'params': self.model.dense_1.C_mem, 'lr': lr * 2},  
                        {'params': self.model.dense_2.C_mem, 'lr': lr * 2},   
                        {'params': self.model.dense_3.C_mem, 'lr': lr * 2},  
                        {'params': self.model.dense_4.C_mem, 'lr': lr * 2},]
            if hasattr(self.model.dense_1, 'tau_n'):
                params.append({'params': self.model.dense_1.tau_n, 'lr': lr * 2})
                params.append({'params': self.model.dense_2.tau_n, 'lr': lr * 2})
                params.append({'params': self.model.dense_3.tau_n, 'lr': lr * 2})
        else :
            self.base_params = [
                            self.model.dense_1.dense.weight,
                            self.model.dense_1.dense.bias,
                            self.model.dense_2.dense.weight,
                            self.model.dense_2.dense.bias,
                            ]
            params = [{'params': self.base_params, 'lr': lr},  
                        {'params': self.model.dense_1.C_mem, 'lr': lr * 2},  
                        {'params': self.model.dense_2.C_mem, 'lr': lr * 2},]
            if hasattr(self.model.dense_1, 'tau_n'):
                params.append({'params': self.model.dense_1.tau_n, 'lr': lr * 2})
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        self.train_loss_history = []
        self.test_loss_history = []
        self.train_accuracy_history = []
        self.test_accuracy_history = []

    def train(self, epoch, log_interval):
        self.model.train()  
        train_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data.view(-1,3,101, 40).to(self.device)
            target = target.view((-1)).long().to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss_fn = nn.CrossEntropyLoss()
            _, pred = torch.max(output.data, 1)
            loss = loss_fn(output, target)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            correct += pred.eq(target.view_as(pred)).sum().item()
            if batch_idx % log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} "
                      f"({100 * batch_idx / len(self.train_loader):.0f}%)]\tLoss: {loss.item():.6f}")
        avg_loss = train_loss / len(self.train_loader)
        accuracy = 100. * correct / len(self.train_loader.dataset)
        self.train_loss_history.append(avg_loss)
        self.train_accuracy_history.append(accuracy)
        print(f"Train Epoch: {epoch}\tAverage Loss: {avg_loss:.6f}\tAccuracy: {correct}/{len(self.train_loader.dataset)} "
              f"({accuracy:.2f}%)")

    def test(self, epoch, test=True):
        self.model.eval()
        test_loss = 0
        correct = 0
        loader = self.test_loader if test else self.val_loader
        with torch.no_grad():
            for data, target in loader:
                data = data.view(-1,3,101, 40).to(self.device)
                target = target.view((-1)).long().to(self.device)
                output = self.model(data)
                loss_fn = nn.CrossEntropyLoss()
                _, pred = torch.max(output.data, 1)
                test_loss += loss_fn(output, target)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(loader.dataset)
        accuracy = 100. * correct / len(loader.dataset)
        self.test_loss_history.append(test_loss)
        self.test_accuracy_history.append(accuracy)
        print(f"\nTest Epoch: {epoch}\tAverage Loss: {test_loss:.6f}\tAccuracy: {correct}/{len(loader.dataset)} "
              f"({accuracy:.2f}%)\n")

    def save_metrics(self):
        with open(os.path.join(self.output_dir, "metrics.txt"), "w") as f:
            f.write("Epoch\tTrain Loss\tTrain Accuracy\tTest Loss\tTest Accuracy\n")
            for i in range(len(self.train_loss_history)):
                f.write(f"{i+1}\t{self.train_loss_history[i]:.6f}\t{self.train_accuracy_history[i]:.2f}\t"
                        f"{self.test_loss_history[i]:.6f}\t{self.test_accuracy_history[i]:.2f}\n")
    
    def plot_metrics(self):
        epochs = range(1, len(self.train_loss_history) + 1)
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss_history, label="Train Loss")
        plt.plot(epochs, self.test_loss_history, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over Epochs")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracy_history, label="Train Accuracy")
        plt.plot(epochs, self.test_accuracy_history, label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy over Epochs")
        plt.legend()
        plt.savefig(f"{self.output_dir}\\train_test_metrics.png")
        plt.show()

    def count_parameters(self): 
        n = sum(p.numel() for p in self.model.parameters())
        print("Number of parameters: %s" % n)

    def run(self, num_epochs, log_interval, output_dir="training_logs"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        for epoch in range(1, num_epochs + 1):
            self.train(epoch, log_interval)
            self.scheduler.step()
            self.test(epoch)
            torch.save(self.model, f"{self.output_dir}/epoch_{epoch}.pth")
        self.save_metrics()
        self.plot_metrics()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size=200
    dataset = SpeechCommandsDataset(batch_size=batch_size, device=device)
    train_loader = dataset.train_loader
    test_loader = dataset.test_loader
    val_loader = dataset.val_loader 
    labels = dataset.labels
    is_bias = True
    neuron = LIF_VO2_dendr_snn
    model = SNN(is_bias=is_bias,device=device,neuron=neuron) 
    model_trainer = SpeechCommandModelTrainer(model, train_loader, test_loader, val_loader, device, lr=0.01, weight_decay=0.0001, step_size=15, gamma=0.1)
    model_trainer.count_parameters()
    output_dir = (neuron(1,1).__class__.__name__) + '_model_params'
    model_trainer.run(num_epochs=80, log_interval=50, output_dir=output_dir)
    model_trainer.count_parameters()