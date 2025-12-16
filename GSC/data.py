import torch
import torchaudio
import matplotlib.pyplot as plt
import IPython.display as ipd
from torchaudio.datasets import SPEECHCOMMANDS
import os
import json
import numpy as np
import torch
import librosa
from sklearn.preprocessing import normalize
import numpy as np
import librosa


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__(r"/root/", download=False)
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

class SpeechCommandsDataset:
    def __init__(self, batch_size=256, device="cuda"):
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.train_set = SubsetSC("training")
        self.test_set = SubsetSC("testing")
        self.val_set = SubsetSC("validation")

        self.labels_file = "speechcommand_labels.json"
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                self.labels = json.load(f)
        else:
            self.labels = sorted(list(set(datapoint[2] for datapoint in self.train_set)))

            with open(self.labels_file, 'w') as f:
                json.dump(self.labels, f)

        self.train_loader = self._create_data_loader(self.train_set, shuffle=True)
        self.test_loader = self._create_data_loader(self.test_set, shuffle=False)
        self.val_loader = self._create_data_loader(self.val_set, shuffle=False)
    
    def _create_data_loader(self, dataset, shuffle):
        if self.device == "cuda":
            num_workers = 8
            pin_memory = True
        else:
            num_workers = 0
            pin_memory = False
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def label_to_index(self, word):
        return torch.tensor(self.labels.index(word))

    def index_to_label(self, index):
        return self.labels[index]

    def pad_sequence(self, batch):
        batch = [item.t() for item in batch]
        batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
        return batch.permute(0, 2, 1)
    
        
    def mel_spectrogram(self, wav):

        sr =16000
        delta_order=2
        stack=True
        S = librosa.feature.melspectrogram(y=wav,
                                        sr=sr,
                                        n_fft=int(30e-3*sr),
                                        hop_length=int(10e-3*sr),
                                        n_mels=40, 
                                        fmax=4000,
                                        fmin=20
                                        )

        M = np.max(np.abs(S))
        if M > 0:
            feat = np.log1p(S / M)
        else:
            feat = S

        if delta_order is not None:
            feat_list = [feat.T]
            for k in range(1, delta_order + 1):
                feat_list.append(librosa.feature.delta(feat, order=k).T)
            if stack:
                return np.stack(feat_list)
            else:
                return np.expand_dims(feat.T, 0)
        else:
            return np.expand_dims(feat.T, 0)


    def rescale(self, input):
        std = np.std(input, axis=1, keepdims=True)
        std[std == 0] = 1 
        return input / std

    def collate_fn(self, batch):
        tensors, targets = [], []
        for waveform, _, label, *_ in batch:
            tensors += [waveform]
            targets += [self.label_to_index(label)]
        tensors = self.pad_sequence(tensors)
        tensors = self.mel_spectrogram(tensors.numpy())

        tensors = self.rescale(tensors)
        tensors = torch.tensor(tensors, dtype=torch.float64).squeeze(3).permute(3, 0, 1, 2)
        targets = torch.stack(targets)
        return tensors, targets

    def play_audio(self, index=0):
        waveform, sample_rate = self.train_set[index][0], self.train_set[index][1]
        return ipd.Audio(waveform.numpy(), rate=sample_rate)

    def play_resampled_audio(self, index=0, new_sample_rate=8000):
        waveform, sample_rate = self.train_set[index][0], self.train_set[index][1]
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        transformed = transform(waveform)
        return ipd.Audio(transformed.numpy(), rate=new_sample_rate)
    
    def plot_audio(self,index=0):

        waveform, sample_rate, *_ = self.train_set[index]
        print("Shape of waveform: {}".format(waveform.size()))
        print("Sample rate of waveform: {}".format(sample_rate))
        plt.plot(waveform.t().numpy())


if __name__ == "__main__":
    dataset = SpeechCommandsDataset()
    dataset.test_loader
    dataset.train_loader
    dataset.play_audio()
    dataset.play_resampled_audio()
    dataset.plot_audio()

    a = dataset.test_loader
    for batch_idx, (inputs, targets) in enumerate(a):
        print(f"Batch {batch_idx + 1}:")
        print(f"Inputs: {inputs.shape}")  
        print(f"Targets: {targets.shape}")  

        if batch_idx == 1:  
            break
