import os
import glob
import random
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader

class SpeechCommandsDataset(Dataset):
    TARGET_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    NUM_CLASSES = 12
    SAMPLE_RATE = 16000

    def __init__(self, data_dir, split='train', transform=None, subsample_unknown=0.1):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load validation and testing lists
        val_list_path = os.path.join(data_dir, 'validation_list.txt')
        test_list_path = os.path.join(data_dir, 'testing_list.txt')
        
        def load_list(path):
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return set(line.strip() for line in f)
            return set()
            
        val_list = load_list(val_list_path)
        test_list = load_list(test_list_path)
        
        self.samples = []
        self.background_noise = []
        
        word_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        unknown_samples = []
        
        for word in word_folders:
            if word == '_background_noise_':
                noise_files = glob.glob(os.path.join(data_dir, word, '*.wav'))
                for nf in noise_files:
                    waveform, sr = torchaudio.load(nf)
                    if sr != self.SAMPLE_RATE:
                        waveform = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(waveform)
                    self.background_noise.append(waveform[0])
                continue
                
            is_target = word in self.TARGET_WORDS
            label = self.TARGET_WORDS.index(word) if is_target else 10
            
            word_files = glob.glob(os.path.join(data_dir, word, '*.wav'))
            for wf in word_files:
                rel_path = f"{word}/{os.path.basename(wf)}"
                if rel_path in test_list:
                    item_split = 'test'
                elif rel_path in val_list:
                    item_split = 'val'
                else:
                    item_split = 'train'
                    
                if item_split == split:
                    if label == 10:
                        unknown_samples.append((wf, label))
                    else:
                        self.samples.append((wf, label))
                        
        if split == 'train' and subsample_unknown < 1.0:
            random.shuffle(unknown_samples)
            num_unknown = int(len(self.samples) * subsample_unknown)
            unknown_samples = unknown_samples[:num_unknown]
            
        self.samples.extend(unknown_samples)
        
        # Generate silence clips (class 11)
        if split == 'train':
            num_silence = int(len(self.samples) * 0.1)
        elif split == 'val':
            num_silence = int(len(self.samples) * 0.1)
        else:
            num_silence = int(len(self.samples) * 0.1)
            
        for _ in range(num_silence):
            self.samples.append(('silence', 11))
            
    def _get_silence(self):
        if not self.background_noise:
            return torch.zeros(1, self.SAMPLE_RATE)
        noise = random.choice(self.background_noise)
        if len(noise) > self.SAMPLE_RATE:
            start = random.randint(0, len(noise) - self.SAMPLE_RATE)
            return noise[start:start+self.SAMPLE_RATE].unsqueeze(0)
        else:
            pad = self.SAMPLE_RATE - len(noise)
            return torch.nn.functional.pad(noise, (0, pad)).unsqueeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        
        if path == 'silence':
            waveform = self._get_silence()
        else:
            waveform, sr = torchaudio.load(path)
            if sr != self.SAMPLE_RATE:
                waveform = torchaudio.transforms.Resample(sr, self.SAMPLE_RATE)(waveform)
            if waveform.shape[1] < self.SAMPLE_RATE:
                pad = self.SAMPLE_RATE - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            elif waveform.shape[1] > self.SAMPLE_RATE:
                waveform = waveform[:, :self.SAMPLE_RATE]
            
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, label
