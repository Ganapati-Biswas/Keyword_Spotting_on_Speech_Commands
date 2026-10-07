import torch
import torch.nn as nn
import torchaudio.transforms as T

class LogMelFeatureExtractor(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=480, hop_length=160, n_mels=40, eps=1e-6, per_utt_norm=True):
        super().__init__()
        self.eps = eps
        self.per_utt_norm = per_utt_norm
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=False
        )

    def forward(self, x):
        # x: [B, T] or [B, 1, T]
        if x.dim() == 3:
            x = x.squeeze(1)
        mel = self.mel_spectrogram(x) # [B, n_mels, time]
        log_mel = torch.log(mel + self.eps)
        
        if self.per_utt_norm:
            mean = log_mel.mean(dim=(1, 2), keepdim=True)
            std = log_mel.std(dim=(1, 2), keepdim=True)
            log_mel = (log_mel - mean) / (std + self.eps)
            
        return log_mel.unsqueeze(1) # [B, 1, n_mels, time]

class SpecAugment(nn.Module):
    def __init__(self, time_mask_param=10, freq_mask_param=10):
        super().__init__()
        self.time_masking = T.TimeMasking(time_mask_param=time_mask_param)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=freq_mask_param)

    def forward(self, x):
        # x: [B, C, F, T]
        x = self.time_masking(x)
        x = self.freq_masking(x)
        return x

def extract_log_mel(waveform, **kwargs):
    extractor = LogMelFeatureExtractor(**kwargs)
    return extractor(waveform)
