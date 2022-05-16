import sys
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

import librosa
import torchaudio
import torchaudio.transforms as T

class SliderDataset(Dataset):
    """PyTorch dataset class for task2. Caching to a file supported.

    Args:
        n_mels, frames, n_fft, hop_length, power, transform: Audio conversion settings.
        normalize: Normalize data value range from [-90, 24] to [0, 1] for VAE, False by default.
        cache_to: Cache filename or None by default, use this for your iterative development.
    """
    def __init__(self, dirs, n_mels, frames, n_fft, hop_length, power,
                transform=None,
                normalize=False,
                cache_to=None):
        if type(dirs) != list:
            dirs = [dirs]
        self.dirs = dirs
        self.transform = transform
        self.n_mels, self.frames, self.n_fft = n_mels, frames, n_fft
        self.hop_length, self.power = hop_length, power

        self.audio_paths = [audio_path for dir in self.dirs for audio_path in list(Path(dir).iterdir())]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        audio_path = self.audio_paths[index]
        x = self.get_melspectrogram(audio_path)
        if self.transform is not None:
            x = self.transform(x)
        return x, x

    def get_melspectrogram(self, audio_path):
        # 01 calculate the number of dimensions
        dims = self.n_mels * self.frames

        # 02 generate melspectrogram using librosa
        wav, sr = torchaudio.load(audio_path)
        wav = T.AmplitudeToDB()(wav)
        print(f"{sr=}")
        print(f"{wav.shape=}")
        melspec_transformer = T.MelSpectrogram(sample_rate=sr,
                                         n_fft=self.n_fft,
                                         hop_length=self.hop_length,
                                         n_mels=self.n_mels,
                                         power=self.power)
        res = melspec_transformer(wav).squeeze()

        # 03 convert melspectrogram to log mel energy
        # log_mel_spectrogram = 20.0 / self.power * np.log10(mel_spectrogram + sys.float_info.epsilon)
        # print(f"{log_mel_spectrogram.shape=}")

        # # 04 calculate total vector size
        # vector_array_size = len(log_mel_spectrogram[0, :]) - self.frames + 1
        # print(f"{vector_array_size=}")

        # # 05 skip too short clips
        # if vector_array_size < 1:
        #     return np.empty((0, dims))
        
        # TODO: why do this???
        # # 06 generate feature vectors by concatenating multiframes
        # vector_array = np.zeros((vector_array_size, dims))
        # for t in range(self.frames):
        #     vector_array[:, self.n_mels * t: self.n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

        return res