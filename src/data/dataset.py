from pathlib import Path
import re

import torch
from torch.utils.data import Dataset

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
                normalize=False,
                maxlen=313,
                use_cnn=False):
        if type(dirs) != list:
            dirs = [dirs]
        self.dirs = dirs
        self.maxlen = maxlen
        self.normalize = True
        self.use_cnn = use_cnn
        self.audio_paths = [audio_path for dir in self.dirs for audio_path in list(Path(dir).iterdir())]
        self.prog = re.compile('(\w*)?_?id_(\d*)_(\d*)')

        self.n_mels, self.frames, self.n_fft = n_mels, frames, n_fft
        self.hop_length, self.power = hop_length, power
       
    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, index):
        res = {}
        audio_path = self.audio_paths[index]
        regex_res = self.prog.match(audio_path.name)
        res["label"], res["machine_id"], res["audio_id"] = regex_res.group(1, 2, 3)
        res["label"] = res["label"][:-1]  # remove trailing underscore
        res["path"] = str(audio_path)
        
        x = self.get_melspectrogram(audio_path)
        L = x.shape[0]
        if L < self.maxlen:
            x = torch.pad(x, (0, 0, 0, self.maxlen - L))
        elif L > self.maxlen:
            x = x[:self.maxlen, :]
            
        if self.use_cnn:
            x = x[:-1, :].unsqueeze(0)
            res["input"] = x
        else:
            res["input"] = x[:-1, :]
            res["target"] = x[1:, :]
        return res

    def get_melspectrogram(self, audio_path):
        wav, sr = torchaudio.load(audio_path)
        # print(f"{sr=}")
        # print(f"{wav.shape=}")
        melspec_transformer = T.MelSpectrogram(sample_rate=sr,
                                         n_fft=self.n_fft,
                                         hop_length=self.hop_length,
                                         n_mels=self.n_mels,
                                         power=self.power,
                                         normalized=False)
        res = melspec_transformer(wav).squeeze().T
        res = T.AmplitudeToDB(top_db=80., stype="power")(res)
        if self.normalize:
            min_level_db = -100
            res = torch.clip((res - min_level_db) / -min_level_db, 0, 1)

        return res
