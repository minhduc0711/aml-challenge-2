from pathlib import Path
import re
from tqdm import tqdm
import math
from functools import lru_cache

import torch
from torch.utils.data import Dataset

import torchaudio
import torchaudio.transforms as T

class SliderDataset(Dataset):
    """PyTorch dataset class for task2. Caching to a file supported.

    Args:
        n_mels, frames, n_fft, hop_length, power, transform: Audio conversion settings.
        normalize: Normalize data value range from [-90, 24] to [0, 1] for VAE, False by default.
        use_cnn: if True, add additional dimension to melspectrogram to represent num_channels
        iter_over_cols: if True, each data item is a column from a spectrogram, 
                        otherwise each item is a melspectrogram corresponding to one audio file
    """
    def __init__(self, dirs, n_mels, frames, n_fft, hop_length, power,
                normalize=False,
                maxlen=313,
                use_cnn=False,
                iter_over_cols=False):
        if type(dirs) != list:
            dirs = [dirs]
        self.dirs = dirs
        self.maxlen = maxlen
        self.normalize = True
        self.use_cnn = use_cnn
        self.audio_paths = [audio_path for d in self.dirs for audio_path in sorted(Path(d).iterdir())]
        self.prog = re.compile('(\w*)?_?id_(\d*)_(\d*)')

        self.n_mels, self.frames, self.n_fft = n_mels, frames, n_fft
        self.hop_length, self.power = hop_length, power
        
        self.iter_over_cols = iter_over_cols
        if iter_over_cols:
            idx2col = {}
            num_cols = 0
            for file_idx, audio_path in enumerate(tqdm(self.audio_paths, 
                                                       desc="Building index for spectrogram columns")):
                wav, sr = torchaudio.load(audio_path)
                num_frames = math.ceil((sr / hop_length) * (wav.shape[1] / sr))
                for i in range(num_frames):
                    idx2col[i + num_cols] = (file_idx, i)
                num_cols += num_frames
            self.idx2col = idx2col
            self.num_cols = num_cols
       
    def __len__(self):
        return len(self.audio_paths) if not self.iter_over_cols else self.num_cols

    def __getitem__(self, index):
        if not self.iter_over_cols:
            file_idx = index
        else:
            file_idx = self.idx2col[index][0]
        
        res = {}
        audio_path = self.audio_paths[file_idx]
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
        
        if not self.iter_over_cols:
            if self.use_cnn:
                x = x.unsqueeze(0)
                res["input"] = x
            else:
                res["input"] = x[:-1, :]
                res["target"] = x[1:, :]
        else:
            col_idx = self.idx2col[index][1]
            res["input"] = x[col_idx, :]
        return res

    @lru_cache(maxsize=500)
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
