import sys
from pathlib import Path
import re
from tqdm.auto import tqdm
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
    def __init__(self, dirs, n_mels, n_fft, hop_length, power,
                normalize=False,
                maxlen=313,
                use_cnn=False,
                iter_over_cols=False,
                machine_id: int = None):
        if type(dirs) != list:
            dirs = [dirs]
        self.dirs = dirs
        self.maxlen = maxlen
        self.normalize = normalize
        self.use_cnn = use_cnn
        
        self.prog = re.compile('(\w*)?_?id_(\d*)_(\d*)')

        audio_paths = [audio_path for d in self.dirs for audio_path in sorted(Path(d).iterdir())]
        # use only subset of audios belonging to 1 certain machine if wanted
        if machine_id is None:
            self.audio_paths = audio_paths
        else:
            self.audio_paths = []
            for p in audio_paths:
                regex_res = self.prog.match(p.name)
                audio_machine_id = regex_res.group(2)
                if audio_machine_id == machine_id:
                    self.audio_paths.append(p)
            
        self.n_mels, self.n_fft = n_mels, n_fft
        self.hop_length, self.power = hop_length, power
        
        self.iter_over_cols = iter_over_cols
        if iter_over_cols:
            # idx2col = {}
            # num_cols = 0
            # # n_sliding_frames = 5
            # for file_idx, audio_path in enumerate(tqdm(self.audio_paths, 
            #                                            desc="Building index for spectrogram columns")):
            #     wav, sr = torchaudio.load(audio_path)
            #     num_frames = math.ceil((sr / hop_length) * (wav.shape[1] / sr))
            #     # num_frames = num_frames - n_sliding_frames + 1
            #     for i in range(num_frames):
            #         idx2col[i + num_cols] = (file_idx, i)
            #     num_cols += num_frames
            # self.idx2col = idx2col
            # self.num_cols = num_cols
            self.X = []
            self.y = []
            for file_idx, audio_path in enumerate(tqdm(self.audio_paths, 
                                                       desc="Producing melspectrograms for all audios")):
                melspec = self.get_melspectrogram(audio_path)
                self.X.append(melspec)
                regex_res = self.prog.match(audio_path.name)
                label = regex_res.group(1)[:-1]
                self.y.extend([label] * melspec.shape[0])
            self.X = torch.cat(self.X, dim=0)
       
    def __len__(self):
        # return len(self.audio_paths) if not self.iter_over_cols else self.num_cols
        return len(self.audio_paths) if not self.iter_over_cols else self.X.shape[0]

    def __getitem__(self, index):
        # if not self.iter_over_cols:
        #     file_idx = index
        # else:
        #     file_idx = self.idx2col[index][0]
        res = {}
        
        if self.iter_over_cols:
            return {
                "input": self.X[index],
                "label": self.y[index]
            }
                              
        file_idx = index

        audio_path = self.audio_paths[file_idx]
        regex_res = self.prog.match(audio_path.name)
        res["label"], res["machine_id"], res["audio_id"] = regex_res.group(1, 2, 3)
        res["label"] = res["label"][:-1]  # remove trailing underscore
        res["path"] = str(audio_path)
        
        x = self.get_melspectrogram(audio_path)
        L = x.shape[0]
        
        # if not self.iter_over_cols:
        if L < self.maxlen:
            x = torch.pad(x, (0, 0, 0, self.maxlen - L))
        elif L > self.maxlen:
            x = x[:self.maxlen, :]

        if self.use_cnn:
            x = x.unsqueeze(0)
            res["input"] = x
        else:
            res["input"] = x[:-1, :]
            res["target"] = x[1:, :]
        # else:
        #     col_idx = self.idx2col[index][1]
        #     res["input"] = x[col_idx, :]
        return res

    # @lru_cache(maxsize=10000)
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
        # res = 20.0 / self.power * torch.log10(res + sys.float_info.epsilon)
        if self.normalize:
            min_level_db = -100
            res = torch.clip((res - min_level_db) / -min_level_db, 0, 1)
        
        # n_frames = 5
        # frame_len = res.shape[0] - n_frames + 1
        # frames = []
        # for i in range(n_frames):
        #     frames.append(res[i : frame_len + i, :]) 
        # res = torch.cat(frames, dim=1)
        # print(res.shape)

        return res
