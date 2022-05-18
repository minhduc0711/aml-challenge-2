from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from .dataset import SliderDataset

class SliderDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/raw/",
                 batch_size: int = 32,
                 num_workers=4,
                 normalize=False,
                 use_cnn=False,
                 iter_over_cols=False):
        super().__init__()
        data_dir = Path(data_dir)
        self.num_workers = num_workers
        self.batch_size = batch_size

        n_mels = 128
        frames = 5
        n_fft = 1024
        hop_length = 512
        power = 2.0

        self.ds = SliderDataset(data_dir / "dev_data/dev_data/slider/train",
                                n_mels=n_mels,
                                frames=frames,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                power=power,
                                use_cnn=use_cnn,
                                normalize=normalize,
                                iter_over_cols=iter_over_cols)
        N = len(self.ds)
        num_train = int(0.8 * N)
        num_val = N - num_train
        self.train_ds, self.val_ds = random_split(self.ds, [num_train, num_val])

    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
