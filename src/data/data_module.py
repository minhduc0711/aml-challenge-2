from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from .dataset import SliderDataset

class SliderDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data/raw/",
                 batch_size: int = 32,
                 maxlen: int = 312,
                 num_workers: int = 4,
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
        
        subset_dirs = {
            "train": data_dir / "dev_data/dev_data/slider/train",
            "test": data_dir / "dev_data/dev_data/slider/test",
            "predict": data_dir / "eval_data/eval_data/slider/test"
        }
        self.subsets = {}
        for subset_name, subset_dir in subset_dirs.items():
            ds = SliderDataset(subset_dir,
                                n_mels=n_mels,
                                frames=frames,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                power=power,
                                maxlen=maxlen,
                                use_cnn=use_cnn,
                                normalize=normalize,
                                iter_over_cols=iter_over_cols)
            if subset_name == "train":
                # also create a validation set
                N = len(ds)
                num_train = int(0.8 * N)
                num_val = N - num_train
                self.subsets["train"], self.subsets["val"] = \
                        random_split(ds, [num_train, num_val])
            else:
                self.subsets[subset_name] = ds

    def train_dataloader(self):
        return DataLoader(self.subsets["train"], 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.subsets["val"], 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.subsets["test"], 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
    
    def predict_dataloader(self):
        return DataLoader(self.subsets["predict"], 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

