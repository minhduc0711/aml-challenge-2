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

        self.n_mels = 128
        self.n_fft = 1024
        self.hop_length = 512
        self.power = 2.0
        
        self.maxlen = maxlen
        self.normalize = normalize
        self.use_cnn = use_cnn
        self.iter_over_cols = iter_over_cols
        
        self.subset_dirs = {
            "dev_train": data_dir / "dev_data/dev_data/slider/train",
            "dev_test": data_dir / "dev_data/dev_data/slider/test",
            "eval_train": data_dir / "eval_data/eval_data/slider/train",
            "eval_test": data_dir / "eval_data/eval_data/slider/test"
        }
        self.subsets = {}
        self.machine_ids = {
            "dev": ["00", "02", "04"],
            "eval": ["01", "03", "05"],
        }
        # for subset_name, subset_dir in subset_dirs.items():
        #     stage, split = subset_name.split("_")
        #     for machine_id in self.machine_ids[stage]:
        #         ds = SliderDataset(subset_dir,
        #                             n_mels=n_mels,
        #                             n_fft=n_fft,
        #                             hop_length=hop_length,
        #                             power=power,
        #                             machine_id=machine_id,
        #                             maxlen=maxlen,
        #                             use_cnn=use_cnn,
        #                             normalize=normalize,
        #                             iter_over_cols=iter_over_cols)
        #         if split == "train":
        #             # also create a validation set
        #             N = len(ds)
        #             num_train = int(0.9 * N)
        #             num_val = N - num_train
        #             self.subsets[f"{stage}_train_{machine_id}"], self.subsets[f"{stage}_val_{machine_id}"] = \
        #                     random_split(ds, [num_train, num_val])
        #         else:
        #             self.subsets[f"{subset_name}_{machine_id}"] = ds
        self.active_subsets = {}
    
    def setup_subset(self, stage, machine_id):
        if f"{stage}_train_{machine_id}" not in self.subsets.keys():
            for subset_name in [f"{stage}_train", f"{stage}_test"]:
                subset_dir = self.subset_dirs[subset_name]
                ds = SliderDataset(subset_dir,
                                    n_mels=self.n_mels,
                                    n_fft=self.n_fft,
                                    hop_length=self.hop_length,
                                    power=self.power,
                                    machine_id=machine_id,
                                    maxlen=self.maxlen,
                                    use_cnn=self.use_cnn,
                                    normalize=self.normalize,
                                    iter_over_cols=self.iter_over_cols)
                if subset_name == f"{stage}_train":
                    # also create a validation set
                    N = len(ds)
                    num_train = int(0.9 * N)
                    num_val = N - num_train
                    self.subsets[f"{stage}_train_{machine_id}"], self.subsets[f"{stage}_val_{machine_id}"] = \
                            random_split(ds, [num_train, num_val])
                else:
                    self.subsets[f"{subset_name}_{machine_id}"] = ds
            
        self.active_subsets["train"] = self.subsets[f"{stage}_train_{machine_id}"]
        self.active_subsets["val"] = self.subsets[f"{stage}_val_{machine_id}"]
        self.active_subsets["test"] = self.subsets[f"{stage}_test_{machine_id}"]
        
    def train_dataloader(self):
        if "train" not in self.active_subsets:
            raise RuntimeError("Please call setup_subset(stage, machine_id) first")
        return DataLoader(self.active_subsets["train"], 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        if "val" not in self.active_subsets:
            raise RuntimeError("Please call setup_subset(stage, machine_id) first")
        return DataLoader(self.active_subsets["val"], 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        if "test" not in self.active_subsets:
            raise RuntimeError("Please call setup_subset(stage, machine_id) first")
        return DataLoader(self.active_subsets["test"], 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)
