import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from struct_diff.data.dataset import ContinuousStructTokenDataset
from torch.nn.utils.rnn import pad_sequence


class ContinuousStructTokenDataModule(pl.LightningDataModule):

    def __init__(self, base_path, df_path, num_workers = 1, batch_size = 32, decoder = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.base_path = base_path
        self.df_path = df_path
        self.decoder = decoder
        self.dataset = None

    def setup(self, stage: str):
        
        self.dataset = ContinuousStructTokenDataset(base_path = self.base_path, 
                                                    df_path = self.df_path,
                                                    decoder = self.decoder)
        
        self.seed = torch.Generator().manual_seed(0)
    
    def train_dataloader(self):

        def custom_collate_fn(batch):
            # Pad the sequences
            return pad_sequence(batch, batch_first=True)  # Default output [L, B, D]
            # Permute to make batch dimension first

        return DataLoader(self.dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          collate_fn = custom_collate_fn, 
                          num_workers = self.num_workers)
    
    def val_dataloader(self):

        """
        "fake input to appease the pytorch_lightning gods and run validation step
        """
        return torch.zeros(1,128,1280)

    