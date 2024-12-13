import sys
sys.path.append('/burg/home/jb5005')
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torch
import wandb

from struct_diff.data.flow import interpolate
from struct_diff.data.datamodule import ContinuousStructTokenDataModule
from struct_diff.model.models import TransformerModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_transformer_heads = 8
architecture = "transformer"
batching = "padding"

wandb_dir_path = '/pmglocal/jb5005/wandb_files'
os.makedirs(wandb_dir_path, exist_ok = True)

wandb_logger = WandbLogger(name = "emb_0.01", project="struct_diff_v0", save_dir = wandb_dir_path)
wandb_logger.experiment.config["batching"] = batching
wandb_logger.experiment.config["architecture"] = architecture
wandb_logger.experiment.config["nhead"] = num_transformer_heads

save_path = os.path.join('/manitou/pmg/users/jb5005/struct_diff_data/results', wandb_logger.experiment.name)


#Load the data
datamodule = ContinuousStructTokenDataModule(base_path = '/pmglocal/jb5005/struct_diff_data/data', 
                                             df_path = '/burg/home/jb5005/struct_diff/data/seq_len.csv',
                                             num_workers = 0,
                                             decoder = None)
datamodule.setup('train')

#Setup model
model = TransformerModel(lr = 1e-4, hidden_size=2, embedding_scale= 0.01)
model = model.to(device)

trainer = pl.Trainer(max_epochs=2500, enable_progress_bar=False, logger = wandb_logger)

trainer.fit(model = model, datamodule= datamodule)

torch.save(trainer, os.path.join(save_path, "trainer.pth"))
torch.save(model.state_dict(), os.path.join(save_path,'model_state_dict.pth'))
torch.save(datamodule, os.path.join(save_path, "datamodule.pth"))

