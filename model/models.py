import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import sys
sys.path.append('/burg/home/jb5005')

from struct_diff.model.layers import TransformerLayer, positional_encoding, time_encoding
from struct_diff.data.flow import interpolate

class TransformerModel(pl.LightningModule):

    def __init__(self, lr=1e-3, d_model=1280, num_heads=8, hidden_size = 1, dropout = 0.0, embedding_scale = 0.005):
        super().__init__()

        self.trans_layer = TransformerLayer(
            d_model=d_model, 
            num_heads=num_heads, 
            hidden_size=hidden_size,
            dropout=dropout
        )
        self.lr = lr
        self.embedding_scale = embedding_scale
        self.save_hyperparameters()

    def forward(self, x, t):
        mask = x != 0.0 #padding value is 0.0

        #Add positional and time encoding, only on non-padded values
        pos_enc = positional_encoding(x.shape[1], 
                                      x.shape[2], 
                                      scale = self.embedding_scale).unsqueeze(0) #unsqueeze broadcasts across batch
        
        pos_enc = pos_enc.to(self.device)

        time_enc = time_encoding(t, 
                                 x.shape[1], 
                                 x.shape[2], 
                                 scale = self.embedding_scale).unsqueeze(0)
        
        time_enc = time_enc.to(self.device)

        x = x + mask * pos_enc
        x = x + mask * time_enc

        #Run through model
        x = self.trans_layer(x)

        return x
    
    def training_step(self, x1, batch_idx):
        
        #Get random time uniformly sampled between 0 and 1. 
        # Remember 1 = x1 = data
        # 0 = x0 = noise

        mask = x1 != 0.0
        t = torch.rand(1, device = self.device)

        #Get noised version
        x_t, x0 = interpolate(x1, t, mask = mask, device= self.device, source_scale = 0.001)

        pred = self.forward(x_t, t)
        
        #Compute vector field, the target
        u_t = x1-x0


        loss = nn.functional.mse_loss(pred, u_t)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pass


    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        #Using Default parameters for now
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}
