import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from esm.pretrained import ESM3_structure_decoder_v0

import sys
sys.path.append('/burg/home/jb5005')

from struct_diff.model.layers import TransformerLayer, positional_encoding, time_encoding
from struct_diff.data.flow import interpolate

class BaseTransformerModel(pl.LightningModule):

    def __init__(self, 
                 lr=1e-3, 
                 d_model=1280,
                 num_heads=8, 
                 hidden_size = 1, 
                 dropout = 0.0, 
                 embedding_scale = 0.005):
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


class TransformerModel(pl.LightningModule):

    def __init__(self, 
                 lr=1e-3, 
                 d_model=1280,
                 n_blocks=1, 
                 num_heads=8, 
                 hidden_size = 1, 
                 dropout = 0.0, 
                 embedding_scale = 0.005,
                 source_scale = 0.001):
        super().__init__()

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                hidden_size=hidden_size,
                dropout=dropout
            ) for _ in range(n_blocks)
        ])

        
        self.lr = lr
        self.embedding_scale = embedding_scale
        self.source_scale = source_scale
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
        for layer in self.transformer_layers:
            x = layer(x)


        return x
    
    def training_step(self, x1, batch_idx):
        
        #Get random time uniformly sampled between 0 and 1. 
        # Remember 1 = x1 = data
        # 0 = x0 = noise

        mask = x1 != 0.0
        t = torch.rand(1, device = self.device)

        #Get noised version
        x_t, x0 = interpolate(x1, t, mask = mask, device= self.device, source_scale = self.source_scale)

        pred = self.forward(x_t, t)
        
        #Compute vector field, the target
        u_t = x1-x0


        loss = nn.functional.mse_loss(pred, u_t)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Run through a full reverse process and compute the distance to the structure tokens 
        """

        def convert_continuous_to_discrete_token(x1):
            """
            Maps backwards from the continuous token embedding to the discrete version
            Computes the Euclidean distance between the continuous token and all discrete tokens, 
            then selects discrete with the smallest distance

            args: x1 [BxLx1280]
            """
            decoder = ESM3_structure_decoder_v0(device=self.device)
            vocab = decoder.embed(torch.arange(0, 4100, device=self.device))  # [4100, 1280]
            
            # Ensure vocab matches the last dimension of x1
            if vocab.size(-1) != x1.size(-1):
                raise ValueError("Mismatch in embedding dimensions between x1 and vocab.")
            
            # Compute distances [B, L, 4100]
            distances = torch.cdist(x1, vocab, p=2)
            
            # Get the closest discrete tokens
            closest_indices = distances.argmin(dim=-1)  # [B, L]
            min_distances = torch.gather(distances, -1, closest_indices.unsqueeze(-1)).squeeze(-1)  # [B, L]
            
            return closest_indices.squeeze(), min_distances
        
        x_t = self.source_scale* torch.randn([100,128,1280], device = self.device)
        num_steps = 100
        # model.load_from_checkpoint()
        with torch.no_grad():
            for i in range(num_steps):
                t = i/num_steps
                x_t += 1/num_steps * self.forward(x_t, t)
        _, distances = convert_continuous_to_discrete_token(x_t)
        self.log("distance_to_closest_token", distances.mean())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        #Using Default parameters for now
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer}
