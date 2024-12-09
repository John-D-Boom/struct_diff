import torch
import lightning as l
import pandas as pd
from torch.utils.data import Dataset
import os
import numpy as np
import pickle as pkl
import time

from esm.pretrained import ESM3_structure_encoder_v0, ESM3_structure_decoder_v0

import sys
sys.path.append('/burg/home/jb5005')
from struct_diff.data.protein import protein_from_dict


class ContinuousStructTokenDataset(Dataset):
    """
    Class used for loading the single representation of AlphaFold and preparing neural network

    Args:
        base_path (str): Path to the directory containing the data. each 
        df_path (str): Path to the dataframe containing the full list of PDBs to use
    """
    
    # base_path = '/pmglocal/jb5005/struct_diff_data/data
    
    def __init__(self, base_path, df_path, min_length = None, max_length = None, decoder = None):


        self.df = pd.read_csv(df_path)
        
        #Filter out sequences that are too long from dataset
        if max_length is not None:
            self.df = self.df[self.df['seq_len'] <= max_length]
        if min_length is not None:
            self.df = self.df[self.df['seq_len'] >= min_length]

        self.base_path = base_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if decoder is None:
            self.decoder = ESM3_structure_decoder_v0(self.device)
        else:
            self.decoder = decoder

    def __len__(self):
        return len(self.df)
    

    def __getitem__(self, idx):
        pdb_name = self.df.iloc[idx]['name']
        with open(os.path.join(self.base_path, pdb_name + '.pkl'), 'rb') as f:
            protein_dict = pkl.load(f)

        protein = protein_from_dict(protein_dict)
        cont_emb = self.decoder.embed(torch.tensor(protein.get_struct_sequence(), device = self.device))
        return torch.tensor(cont_emb)
