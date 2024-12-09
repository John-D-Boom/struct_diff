import numpy as np
import torch
from esm.pretrained import ESM3_structure_encoder_v0, ESM3_structure_decoder_v0
import sys
sys.path.append('/burg/home/jb5005')

from struct_diff.utils.protein_io import get_continuous_embeddings

def protein_from_dict(protein_dict):
    """
    Creates a Protein object from a dictionary
    """
    return Protein(name = protein_dict['name'], 
                   pdb_filepath = protein_dict['pdb_filepath'], 
                   struct_sequence = protein_dict['struct_tokens'], 
                   c_alpha_pos = protein_dict['positions'],
                   sequence = protein_dict['sequence'])

class Protein:

    def __init__(self, name, pdb_filepath, struct_sequence, c_alpha_pos, sequence):
        
        self.name = name
        self.pdb_filepath = pdb_filepath
        self.struct_sequence = struct_sequence
        self.c_alpha_pos = c_alpha_pos
        self.sequence = sequence

    def get_struct_sequence(self):
        """
        Returns the structural sequence of the protein
        """
        return self.struct_sequence
    
    def get_c_alpha_pos(self):
        """
        Returns the positions of the C-alpha of each residue
        """
        return self.c_alpha_pos

    def get_name(self):
        """
        Returns the name of the protein
        """
        return self.name
    
    def get_continuous_vector(self):
        """
        Returns a continuous vector representation of the protein
        """        
        
        return get_continuous_embeddings(self.struct_sequence)



    

