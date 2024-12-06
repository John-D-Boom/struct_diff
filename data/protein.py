import numpy as np
import torch
from esm.pretrained import ESM3_structure_encoder_v0, ESM3_structure_decoder_v0
import biotite.structure as struc
import biotite.database.pdb as pdb


class Protein:

    def __init__(self, name, pdb_filepath, struct_sequence):
        self.name = name
        self.pdb_filepath = pdb_filepath
        self.struct_sequence = struct_sequence
        self.c_alpha_pos = self.compute_c_alpha_pos(pdb_filepath)

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = ESM3_structure_encoder_v0(device=device)
        
        return None


    def compute_c_alpha_pos(self, pdb_filepath):
        """
        Args:
            pdb_filepath: Path to the pdb file
        Returns:
            The positions of the C-alpha of each residue
        """
        pdb_file = pdb.load(pdb_filepath)
        
        # Get the structure array from the PDB file
        structure = struc.load_structure(pdb_file)
        
        # Get the C-alpha atoms from the structure
        c_alpha_atoms = structure[struc.ATOM_NAME == 'CA']
        
        # Extract the 3D positions of the C-alpha atoms
        c_alpha_positions = c_alpha_atoms.coord
        
        assert c_alpha_positions.shape[0] == len(self.struct_sequence)
        assert c_alpha_positions.shape[1] == 3
        
        return c_alpha_positions    


    

