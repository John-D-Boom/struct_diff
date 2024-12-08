import biotite.structure.io as bsio
import biotite.structure as bst

from esm.pretrained import ESM3_structure_encoder_v0, ESM3_structure_decoder_v0
from esm.utils.structure.protein_chain import ProteinChain

import os
import torch
import numpy as np



def get_alpha_carbon_coords(pdb_file, strucio_object = None, chain_id=None):
    """
    Extract coordinates of alpha carbons from a PDB file.
    
    Args:
        pdb_file (str): Path to the PDB file

        chain_id (str, optional): Specific chain to extract coordinates from
    
    Returns:
        numpy.ndarray: Nx3 array of alpha carbon coordinates
    """
    # Load the structure
    if strucio_object is not None:
        structure = strucio_object
    else:
        structure = bsio.load_structure(pdb_file)
    
    # Filter for alpha carbons
    alpha_carbons = structure[structure.atom_name == "CA"]
    
    # Optional chain filtering
    if chain_id is not None:
        alpha_carbons = alpha_carbons[alpha_carbons.chain_id == chain_id]
    
    # Extract coordinates
    coords = alpha_carbons.coord
    
    return coords

def get_protein_sequence(pdb_file, chain_id=None):
    """
    Extract protein sequence from a PDB file.
    
    Args:
        pdb_file (str): Path to the PDB file
        chain_id (str, optional): Specific chain to extract sequence from
    
    Returns:
        str: Amino acid sequence
    """
    # Load the structure
    structure = bsio.load_structure(pdb_file)
    
    # Filter for amino acid atoms
    amino_acids = structure[structure.atom_name == "CA"]
    
    # Optional chain filtering
    if chain_id is not None:
        amino_acids = amino_acids[amino_acids.chain_id == chain_id]
    
    # Convert to one-letter amino acid codes
    sequence = bst.seq1(amino_acids.res_name)
    
    return ''.join(sequence)

def get_esm_struct_token(esm_protein_chain, device = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESM3_structure_encoder_v0(device=device)
    model.eval()

    coords, _, residue_index = esm_protein_chain.to_structure_encoder_inputs()
    coords = coords.to(device)
    residue_index = residue_index.to(device)
    _, structure_tokens = model.encode(coords, residue_index=residue_index)

    return structure_tokens.squeeze().to("cpu").detach().numpy()

def get_ProteinChain(pdb_filepath, id=None, chain_id="detect"):
    """Returns an ESM3 ProteinChain object from a PDB file"""
  
    return ProteinChain.from_pdb(pdb_filepath, id=id, chain_id=chain_id)

def get_protein_inputs(pdb_filepath, strucio_object = None, id=None, chain_id=None):
    """
    One function that returns a dict of all of the inputs to the Protein class in data.protein 
    """

    #Get the esm_proteinchain object
    if chain_id is None:
        esm_chain_id = "detect"
    else:
        esm_chain_id = chain_id

    esm_protein_chain = get_ProteinChain(pdb_filepath, id=id, chain_id=esm_chain_id)
    struct_tokens = get_esm_struct_token(esm_protein_chain)
    sequence = esm_protein_chain.sequence

    positions = get_alpha_carbon_coords(pdb_filepath, 
                                        strucio_object= strucio_object, 
                                        chain_id=esm_protein_chain.chain_id)
    
    name = pdb_filepath.split("/")[-1].split(".")[0]
    

    return {"pdb_filepath": pdb_filepath, 
            "name": name, 
            "sequence": sequence, 
            "positions": positions, 
            "struct_tokens": struct_tokens}


def get_continuous_embeddings(struct_tokens, device = None):
    """
    Converts the structure tokens into continuous embeddings
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = ESM3_structure_decoder_v0(device=device)
    embeddings = decoder.embed(torch.tensor(struct_tokens, device = device))
    return embeddings.to("cpu").detach().numpy()





