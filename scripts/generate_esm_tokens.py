""""
This is the main script for generating the ESM3 structure tokens used a dataset
for the rest of the model
"""

import os
import time
import sys
import pickle as pkl
sys.path.append('/burg/home/jb5005')

import biotite.structure.io as bsio
import biotite.structure as bst

from struct_diff.utils.protein_io import get_protein_inputs

mmcif_path = '/manitou/pmg/projects/resources/openfold_data/data/pdb_mmcif/mmcif_files'
mmcif_arr = os.listdir(mmcif_path)

pdb_write_path = '/pmglocal/jb5005/struct_diff_data/pdb'
data_write_path = '/pmglocal/jb5005/struct_diff_data/data'

os.makedirs(pdb_write_path, exist_ok=True)
os.makedirs(data_write_path, exist_ok=True)

mmcif_index = 0
write_index = 0
start = time.time()

while write_index < 20000:

    if write_index % 100 == 0:
        print("Time for last hundred: ", write_index, time.time() - start)
        print("estimated hours left: ", (20000 - write_index)/100 * (time.time() - start)/3600)
        start = time.time()

    pdb_name = mmcif_arr[mmcif_index].split(".")[0]
    file_path = os.path.join(mmcif_path, mmcif_arr[mmcif_index])
    pdb_path = os.path.join(pdb_write_path, pdb_name + ".pdb")

    try:
        cif_bio_struct = bsio.load_structure(file_path)

        bsio.save_structure(pdb_path, cif_bio_struct)
        

        protein_inputs = get_protein_inputs(pdb_path, 
                                            strucio_object=cif_bio_struct, 
                                            id = pdb_name,
                                            chain_id = None)
        
        if len(protein_inputs["sequence"]) <= 300:
            data_path = os.path.join(data_write_path, pdb_name + ".pkl")
            with open(data_path, 'wb') as f:
                pkl.dump(protein_inputs, f)

            write_index +=1
        else:
            print(pdb_name, "too long")
    
    except Exception as e:
        print(pdb_name, mmcif_index, e)

    mmcif_index +=1




    

