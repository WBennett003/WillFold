from Models.Modules import AlphaFold3_TEMPLE
from Utils.parser import Parse_Cif
import torch
import matplotlib.pyplot as plt
import os
from torch.profiler import profile, record_function, ProfilerActivity

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def testing_AF(run_profile=False):
    if torch.cuda.is_available():
        print("Found Cuda")
        device = torch.device("cuda")
    else:
        print("No GPU found")
        device = torch.device("cpu")
    print(f"Pytorch using device : {device}")

    residue_index1, token_index1, asym_id1, entity_id1, sym_id1, restype1, is_protein1, is_dna1, is_rna1, is_ligand1, ref_pos1, ref_mask1, ref_element1, ref_charge1, ref_atom_name_chars1, ref_space_uid1, token_bonds, token_idx_for_atoms1, refernce_token_atoms, b_factor1, occ1 = Parse_Cif('data/1sld.cif')
    # residue_index2, token_index2, asym_id2, entity_id2, sym_id2, restype2, is_protein2, is_dna2, is_rna2, is_ligand2, ref_pos2, ref_mask2, ref_element2, ref_charge2, ref_atom_name_chars2, ref_space_uid2, token_idx_for_atoms2, b_factor2, occ2 = Parse_Cif('')

    f = {
        "residue_idx" : residue_index1.unsqueeze(0).to(device), 
        "token_idx" : token_index1.unsqueeze(0).to(device), 
        "asym_id" : asym_id1.unsqueeze(0).to(device), 
        "entity_id" : entity_id1.unsqueeze(0).to(device), 
        "sym_id" : sym_id1.unsqueeze(0).to(device), 
        "restype" : torch.nn.functional.one_hot(restype1, 32).unsqueeze(0).to(device),
        "is_protein" : is_protein1.unsqueeze(0).to(device), 
        "is_dna" : is_dna1.unsqueeze(0).to(device), 
        "is_rna" : is_rna1.unsqueeze(0).to(device), 
        "is_ligand" : is_ligand1.unsqueeze(0).to(device), 
        "ref_pos" : ref_pos1.unsqueeze(0).to(device), 
        "ref_mask" : ref_mask1.unsqueeze(0).to(device), 
        "ref_element" : torch.nn.functional.one_hot(ref_element1, 128).unsqueeze(0).to(device),
        "ref_charge" : ref_charge1.unsqueeze(0).to(device), 
        "ref_atom_name_chars" : torch.nn.functional.one_hot(ref_atom_name_chars1, 64).unsqueeze(0).to(device), 
        "ref_space_uid" : ref_space_uid1.unsqueeze(0).to(device), 
        "token_bonds" : token_bonds.unsqueeze(0).to(device),
        "token_idx_for_atoms" : token_idx_for_atoms1.unsqueeze(0).to(device), 
        "reference_tokens" : refernce_token_atoms.unsqueeze(0).to(device)
    }

    
    Ntokens = f['residue_idx'].shape[1]
    Natoms = f['ref_pos'].shape[1]

    params ={
        "s_dim" : 384 + 32,
        "Ntokens" : Ntokens,
        "Natoms" : Natoms,
        "rmax" : 32,
        "smax" : 2,
        "AAE_f_sum_dim" : 389, 
        "AAE_nblocks" : 1, # was 3
        "AAE_nheads" : 4,
        "AAE_nqueries" : 32,
        "AAE_mkeys" : 128,
        "AAE_transition_n" : 2,
        "AAE_ctoken" : 384,
        "AAD_nheads" : 4,
        "AAD_nblocks" : 1, #3
        "IFE_s_size" : 384 + 32,
        "AT_nblock" : 1, # was 3,
        "AT_nheads" : 4,
        "AT_nqueries" : 32,
        "AT_nkeys" : 128,
        "S_subset_centres" : [15.5, 47.5, 79.5],
        "cz" : 128,
        "cs" : 384,
        "ca" : 384,
        "DM_ctokens" : 768,
        "DM_DT_nblocks" : 1, #24,
        "DM_DT_nheads" : 16,
        "DT_transition_scale" : 2,
        "catom_pairs" : 16,
        "catoms" : 128,
        "sigma_data" : 16,
        "gamma_0" : 0.8,
        "gamma_min" : 1.0,
        "noise_scale_lambda" : 1.003,
        "step_scale_eta" : 1.5,
        "strans" : 1,
        "timesteps" : 2,
        "Pairformer_nblocks" : 1, # was 48,
        "Pairformer_pair_nheads" : 16,
        "Pairformer_drop_p" : 0.25,
        "Pairformer_transition_scale" : 4,
        "Triangle_attention_channel" : 32,
        "Triangle_attention_nheads" : 4,
        "Triangle_multi_c" : 128,
        "pae_c" : 64,
        "pde_c" : 64,
        "plddt_c" : 50,
        "pres_c" : 2,
        "Conf_Head_nblocks" : 4,
        "Fourier_n" : 256,
        "N_cycles" : 1,
        "device" : device
    }
    if run_profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
            with record_function("testing"):
                AF3 = AlphaFold3_TEMPLE(params).to(device)
                with torch.no_grad():
                    x = AF3(f)
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=25))
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=25))
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        prof.export_chrome_trace("trace.json")
    else:
        AF3 = AlphaFold3_TEMPLE(params).to(device)
        with torch.no_grad():
            x = AF3(f)[0].cpu()
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    # plt.show()
    pass



if __name__ == '__main__':
    testing_AF(True)