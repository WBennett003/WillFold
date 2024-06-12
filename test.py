from Models.Modules import AlphaFold3_TEMPLE
from Utils.parser import Parse_Cif
from Utils.input_encoding import process_entities
from Utils.extract_entities import parse_CIF
import torch
import matplotlib.pyplot as plt
import os
from torch.profiler import profile, record_function, ProfilerActivity

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def make_batched_to_device(f, batchify=False, device=None):
    fn = {}
    for key in f.keys():
        print(key, f[key].shape, f[key].sum())
        if batchify:
            fn[key] = f[key].unsqueeze(0).to(device)
        else:
            fn[key] = f[key].to(device)
    return fn

def testing_AF(run_profile=False):
    if torch.cuda.is_available():
        print("Found Cuda")
        device = torch.device("cuda")
    else:
        print("No GPU found")
        device = torch.device("cpu")
    print(f"Pytorch using device : {device}")

    entities = parse_CIF("data/1sld.cif")
    f = process_entities(entities)

    f = make_batched_to_device(f, batchify=True, device=device)

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
                    x, p_pae, p_pde, p_plddt, p_res = AF3(f)
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=25))
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=25))
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        prof.export_chrome_trace("trace.json")
    else:
        AF3 = AlphaFold3_TEMPLE(params).to(device)
        with torch.no_grad():
            x, p_pae, p_pde, p_plddt, p_res = AF3(f)
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    # plt.show()
    pass

def test_input_encoding():
    entities = parse_CIF("data/1sld.cif")
    f = process_entities(entities)
    pass

if __name__ == '__main__':
    testing_AF(True)
    # test_input_encoding()