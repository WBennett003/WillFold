import torch
from torch.utils.data import Dataset
from Utils.extract_entities import parse_CIF, extract_coordinates
from Utils.input_encoding import process_entities


def make_batched_to_device(f, batchify=False, device=None):
    fn = {}
    for key in f.keys():
        print(key, f[key].shape, f[key].sum())
        if batchify:
            fn[key] = f[key].unsqueeze(0).to(device)
        else:
            fn[key] = f[key].to(device)
    return fn

def collate(batch):
    batched_f = {}
    for key in batch[0][0].keys():
        batched_f[key] = []

    batched_xyz = []
    for i,b in enumerate(batch):
        for key in b[0].keys():
            batched_f[key].append(b[0][key])
        batched_xyz.append(b[1])
    
    for key in batched_f.keys():
        batched_f[key] = torch.concat(batched_f[key], dim=0)
    
    batched_xyz = torch.concat(batched_xyz, dim=0)
    return batched_f, batched_xyz

class mmcif_dataset(Dataset):
    def __init__(self, mmcif_path, file_suffix='.cif', samples=[], Ntokens=None, Natoms=None, device=None) -> None:
        super().__init__()

        self.mmcif_path = mmcif_path
        self.file_suffix = file_suffix
        self.samples = samples
        self.Ntokens = Ntokens
        self.Natoms = Natoms
        self.device = device

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        file_loc = self.mmcif_path+self.samples[index]+self.file_suffix
        entities, bonds = parse_CIF(file_loc, get_bonds=True)
        f = process_entities(entities, bonds, padding_size=4) #TODO: NEeds to pad for batching!!!
        f = make_batched_to_device(f, True, self.device)
        xyz = extract_coordinates(file_loc).unsqueeze(0).to(self.device) # TODO: check that the atoms are resolved at the start and add cropping
        return f, xyz
