import torch
from torch.utils.data import Dataset
from Utils.extract_entities import parse_CIF, extract_coordinates
from Utils.input_encoding import process_entities

class mmcif_dataset(Dataset):
    def __init__(self, mmcif_path, file_suffix='.cif', samples=[]) -> None:
        super().__init__()

        self.mmcif_path = mmcif_path
        self.file_suffix = file_suffix
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        file_loc = self.mmcif_path+self.samples[index]+self.file_suffix
        entities, bonds = parse_CIF(file_loc, get_bonds=True)
        f = process_entities(entities, bonds, padding_size=4)
        xyz = extract_coordinates(file_loc) # TODO: check that the atoms are resolved at the start and add cropping
        return f, xyz
