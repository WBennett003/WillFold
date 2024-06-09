import torch
from rdkit import Chem
from pdbeccdutils.core import ccd_reader
from pdbeccdutils.core.ccd_reader import CCDReaderResult
import json
import requests

with open("token_ref.json", 'r') as f:
    token_ref = json.load(f)

class CCD_manager:
    def __init__(self, reference_filepath='Reference_Confomer/', json_name='reference_conformer.json'):
        self.path = reference_filepath
        self.json_filename = json_name
        with open(reference_filepath+json_name, 'r') as f:
            self.ref = json.load(f)
    

    def get_ccd(self, ccd_code):
        if ccd_code in self.ref['CCD']:
            with open(self.path+ccd_code+'.sdf', 'r') as f:
                r = f.read()
            mol = Chem.MolFromMolBlock(r)
            
        else:
            r = requests.get("https://www.ebi.ac.uk/pdbe-srv/pdbechem/chemicalCompound/show/"+ccd_code+".sdf").text
            mol = Chem.MolFromMolBlock(r)
            mol = Chem.RemoveAllHs(mol)      
            with Chem.SDWriter(self.path+ccd_code+'.sdf') as w:
                for m in mol:
                    w.write(m)

            self.ref['CCD'].append(ccd_code)

            with open(self.path+self.json_filename, 'w') as f:
                json.dump(self.ref)     

        return mol

Element_dict = {
    "H" : 1,
    "He" : 2,
    "Li" : 3,
    "Be" : 4,
    "B" : 5,
    "C" : 6,
    "N" : 7,
    "O" : 8,
    "F" : 9,
    "Ne" : 10,
    "Na" : 11,
    "Mg" : 12,
    "Al" : 13,
    "Si" : 14,
    "P" : 15,
    "S" : 16,
    "Cl" : 17,
    "Ar" : 18,
    "K" : 19,
    "Ca" : 20,
    "Sc" : 21, 
    "Ti" : 22,
    "V" : 23,
    "Cr" : 24,
    "Mn" : 25,
    "Fe" : 26,
    "Co" : 27,
    "Ni" : 28,
    "Cu" : 29,
    "Zn" : 30,
    "Ga" : 31,
    "Ge" : 32,
    "As" : 33,
    "Se" : 34,
    "Br" : 35,
    "Kr" : 36,
    "Rb" : 37,
    "Sr" : 38,
    "Y" : 39,
    "Zr" : 40,
    "Nb" : 41,
    "Mo" : 42,
    "Tc" : 43,
    "Ru" : 44,
    "Rh" : 45,
    "Pd" : 46,
    "Ag" : 47,
    "Cd" : 48
}

def generate_atoms(residue, pad_size=4, N_max_atoms=24):
    atoms = token_ref[residue]['atoms']
    length = len(atoms)
    atom_mask = torch.zeros(N_max_atoms)
    atom_mask[:length] = 1.0

    elements = [Element_dict[a[0]] for a in atoms]

    atom_padding = ['    ' for i in range(N_max_atoms - length)] 
    atoms.extend(atom_padding)
    for i, a in enumerate(atoms):
        atoms[i] = a + ' '* (pad_size-len(a))
        elements.append(a[0])

    ref_atoms_chars = [[ord(c)-32 for c in a] for a in atoms]

    ref_atom_idx = token_ref[residue]['ref_atom_idx']

    

    return atoms, ref_atoms_chars, atom_mask, elements, ref_atom_idx

def process_entities(entities, ccd_reader_results=None, N_max_atoms=24):
    tokens = []
    res_types = []
    ref_atom_name_chars = []
    ref_mask = []
    ref_pos = []
    ref_elements = []
    ref_atom_idx = []

    total_tokens = []
    total_atoms = []
    entity_ids = []
    asym_id = []
    sym_id = []

    is_protein = []
    is_rna = []
    is_dna = []
    is_ligand = []

    asym_count = 0

    for i, entity in enumerate(entities):
        entity_tokens = []
        entity_atoms = []
               
        for token in entity:
            if token in list(token_ref.keys()):

                if token_ref[token]['mol_type'] == 'protein':
                    is_ligand.append(0)
                    is_protein.append(1)
                    is_rna.append(0)
                    is_dna.append(0)

                    res_types.append(token_ref[token]['token_idx'])
                    entity_tokens.append(token)
                    atoms, atom_name_char, mask, elements, ref_idx = generate_atoms(token)
                    entity_atoms.append(atoms)
                    ref_atom_name_chars.append(atom_name_char)
                    ref_mask.append(mask)
                    ref_elements.append(elements)
                    ref_atom_idx.append(ref_idx)

                elif token_ref[token]['mol_type'] == 'dna':
                    is_ligand.append(0)
                    is_protein.append(0)
                    is_rna.append(0)
                    is_dna.append(1)

                    res_types.append(token_ref[token]['token_idx'])
                    entity_tokens.append(token)
                    atoms, atom_name_char, mask, elements, ref_idx = generate_atoms(token)
                    entity_atoms.append(atoms)
                    ref_atom_name_chars.append(atom_name_char)
                    ref_mask.append(mask)
                    ref_elements.append(elements)
                    ref_atom_idx.append(ref_idx)

                elif token_ref[token]['mol_type'] == 'rna':
                    is_ligand.append(0)
                    is_protein.append(0)
                    is_rna.append(0)
                    is_dna.append(1)


                    res_types.append(token_ref[token]['token_idx'])
                    entity_tokens.append(token)
                    atoms, atom_name_char, mask, elements, ref_idx = generate_atoms(token)
                    entity_atoms.append(atoms)
                    ref_atom_name_chars.append(atom_name_char)
                    ref_mask.append(mask)
                    ref_elements.append(elements)
                    ref_atom_idx.append(ref_idx)

        else:
            is_ligand.append(1)
            is_protein.append(0)
            is_rna.append(0)
            is_dna.append(0)


            #NOTE : Need to fix this to get atom generation.
            atoms, atoms_names,  = generate_atoms(entity)


            padding = ((N_max_atoms - len(atoms)) * '<PAD> ').split(' ')

            for atom in atoms:
                res_types.append(0)
                entity_tokens.append("UNK")
                ass = [atom]
                ass.append(padding)
                
                entity_atoms.append(ass)
                entity_ids.append(i)    


        if entity_tokens in total_tokens:
            idx = total_atoms.index(entity_tokens)
            sym_idx = total_tokens.count(entity_tokens)
        else:
            idx = asym_count + 1
            asym_count += 1    
            sym_idx = 0

        asym_id.extend([idx for i in range(len(entity_tokens))])
        sym_id.extend([sym_idx for i in range(len(entity_tokens))])
        total_tokens.append(entity_tokens)
        total_atoms.append(entity_atoms)
        tokens.extend(entity_tokens)
        atoms.extend(entity_atoms)

            

