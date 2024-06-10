import torch
from rdkit import Chem
from pdbeccdutils.core import ccd_reader
from pdbeccdutils.core.ccd_reader import CCDReaderResult
import json
import requests

with open("token_ref.json", 'r') as f:
    token_ref = json.load(f)

Element_dict = {
    " " : 0,
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

def remove_condense_atoms():
    pass

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
            r = requests.get("https://www.ebi.ac.uk/pdbe-srv/pdbechem/chemicalCompound/show/"+ccd_code+"_ideal.sdf").text
            mol = Chem.MolFromMolBlock(r)
            mol = Chem.RemoveAllHs(mol)    
            if ccd_code in list(token_ref.keys()):
                match token_ref[ccd_code]['mol_type']:
                    case "protein":
                        #remove the carbonates hydroxyl
                        mol = Chem.rdchem.EditableTpMol(mol)
                        natoms = len([a for a in mol.GetAtoms()])
                        mol.RemoveAtom(natoms)
                        mol = mol.GetMol()
                    case "dna":
                        mol = Chem.rdchem.EditableTpMol(mol)
                        mol.RemoveAtom(2)
                        mol = mol.GetMol()
                    case "rna":
                        mol = Chem.rdchem.EditableTpMol(mol)
                        mol.RemoveAtom(2)
                        mol = mol.GetMol()

            with Chem.SDWriter(self.path+ccd_code+'.sdf') as w:
                for m in mol:
                    w.write(m)

            self.ref['CCD'].append(ccd_code)

            with open(self.path+self.json_filename, 'w') as f:
                json.dump(self.ref)     

        #get atoms
        atoms_name = []
        atoms_pos = []
        for atoms in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atoms.GetIdx())
            atoms_pos.append([pos.x, pos.y, pos.z])
            atoms_name.append(atoms.GetSymbol())

        bonds = []
        for bond in mol.GetBonds():
            ai = bond.GetBeginAtomIdx()
            aj = bond.GetEndAtomIdx()
            bonds.append([ai, aj])

        return mol, atoms_name, atoms_pos, bonds

def generate_atoms(entity, pad_size=4, N_max_atoms=24, parser=None):
    atoms = token_ref[entity]['atoms']
    length = len(atoms)

    mol, atoms_name, atoms_pos, bonds = parser.get_ccd(entity)

    elements = [Element_dict[a[0]] for a in atoms]
    assert elements == atoms_name

    for i, a in enumerate(atoms):
        atoms[i] = a + ' '* (pad_size-len(a))

    ref_atoms_chars = [[ord(c)-32 for c in a] for a in atoms]

    ref_atom_idx = token_ref[entity]['ref_atom_idx']

    return atoms, ref_atoms_chars, elements, ref_atom_idx, atoms_pos

def generate_ligand_atoms(entity, pad_size=4, N_max_atoms=24, parser=None):
    mol, atoms_name, atoms_pos, bonds = parser.get_ccd(entity)
    length = len(atoms_name)
    
    ref_atoms_chars = []
    elements = []

    for i, atom in enumerate(atoms_name):
        elements.append(atom[0])

        atom  += ' ' * (pad_size - len(atom))
        ref_atoms_chars.append([ord(c) for c in atom])
    
    return atoms_name, atoms_pos, ref_atoms_chars, elements, bonds




def process_entities(entities, implicit_token_bonds=None, padding_size=4, N_max_atoms=24):

    ccd_manager = CCD_manager()

    tokens = []
    res_types = []
    ref_atom_name_chars = []
    ref_mask = []
    ref_pos = []
    ref_elements = []
    ref_atom_idx = []
    ref_space_uid = []
    chain_set_pairs = []

    total_tokens = []
    total_atoms = []
    entity_ids = []
    asym_id = []
    sym_id = []

    is_protein = []
    is_rna = []
    is_dna = []
    is_ligand = []

    token_bonds = implicit_token_bonds

    asym_count = 0

    for i, entity in enumerate(entities):
        entity_tokens = []
        entity_atoms = []
               
        for token in entity:
            if token in list(token_ref.keys()):
                uid = entity+token
                if uid in chain_set_pairs:
                    ref_space_uid.append(chain_set_pairs.index(uid))
                else:
                    chain_set_pairs.append(uid)
                    ref_space_uid.append(len(chain_set_pairs))

                if token_ref[token]['mol_type'] == 'protein':
                    is_ligand.append(0)
                    is_protein.append(1)
                    is_rna.append(0)
                    is_dna.append(0)

                    res_types.append(token_ref[token]['token_idx'])
                    entity_tokens.append(token)
                    atoms, atom_name_char, mask, elements, ref_idx, atoms_pos = generate_atoms(token, padding_size, N_max_atoms, ccd_manager)
                    entity_atoms.append(atoms)
                    ref_atom_name_chars.append(atom_name_char)
                    ref_mask.append(mask)
                    ref_elements.append(elements)
                    ref_atom_idx.append(ref_idx)
                    ref_pos.append(atoms_pos)

                elif token_ref[token]['mol_type'] == 'dna':
                    is_ligand.append(0)
                    is_protein.append(0)
                    is_rna.append(0)
                    is_dna.append(1)

                    res_types.append(token_ref[token]['token_idx'])
                    entity_tokens.append(token)
                    atoms, atom_name_char, mask, elements, ref_idx, atoms_pos = generate_atoms(token, padding_size, N_max_atoms, ccd_manager)
                    entity_atoms.append(atoms)
                    ref_atom_name_chars.append(atom_name_char)
                    ref_mask.append(mask)
                    ref_elements.append(elements)
                    ref_atom_idx.append(ref_idx)
                    ref_pos.append(atoms_pos)


                elif token_ref[token]['mol_type'] == 'rna':
                    is_ligand.append(0)
                    is_protein.append(0)
                    is_rna.append(0)
                    is_dna.append(1)

                    res_types.append(token_ref[token]['token_idx'])
                    entity_tokens.append(token)
                    atoms, atom_name_char, mask, elements, ref_idx, atoms_pos = generate_atoms(token, padding_size, N_max_atoms, ccd_manager)
                    entity_atoms.append(atoms)
                    ref_atom_name_chars.append(atom_name_char)
                    ref_mask.append(mask)
                    ref_elements.append(elements)
                    ref_atom_idx.append(ref_idx)
                    ref_pos.append(atoms_pos)

                if entity_tokens in total_tokens:
                    idx = total_atoms.index(entity_tokens)
                    sym_idx = total_tokens.count(entity_tokens)
                else:
                    idx = asym_count + 1
                    asym_count += 1    
                    sym_idx = 0

                asym_id.append(*[idx for i in range(len(entity_tokens))])
                sym_id.append(*[sym_idx for i in range(len(entity_tokens))])


        else:
            is_ligand.append(1)
            is_protein.append(0)
            is_rna.append(0)
            is_dna.append(0)


            #NOTE : Need to fix this to get atom generation.
            atoms_name, atoms_pos, ref_atoms_chars, elements, bonds  = generate_ligand_atoms(entity, padding_size, N_max_atoms, ccd_manager)

            for j,atom in enumerate(atoms_name):
                res_types.append(0)
                entity_tokens.append("UNK")
                entity_atoms.append(atom)                
                entity_ids.append(i)    


                uid = entity+token
                if uid in chain_set_pairs:
                    ref_space_uid.append(chain_set_pairs.index(uid))
                else:
                    chain_set_pairs.append(uid)
                    ref_space_uid.append(len(chain_set_pairs))


                if entity_tokens in total_tokens:
                    idx = total_atoms.index(entity_tokens)
                    sym_idx = total_tokens.count(entity_tokens)
                else:
                    idx = asym_count + 1
                    asym_count += 1    
                    sym_idx = 0

                asym_id.append(*[idx for i in range(len(entity_tokens))])
                sym_id.append(*[sym_idx for i in range(len(entity_tokens))])
        



