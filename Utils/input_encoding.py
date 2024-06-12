import torch
import torch.nn.functional as F
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

class CCD_manager:
    def __init__(self, reference_filepath='data/CCD/', json_name='reference_conformer.json'):
        self.path = reference_filepath
        self.json_filename = json_name
        with open(json_name, 'r') as f:
            self.ref = json.load(f)
    

    def get_ccd(self, ccd_code):
        if ccd_code in self.ref['CCD']:
            with open(self.path+ccd_code+'.sdf', 'r') as f:
                r = f.read()
            mol = Chem.MolFromMolBlock(r)
            
        else:
            r = requests.get("https://www.ebi.ac.uk/pdbe/static/files/pdbechem_v2/"+ccd_code+"_ideal.sdf").text
            mol = Chem.MolFromMolBlock(r)
            mol = Chem.RemoveAllHs(mol)    
            if ccd_code in list(token_ref.keys()):
                match token_ref[ccd_code]['mol_type']:
                    case "protein":
                        #remove the carbonates hydroxyl
                        natoms = len([a for a in mol.GetAtoms()])
                        mol = Chem.rdchem.EditableMol(mol)
                        mol.RemoveAtom(natoms-1)
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
                w.write(mol)

            self.ref['CCD'].append(ccd_code)

            with open(self.json_filename, 'w') as f:
                json.dump(self.ref, f)     

        #get atoms
        atoms_name = []
        atoms_pos = []
        atoms_charge = []
        for atoms in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atoms.GetIdx())
            atoms_pos.append([pos.x, pos.y, pos.z])
            atoms_name.append(atoms.GetSymbol())
            atoms_charge.append(atoms.GetFormalCharge())

        bonds = []
        for bond in mol.GetBonds():
            ai = bond.GetBeginAtomIdx()
            aj = bond.GetEndAtomIdx()
            bonds.append([ai, aj])

        

        return mol, atoms_name, atoms_pos, bonds, atoms_charge

def generate_atoms(entity, pad_size=4, N_max_atoms=24, parser=None):
    atoms = token_ref[entity]['atoms']
    length = len(atoms)

    mol, atoms_name, atoms_pos, bonds, charge = parser.get_ccd(entity)

    assert [a[0] for a in atoms] == atoms_name

    elements = [Element_dict[a[0]] for a in atoms]

    for i, a in enumerate(atoms):
        atoms[i] = a + ' '* (pad_size-len(a))

    ref_atoms_chars = [[ord(c)-32 for c in a] for a in atoms]

    ref_atom_idx = token_ref[entity]['ref_atom_idx']

    return atoms, ref_atoms_chars, elements, ref_atom_idx, atoms_pos, charge

def generate_ligand_atoms(entity, pad_size=4, N_max_atoms=24, parser=None):
    mol, atoms_name, atoms_pos, bonds, charge = parser.get_ccd(entity)
    length = len(atoms_name)
    
    ref_atoms_chars = []
    elements = []

    for i, atom in enumerate(atoms_name):
        elements.append(atom[0])

        atom  += ' ' * (pad_size - len(atom))
        ref_atoms_chars.append([ord(c)-32 for c in atom])
    
    return atoms_name, atoms_pos, ref_atoms_chars, elements, bonds, charge

def process_entities(entities, implicit_token_bonds=[], padding_size=4, N_max_atoms=24):

    ccd_manager = CCD_manager()

    residue_index = []
    entity_ids = []
    asym_id = []
    sym_id = []

    is_protein = []
    is_rna = []
    is_dna = []
    is_ligand = []


    res_types = []
    ref_atom_name_chars = []
    ref_mask = []
    ref_pos = []
    ref_elements = []
    ref_charge = []
    ref_atom_idx = []
    ref_space_uid = []

    represented_atoms = []
    
    chain_set_pairs = []

    total_tokens = []
    total_atoms = []

    previous_entity = []
    
    token_bonds = implicit_token_bonds

    asym_count = 0
    token_shift = 0
    atom_count = 0
    for i, entity in enumerate(entities):
        entity_tokens = []
        entity_atoms = []
        res_count = 0

        if entity in previous_entity:
            idx = previous_entity.index(entity)
            sym_idx = previous_entity.count(entity)
            previous_entity.append(entity)
        else:
            idx = asym_count
            asym_count += 1    
            sym_idx = 0

        for j, token in enumerate(entity):
            if token in list(token_ref.keys()):
                match token_ref[token]['mol_type']: 
                    case 'protein':
                        is_ligand.append(0)
                        is_protein.append(1)
                        is_rna.append(0)
                        is_dna.append(0)
                    case 'dna':
                        is_ligand.append(0)
                        is_protein.append(0)
                        is_rna.append(0)
                        is_dna.append(1)
                    case 'rna':
                        is_ligand.append(0)
                        is_protein.append()
                        is_rna.append(1)
                        is_dna.append(0)

                residue_index.append(res_count)
                uid = str(i)+token

                res_types.append(token_ref[token]['token_id'])
                entity_tokens.append(token)
                atoms, atom_name_char, elements, ref_idx, atoms_pos, charge = generate_atoms(token, padding_size, N_max_atoms, ccd_manager)
                entity_atoms.append(atoms)
                entity_ids.append(i)    

                ref_atom_name_chars.extend(atom_name_char)
                # ref_mask.append(])
                ref_elements.extend(elements)
                ref_charge.extend(charge)
                ref_pos.extend(atoms_pos)
                ref_atom_idx.extend([len(residue_index)-1 for _ in range(len(elements))])
                represented_atoms.append(atom_count+ref_idx)

                atom_count += len(atoms)


                asym_id.append(idx)
                sym_id.append(sym_idx)
                res_count += 1


                if uid in chain_set_pairs:
                    ref_space_uid.extend([chain_set_pairs.index(uid) for _ in range(len(elements))])
                else:
                    chain_set_pairs.append(uid)
                    ref_space_uid.extend([len(chain_set_pairs) for _ in range(len(elements))])
                    


            else:


                #NOTE : Need to fix this to get atom generation.
                atoms_name, atoms_pos, ref_atoms_chars, elements, bonds, charge  = generate_ligand_atoms(token, padding_size, N_max_atoms, ccd_manager)

                for k,atom in enumerate(atoms_name):
                    is_ligand.append(1)
                    is_protein.append(0)
                    is_rna.append(0)
                    is_dna.append(0)

                    residue_index.append(res_count)
                    res_types.append(0)
                    entity_tokens.append("UNK")
                    entity_atoms.append(atom)                
                    entity_ids.append(i)    
                    ref_pos.append(atoms_pos[k])
                    ref_elements.append(Element_dict[elements[k]])
                    ref_charge.append(charge[k])
                    ref_atom_name_chars.append(ref_atoms_chars[k])
                    ref_atom_idx.append(res_count)

                    represented_atoms.append(atom_count)

                    atom_count += 1
                    res_count +=1

                    asym_id.append(idx)
                    sym_id.append(sym_idx)

                    uid = str(i)+token
                    if uid in chain_set_pairs:
                        ref_space_uid.append(chain_set_pairs.index(uid))
                    else:
                        chain_set_pairs.append(uid)
                        ref_space_uid.append(len(chain_set_pairs))

                for bond in bonds:
                    token_bonds.append([token_shift +j+ bond[0], token_shift +j+ bond[1]])

        token_shift += len(entity)
    
    token_bonds_matrix = torch.zeros((len(residue_index), len(residue_index)))
    for bond in token_bonds:
        token_bonds_matrix[bond[0], bond[1]] = 1
        token_bonds_matrix[bond[1], bond[0]] = 1
    
    token_bonds_matrix
        
    f = {
    "residue_idx" : torch.tensor(residue_index),
    "token_idx" : torch.arange(0, len(residue_index)),
    "asym_id" : torch.tensor(asym_id),
    "entity_id" : torch.tensor(entity_ids),
    "sym_id" : torch.tensor(sym_id),
    "restype" : F.one_hot(torch.tensor(res_types), 32),
    "is_protein" : torch.tensor(is_protein),
    "is_dna" : torch.tensor(is_dna),
    "is_rna" : torch.tensor(is_rna),
    "is_ligand" : torch.tensor(is_ligand),
    "ref_pos" : torch.tensor(ref_pos),
    "ref_mask" : torch.ones(len(ref_pos)),
    "ref_element" : F.one_hot(torch.tensor(ref_elements), 128),
    "ref_charge" : torch.tensor(ref_charge),
    "ref_atom_name_chars" : F.one_hot(torch.tensor(ref_atom_name_chars), 64),
    "ref_space_uid" : torch.tensor(ref_space_uid),
    "token_bonds" : token_bonds_matrix,
    "reference_tokens" : F.one_hot(torch.tensor(ref_atom_idx), len(residue_index)).float(),
    "represenative_atoms" : F.one_hot(torch.tensor(represented_atoms), len(ref_pos)).float()
    }
    return f
