import torch
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import json

     

Token_dict = {
    "UNK" : 0,
    "ALA" : 1,
    "ARG" : 2,
    "ASN" : 3,
    "ASP" : 4,
    "CYS" : 5,
    "GLU" : 6,
    "GLN" : 7,
    "GLY" : 8,
    "HIS" : 9,
    "ILE" : 10,
    "LEU" : 11,
    "LYS" : 12,
    "MET" : 13,
    "PHE" : 14,
    "PRO" : 15,
    "SER" : 16,
    "THR" : 17,
    "TRP" : 18,
    "TYR" : 19,
    "VAL" : 20,
    "DA" : 21,
    "DC" : 22,
    "DG" : 23,
    "DT" : 24,
    "A" : 25,
    "C" : 26,
    "G" : 27,
    "U" : 28
}

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

Chain_dict = {
    "A" : 0,
    "B" : 1,
    "C" : 2,
    "D" : 3,
    "E" : 4,
    "F" : 5,
    "G" : 6,
    "H" : 7,
    "I" : 8
}

proteins_tokens = list(Token_dict.keys())[1:21]
DNA_tokens = list(Token_dict.keys())[21:25]
RNA_tokens = list(Token_dict.keys())[25:]

def Parse_Cif(filelocation):
    x = MMCIF2Dict(filelocation)

    N_tokens, N_atoms, token_index, residue_index, asym_id, entity_id, syn_id, res_type, is_protein, is_dna, is_rna, is_ligand, ref_pos, ref_mask, ref_element, ref_charge, ref_atom_name_chars, ref_space_uid,token_idx_for_atoms, refernce_token_atoms, b_factor, occ= get_atomic_data(x)

    token_bonds = get_connected_comps(x, N_tokens)

    residue_index = torch.tensor(residue_index)
    asym_id = torch.tensor(asym_id)
    entity_id = torch.tensor(entity_id)
    sym_id = torch.tensor(syn_id)
    restype = torch.tensor([Token_dict[a] for a in res_type])
    is_protein = torch.tensor(is_protein)
    is_dna = torch.tensor(is_dna)
    is_rna = torch.tensor(is_rna)
    is_ligand = torch.tensor(is_ligand)

    ref_pos = torch.tensor(ref_pos)
    ref_mask = torch.tensor(ref_mask)
    ref_element = torch.tensor([Element_dict[a] for a in ref_element])  
    ref_charge = torch.tensor(ref_charge)
    ref_atom_name_chars = torch.tensor(ref_atom_name_chars)
    ref_space_uid = torch.tensor(ref_space_uid)

    reference_tokens = torch.tensor(refernce_token_atoms)

    token_idx_for_atoms = torch.tensor(token_idx_for_atoms)

    # f = {
    #     "residue_index" : residue_index,
    #     "token_index" : token_index,
    #     "asym_id" : asym_id,
    #     "entity_id" : entity_id,
    #     "sym_id" : sym_id,
    #     "restype" : restype,
    #     "is_protein" : is_protein,
    #     "is_dna" : is_dna,
    #     "is_rna" : is_rna,
    #     "is_ligand" : is_ligand,
    #     "ref_pos" : ref_pos,
    #     "ref_mask" : ref_mask,
    #     "ref_element" : ref_element,
    #     "ref_charge" : ref_charge,
    #     "ref_atom_name_char" : ref_atom_name_chars,
    #     "token_bonds" : token_bonds,
    #     "beta_factors" : b_factor,
    #     "occupancy" : occ,
    #     "token_idx_atom" : token_idx_for_atoms
    # }

    return residue_index, token_index, asym_id, entity_id, sym_id, restype, is_protein, is_dna, is_rna, is_ligand, ref_pos, ref_mask, ref_element, ref_charge, ref_atom_name_chars, ref_space_uid, token_bonds, token_idx_for_atoms, reference_tokens, b_factor, occ


def get_connected_comps(x, ntokens):
    conn_ptnr_1_comp = x['_struct_conn.ptnr1_label_seq_id']             
    conn_ptnr_2_comp = x['_struct_conn.ptnr2_label_seq_id']             

    token_conns = list(zip([conn_ptnr_1_comp, conn_ptnr_2_comp]))

    con_matrix = torch.zeros((ntokens, ntokens))
    for con in token_conns:
        con_matrix[int(con[0][0]), int(con[0][1])] = 1
        con_matrix[int(con[0][1]), int(con[0][0])] = 1

    return con_matrix

def get_atomic_data(x):
    keys = list(x.keys())

    residue_index =[]
    token_index = []
    asym_id = []
    syn_id = []
    entity_id = []
    res_type = []

    is_protein = []
    is_dna = []
    is_rna = []
    is_ligand = []

    ref_pos = []
    ref_mask = []
    ref_element = []
    ref_charge = []
    ref_atom_name_chars = []
    ref_space_uid = []

    b_factor = []
    occ = []
    token_idx_for_atoms = []
    refernce_token_atoms = []

    uid_list = {}

    x_pos = x['_atom_site.Cartn_x']
    y_pos = x['_atom_site.Cartn_y']
    z_pos = x['_atom_site.Cartn_z']
    occupancy = x['_atom_site.occupancy']
    beta_temp = x['_atom_site.B_iso_or_equiv']

    atom_id = x['_atom_site.type_symbol']
    alt_atom_id = x['_atom_site.label_atom_id']
    entity = x['_atom_site.label_entity_id']
    asym = x['_atom_site.label_asym_id']
    seq_id = x['_atom_site.label_seq_id']
    comp_id = x['_atom_site.label_comp_id']
    charge_id = x['_atom_site.pdbx_formal_charge']

    if '_pdbx_entity_src_syn' in keys:
        syn_entity_id = x['_pdbx_entity_src_syn.entity_id']
        syn_pdbx_src_id = x['_pdbx_entity_src_syn.pdbx_src_id']
    else:
        syn_entity_id = []
        syn_pdbx_src_id = []
    

    token_offset = 0
    ligand_offset = 0
    token_offset -= int(seq_id[0]) #N-termini might not be determined and so are skipped
    skip_res_id = None
    last_chain = None

    for i, res in enumerate(comp_id):
        if seq_id[i] != '.':
            tok_idx = int(seq_id[i]) + token_offset + ligand_offset
        elif seq_id[i] == '.':
            tok_idx = token_offset + ligand_offset

        if entity[i] != last_chain:
            last_chain == entity[i]
            ligand_offset = 0
            if seq_id[i] != '.':
                token_offset -= int(seq_id[i])
            else:
                token_offset = 0    

        # atom encodings
        if atom_id[i] != "H":
            ref_pos.append([float(x_pos[i]), float(y_pos[i]), float(z_pos[i])])
            ref_mask.append(1) #No clue what to and not to include????
            ref_element.append(atom_id[i])
            b_factor.append(float(beta_temp[i]))
            occ.append(float(occupancy[i]))
            token_idx_for_atoms.append(tok_idx)

            if charge_id[i] == '?':
                ref_charge.append(0)
            else:
                ref_charge.append(int(charge_id[i]))

            ord_list = []
            for j, char in enumerate(alt_atom_id[i]):
                ord_list.append(ord(char)-32)
            
            for k in range(3-j):
                ord_list.append(0)
            
            if entity[i] + comp_id[i] in list(uid_list.keys()):
                ref_space_uid.append(uid_list[entity[i] + comp_id[i]])
            else:
                uid_list[entity[i] + comp_id[i]] = len(uid_list.keys())
                ref_space_uid.append(uid_list[entity[i] + comp_id[i]])


            ref_atom_name_chars.append(ord_list)  

            if (alt_atom_id[i] == "CB" and comp_id[i] in ["ALA" ,"ARG" ,"ASN" ,"ASP" ,"CYS" , "GLU" , "GLN" ,"HIS" ,"ILE" ,"LEU" ,"LYS" ,"MET" ,"PHE" ,"PRO" ,"SER" ,"THR" ,"TRP" ,"TYR" ,"VAL"]) or (alt_atom_id[i] == "CB" and comp_id[i] == 'GLY' ) or (alt_atom_id[i] == "C2" and  comp_id[i] in ["DT", "DC", "C", "U"]) or (alt_atom_id[i] == "C2" and  comp_id[i] in ["DT", "DC", "C", "U"]) or (comp_id[i] not in Token_dict):
                refernce_token_atoms.append(1)
            else:
                refernce_token_atoms.append(0)
                
        #token encodings
        if skip_res_id != seq_id[i]:
            if res in proteins_tokens:
                is_protein.append(1)
                is_dna.append(0)
                is_rna.append(0)
                is_ligand.append(0)
                
                if seq_id[i] == '.':
                    residue_index.append(0)
                else:
                    residue_index.append(tok_idx)

                entity_id.append(int(entity[i]))
                asym_id.append(Chain_dict[asym[i]])
                res_type.append(comp_id[i])

                if entity[i] in syn_entity_id:
                    idx = syn_entity_id.index(entity[i])
                    syn_id.append(int(syn_pdbx_src_id[idx]))
                else:
                    syn_id.append(0)


                skip_res_id = seq_id[i]

            elif res in DNA_tokens:
                is_protein.append(0)
                is_dna.append(1)
                is_rna.append(0)
                is_ligand.append(0)

                residue_index.append(tok_idx)
                entity_id.append(int(entity[i]))
                asym_id.append(Chain_dict[asym[i]])
                res_type.append(comp_id[i])

                if entity[i] in syn_entity_id:
                    idx = syn_entity_id.index(entity[i])
                    syn_id.append(int(syn_pdbx_src_id[idx]))
                else:
                    syn_id.append(0)

                skip_res_id = seq_id[i]

            elif res in RNA_tokens:
                is_protein.append(0)
                is_dna.append(0)
                is_rna.append(1)
                is_ligand.append(0)

                residue_index.append(tok_idx)
                entity_id.append(int(entity[i]))
                asym_id.append(Chain_dict[asym[i]])
                res_type.append(comp_id[i])

                if entity[i] in syn_entity_id:
                    idx = syn_entity_id.index(entity[i])
                    syn_id.append(int(syn_pdbx_src_id[idx]))
                else:
                    syn_id.append(0)

                skip_res_id = seq_id[i]

            elif res not in ["H2O"]:
                if atom_id != "H":
                    is_protein.append(0)
                    is_dna.append(0)
                    is_rna.append(0)
                    is_ligand.append(1)

                    residue_index.append(tok_idx)
                    entity_id.append(int(entity[i]))
                    asym_id.append(Chain_dict[asym[i]])
                    res_type.append("UNK")
                
                    if entity[i] in syn_entity_id:
                        idx = syn_entity_id.index(entity[i])
                        syn_id.append(int(syn_pdbx_src_id[i]))
                    else:
                        syn_id.append(0)

                    ligand_offset += 1

    N_tokens = len(residue_index)
    N_atoms = len(ref_mask)
    token_index = torch.arange(0, N_tokens, 1)
    return N_tokens, N_atoms, token_index, residue_index, asym_id, entity_id, syn_id, res_type, is_protein, is_dna, is_rna, is_ligand, ref_pos, ref_mask, ref_element, ref_charge, ref_atom_name_chars, ref_space_uid, token_idx_for_atoms, refernce_token_atoms, b_factor, occ, 


def test():
    x = Parse_Cif('data/1sld.cif')
    print(x)

if __name__ == "__main__":
    test()