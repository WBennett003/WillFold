from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import json


with open('token_ref.json') as rf:
    residue_names = json.load(rf)


# Parse MMCIF with biopython

# Specs in file:///home/p/Downloads/41586_2024_7487_MOESM1_ESM.pdf#subsection.2.8


def parse_polymer(struct_dict, entity_id : int):
    
    ret = []
    for i, code in enumerate(struct_dict['_entity_poly_seq.mon_id']):
        if struct_dict['_entity_poly_seq.entity_id'][i] == str(entity_id):
            ret.append(code)

    return ret

def parse_nonpolymer(struct_dict, entity_id : int):
    entity_index = struct_dict['_pdbx_entity_nonpoly.entity_id'].index(str(entity_id))
    return [struct_dict['_pdbx_entity_nonpoly.comp_id'][entity_index]]

# TODO: trim residues with no data
def parse_CIF(filepath : str):

    struct_dict = MMCIF2Dict(filepath)

    structure = []

    for i in struct_dict['_entity.id']:
        
        i = int(i)

        match (struct_dict['_entity.type'][i - 1]):
            case 'polymer':
                structure.append(parse_polymer(struct_dict, i))
            case 'non-polymer':
                structure.append(parse_nonpolymer(struct_dict, i))
            case 'water':
                continue
            case _:
                raise Exception(f"Unrecognised entity type, {struct_dict['_entity.type'][i - 1]}")

    return structure


def extract_coordinates(filepath : str, N_max_atom : int):
    
    struct_dict = MMCIF2Dict(filepath)

    coordinates = []
    res = []

    curr = struct_dict['_atom_site.label_seq_id'][0]

    for i in range(0, len(struct_dict['_atom_site.id'])):

        row = {
            'label_seq_id' : struct_dict['_atom_site.label_seq_id'][i],
            'label_comp_id' : struct_dict['_atom_site.label_comp_id'][i], 
            'cartn_x' : float(struct_dict['_atom_site.Cartn_x'][i]), 
            'cartn_y' : float(struct_dict['_atom_site.Cartn_y'][i]), 
            'cartn_z' : float(struct_dict['_atom_site.Cartn_z'][i])
        }

        if row['label_seq_id'] != curr or row['label_comp_id'] not in residue_names : 
            # pad res
            coordinates.append(res) # + ([0] * (N_max_atom - len(res)))
            res = []
            curr = row['label_seq_id']

        res.append([row['cartn_x'], row['cartn_y'], row['cartn_z']])

    coordinates.append(res + ([0] * (N_max_atom - len(res))))

    print(int(struct_dict['_atom_site.id'][-1]))

    return coordinates


def test_generic(function, args=None):

    import os

    def print_data(data):
        for d in data:
            print(d)


    for file in os.listdir('data'):
        print('data/' + file)
        if args:
            with open('test_' + file, 'w') as wf:
                wf.write("\n".join(str(d) for d in function('data/' + file, *args)))
        else:
            print_data(function('data/' + file))

def plt_test():

    import matplotlib.pyplot as plt
    import numpy as np
    import os


    def plot(filepath):
        coords = extract_coordinates(filepath, 24)

        tkn_cnt = 0

        to_plt = []

        for tkn in coords:
            tkn_cnt += 1
            for pos in tkn:
                if pos != 0:
                    to_plt.append(pos)

        to_plt = np.array(to_plt)

        print(to_plt)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')        
        ax.scatter(to_plt[:, 0], to_plt[:, 1], to_plt[:, 2])
        plt.show()

    for file in os.listdir('data'):
        plot('data/' + file)


# test_generic(extract_coordinates, [24])
plt_test()