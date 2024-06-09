from Bio.PDB.MMCIF2Dict import MMCIF2Dict

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

def test():

    import os
    
    def print_data(data):
        for d in data:
            print(d)


    for file in os.listdir('data'):
        print('data/' + file)
        print_data(parse_CIF('data/' + file))