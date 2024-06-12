import requests
import json
import pandas as pd
from Utils.pdb_dataset_curation import run

class generater:
    def __init__(self, master_csv='data/pdb_data_no_dups.csv', ccd_dir='data/CCD', mmcif_inp_path='data/mmCIF_inp/', mmcif_out_path='data/mmCIF/'):
        self.master_df = pd.read_csv(master_csv)
        self.mmCIF_inp_path = mmcif_inp_path
        self.mmCIF_out_path = mmcif_out_path
        self.ccd_dir = ccd_dir

    def filter_mmcif(self):
        pass

    def get_mmcifs(self, nsamples=100):
        samples = self.master_df['structureId'][:nsamples]
        for sample in samples:
            r = requests.get(f'https://files.rcsb.org/view/{sample}.cif')
            with open(self.mmCIF_inp_path+sample+'.cif', 'w+') as f:
                f.write(r.text)

    def clean_mmcifs(self, num_workers, chunksize, skip_existing=True):
        run(self.mmCIF_out_path, num_workers, chunksize, skip_existing, self.ccd_dir, mmcif_dir=self.mmCIF_inp_path)

if __name__ == '__main__':
    g = generater()
    # g.get_mmcifs(1000)
    g.clean_mmcifs(3, 100)