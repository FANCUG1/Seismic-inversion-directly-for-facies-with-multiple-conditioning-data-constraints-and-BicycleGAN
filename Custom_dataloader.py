import os
import random

import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils
from torch.utils.data import Dataset, DataLoader
import glob


class FaciesSeismicDataset(Dataset):

    def __init__(self, root_dir, nsim, num_con_data, transform=None):
        self.root_dir = root_dir  # D:/python_code/BicycleGAN_Seismic_Inversion/Dataset_syn/
        self.lendata = os.listdir(root_dir + '/Facies_TI')  # TI的名称
        self.transform = transform
        self.nsim = nsim
        self.num_con_data = num_con_data

    def __len__(self):
        return len(self.lendata)

    def select_vertical_data(self, TI, num_wells):
        *_, y, x = TI.shape[1], TI.shape[2]
        TI_ = TI.detach().cpu().numpy()
        TI_ = TI_.reshape((y, x))

        well_data = np.zeros((y, x))

        count = 0
        while count < num_wells:
            loc_x = random.randint(0, x - 1)
            if (well_data[:, loc_x] == 0).all():
                well_data[:, loc_x] = TI_[:, loc_x]
                count += 1

        well_data = well_data.reshape((1, y, x))
        well_data = torch.from_numpy(well_data).type(torch.FloatTensor)

        return well_data

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        fac_filename = self.root_dir + 'Facies_TI/' + f'{idx}.pt'
        seis_filename = self.root_dir + 'Seismic_TI/' + f'{idx}.pt'
        fac_file = torch.load(fac_filename)
        seis_file = torch.load(seis_filename)
        seis_file = seis_file.clamp(-1, 1)

        fac_file = (fac_file - 0.5) / 0.5
        fac_file = fac_file.clamp(-1, 1)

        # 选取条件硬数据
        con_data = self.select_vertical_data(TI=fac_file, num_wells=self.num_con_data)

        return (con_data, seis_file, fac_file)
