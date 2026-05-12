import torch
import numpy as np
import argparse
from Gslib import Gslib
from FMclasses import ForwardModeling, ElasticModels
import shutil
import subprocess
import os
import torchvision


def write_zones(args, filename, zone):
    with open(args.project_path + args.in_folder + filename, 'w') as ffid:
        ffid.write(filename + '\n')
        ffid.write('4\n')
        ffid.write('x\n')
        ffid.write('y\n')
        ffid.write('z\n')
        ffid.write('Ip\n')
        for lin in range(len(zone)):
            ffid.write(' '.join(zone.loc[lin].astype(str)) + '\n')


def precompute_Seismic(args, Ip_models, FM):
    print('Pre-computing TI Seismic')
    if args.ip_type != 0:
        if os.path.isdir(args.project_path + args.TI_path + '/Seismic_TI'):
            shutil.rmtree(args.project_path + args.TI_path + '/Seismic_TI')

        if os.path.isdir(args.project_path + args.TI_path + '/Facies_TI'):
            shutil.rmtree(args.project_path + args.TI_path + '/Facies_TI')

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomInvert(1),
        torchvision.transforms.RandomRotation([180, 180])
    ])

    dataset = torchvision.datasets.ImageFolder(root=args.project_path + args.TI_path, transform=transforms)
    N = len(dataset)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=N)

    os.mkdir(args.project_path + args.TI_path + '/Seismic_TI')
    os.mkdir(args.project_path + args.TI_path + '/Facies_TI')

    for _, data in enumerate(dataloader, 0):
        data = data[0][:, 0, None, :, :]
        Ip_models.writeallfac_dss(data.detach().cpu().numpy())
        Ip_models.simulations = torch.zeros((args.nsim, 1, data.shape[2], data.shape[3]))

        for i in range(N):
            Ip_models.write_parfile(i, 'unc', args.nsim)
            subprocess.run(args=[f'{Ip_models.inf}DSS.C.64.exe', f'{Ip_models.inf}ssdir.par'], stdout=subprocess.DEVNULL)

            for ssi in range(args.nsim):
                Ip_data = Gslib().Gslib_read(f'{Ip_models.ouf}/dss/ip_real_{ssi + 1}.out').data.values.squeeze()
                Ip_data = Ip_data.reshape((1, args.nz, args.nx))
                Ip_models.simulations[ssi] = torch.from_numpy(Ip_data)

            syn_TI = FM.calc_synthetic(Ip_models.simulations.detach()).detach()

            for ssi in range(args.nsim):
                torch.save(syn_TI[ssi], args.project_path + args.TI_path + f'/Seismic_TI/{i}.pt')

            dtw = torch.from_numpy(data[i].detach().cpu().numpy())
            torch.save(dtw, args.project_path + args.TI_path + f'/Facies_TI/{i}.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_path", default='E:/DualGAN_related/Seismic_inversion/code/BicycleGAN_Seismic_Inversion_2/', type=str,
                        help="working directory")
    parser.add_argument("--in_folder", default='Input_synthetic_case/', type=str, help="input folder")
    parser.add_argument("--out_folder", default='Out_syn_model1', type=str, help='output folder')
    parser.add_argument("--TI_path", default='Dataset_syn/', type=str, help="Training Images folder")
    parser.add_argument("--well_data", default='well_data', type=str, help="Well data filename")
    parser.add_argument("--real_seismic_file", default='/Real_seismic_DSS1.out', type=str, help="seismic file name")
    parser.add_argument("--wavelet_file", default='wavelet_simm.asc', type=str, help="Name of the W	avelet file")
    parser.add_argument("--W_weight", default=1 / 300, type=float,
                        help="Factor of scaling for wavelet (negative for normal polarity (python indexing))")
    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size, batch subset fraction (for big datasets)")
    parser.add_argument("--epochs", default=501, type=int, help="N of Epochs")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for the Adam optimizer")
    parser.add_argument("--decayRate", default=0.5, type=float, help="learning rate decay")
    parser.add_argument("--lr_steps", default=20, type=float, help="learning rate decay")
    parser.add_argument("--alpha_c", default=0, type=float, help='Content loss weight (None if there is no well)')
    parser.add_argument("--precompute_TI_seismic", default=True, type=bool, help='if True, computes nsim Seismic realizations from TI facies, to use as training data. \
                                If False, it computes seismic at each epoch from a new DSS realization')
    parser.add_argument("--precomputed", default=False, type=bool,
                        help='if True, Seismic_TI and Facies_TI folders contain already the Training Data')
    parser.add_argument("--nsim", default=1, type=int,
                        help='If alpha_s is not None, calculates seismic from average of nsim simulations for seismic misfit')
    parser.add_argument("--nx", default=100, type=int, help='x size of inversion grid')
    parser.add_argument("--ny", default=1, type=int, help='y size of inversion grid')
    parser.add_argument("--nz", default=80, type=int, help='z size of inversion grid')
    parser.add_argument("--n_facies", default=2, type=int, help='Number of facies in the model')
    parser.add_argument("--var_N_str", default=[1, 1], type=int,
                        help='number of variogram structures per facies [fac0, fac1,...]')
    parser.add_argument("--var_nugget", default=[0, 0], type=float, help='variogram nugget per facies [fac0, fac1,...]')
    parser.add_argument("--var_type", default=[[1], [1]], type=int,
                        help='variogram type per facies [fac0[str1,str2,...], fac1[str1,str2,...],...]: 1=spherical,2=exponential,3=gaussian')
    parser.add_argument("--var_ang", default=[[0, 0], [0, 0]], type=float,
                        help='variogram angles per facies [fac0[angX,angZ], fac1[angX,angZ],...]')
    parser.add_argument("--var_range", default=[[[80, 30]], [[80, 50]]], type=float,
                        help='variogram ranges per structure and per facies [fac0[str0[rangeX,rangeZ],str1[rangeX,rangeZ]...],fac1[str1[rangeX,rangeZ],...],...]')
    parser.add_argument("--null_val", default=-9999.00, type=float, help='null value in well data')
    parser.add_argument("--ip_type", default=1)
    parser.add_argument("--type_of_FM", default='fullstack', type=str)

    if torch.cuda.is_available():
        parser.add_argument("--device", default='cuda')
    else:
        parser.add_argument("--device", default='cpu')

    args = parser.parse_args()

    well_data = Gslib().Gslib_read(args.project_path + args.in_folder + args.well_data).data
    well_data = well_data[well_data != args.null_val].dropna()
    bounds_zones = {}
    y_idx = 29

    if y_idx + 1 in well_data.j_index.values:
        well_data = well_data[well_data.j_index == y_idx + 1]
        well_data.j_index = 1
        typ = 'cond'
    else:
        typ = 'unc'

    for fac in range(args.n_facies):
        zone = well_data[['i_index', 'j_index', 'k_index', 'Ip']][well_data.Facies == fac].reset_index(drop=True)
        filename = 'Ip_zone%d_cond.out' % fac
        write_zones(args, filename, zone)
        bounds_zones[fac] = np.array([zone.Ip.min(), zone.Ip.max()])

        zone.loc[:, 'j_index'] = 1000
        filename = 'Ip_zone%d_unc.out' % fac
        write_zones(args, filename, zone)

    Ip_models = ElasticModels(args, ipmax=well_data.Ip.max(), ipmin=well_data.Ip.min(), ipzones=bounds_zones)

    FM = ForwardModeling(args)
    FM.load_wavelet(args)

    precompute_Seismic(args=args, Ip_models=Ip_models, FM=FM)