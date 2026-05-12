import argparse
import torch

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_path", default='D:/fwy_project2/BicycleGAN_Seismic_Inversion/', type=str,
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
    parser.add_argument("--nsim", default=16, type=int,
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

    return parser
