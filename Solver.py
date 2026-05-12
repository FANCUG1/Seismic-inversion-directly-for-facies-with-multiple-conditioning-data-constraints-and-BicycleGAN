import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
import matplotlib.pyplot as plt
from Custom_dataloader import FaciesSeismicDataset
from GAN_module import Generator, Discriminator, Encoder
import util
from Gslib import Gslib
import os
from torch.utils.data import DataLoader


def mse_loss(score, target=1):
    dtype = type(score)

    if target == 1:
        label = util.var(torch.ones(score.size()),requires_grad=False)
    elif target == 0:
        label = util.var(torch.zeros(score.size()), requires_grad=False)

    criterion = nn.MSELoss()
    loss = criterion(score, label)

    return loss


def L1_loss(pred, target):
    return torch.mean(torch.abs(pred - target))


def lr_decay_rule(epoch, start_decay=100, lr_decay=100):
    decay_rate = 1.0 - (max(0, epoch - start_decay) / float(lr_decay))
    return decay_rate


class Solver():
    def __init__(self, root='./Dataset_syn/', result_dir='result', weight_dir='weight', load_weight=False,
                 batch_size=2, test_size=20, test_img_num=5, img_size=(80, 100), num_epoch=100, save_every=100, lr=0.0002,
                 beta_1=0.5, beta_2=0.999, lambda_kl=0.01, lambda_img=10, lambda_z=0.5, con_weight=0.5, z_dim=8,
                 num_con_data=20, nsim=16, nx=100, ny=1, nz=80):

        # Data type(Can use GPU or not?)
        self.dtype = torch.cuda.FloatTensor
        if torch.cuda.is_available() is False:
            self.dtype = torch.FloatTensor

        self.dataset = FaciesSeismicDataset(root_dir=root, nsim=nsim, num_con_data=num_con_data)
        self.dloader = DataLoader(dataset=self.dataset, batch_size=batch_size, shuffle=True)

        self.D_cVAE = Discriminator(ndim=1).type(self.dtype)
        self.D_cLR = Discriminator(ndim=1).type(self.dtype)
        self.G = Generator(z_dim=z_dim).type(self.dtype)
        self.E = Encoder(z_dim=z_dim).type(self.dtype)

        self.optim_D_cVAE = optim.Adam(self.D_cVAE.parameters(), lr=lr, betas=(beta_1, beta_2))
        self.optim_D_cLR = optim.Adam(self.D_cLR.parameters(), lr=lr, betas=(beta_1, beta_2))
        self.optim_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta_1, beta_2))
        self.optim_E = optim.Adam(self.E.parameters(), lr=lr, betas=(beta_1, beta_2))

        self.fixed_z = util.var(torch.randn(test_size, test_img_num, z_dim))

        self.z_dim = z_dim
        self.lambda_kl = lambda_kl
        self.lambda_img = lambda_img
        self.lambda_z = lambda_z
        self.con_weight = con_weight

        self.result_dir = result_dir
        self.weight_dir = weight_dir
        self.load_weight = load_weight
        self.test_img_num = test_img_num
        self.img_size = img_size
        self.start_epoch = 0
        self.num_epoch = num_epoch
        self.save_every = save_every
        self.num_con_data = num_con_data
        self.nsim = nsim
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.batch_size = batch_size
        self.real_seismic = torch.zeros((batch_size, 1, self.nz, self.nx)).type(self.dtype).detach()
        self.max_abs_amp = 0

    def get_real_seismic(self):
        seismic = Gslib().Gslib_read(filename='./Input_synthetic_case/Real_seismic_DSS1.out').data
        seismic = np.reshape(seismic.values, (self.nx, self.ny, self.nz), order='F')
        seismic = seismic.squeeze()

        self.real_seismic[:, :] = torch.tensor(seismic.T)

    def set_train_phase(self):
        self.D_cVAE.train()
        self.D_cLR.train()
        self.G.train()
        self.E.train()

    def save_weight(self, epoch=None):
        if epoch is None:
            d_cVAE_name = 'D_cVAE.pth'
            d_cLR_name = 'D_cLR.pth'
            g_name = 'G.pth'
            e_name = 'E.pth'
        else:
            d_cVAE_name = '{epochs}-{name}'.format(epochs=str(epoch), name='D_cVAE.pth')
            d_cLR_name = '{epochs}-{name}'.format(epochs=str(epoch), name='D_cLR.pth')
            g_name = '{epochs}-{name}'.format(epochs=str(epoch), name='G.pth')
            e_name = '{epochs}-{name}'.format(epochs=str(epoch), name='E.pth')

        torch.save(self.D_cVAE.state_dict(), os.path.join(self.weight_dir, d_cVAE_name))
        torch.save(self.D_cVAE.state_dict(), os.path.join(self.weight_dir, d_cLR_name))
        torch.save(self.G.state_dict(), os.path.join(self.weight_dir, g_name))
        torch.save(self.E.state_dict(), os.path.join(self.weight_dir, e_name))

    def all_zero_grad(self):
        self.optim_D_cVAE.zero_grad()
        self.optim_D_cLR.zero_grad()
        self.optim_G.zero_grad()
        self.optim_E.zero_grad()

    def cal_con_loss(self, fake, con_data, ground_truth):
        bs = ground_truth.shape[0]
        y, x = ground_truth.shape[2], ground_truth.shape[3]

        binary_mask = []
        con_data_ = con_data.detach().cpu().numpy()

        for i in range(bs):
            temp = con_data_[i].reshape((y, x))
            temp2 = np.zeros((y, x))

            for j in range(y):
                for k in range(x):
                    if temp[j, k] == -1 or temp[j, k] == 1:
                        temp2[j, k] = 1

            temp2 = temp2.reshape((1, y, x))
            binary_mask.append(temp2)

        binary_mask = np.array(binary_mask)
        binary_mask = torch.from_numpy(binary_mask).type(torch.cuda.FloatTensor)

        con_loss = self.con_weight * torch.mean(torch.abs(binary_mask * (ground_truth - fake)))
        return con_loss

    def save_inversion_image(self, result_img, img_path):
        bs = result_img.shape[0]
        result_img_ = result_img.detach().cpu().numpy()
        con_data = result_img_[0].reshape((self.nz, self.nx))
        seismic = result_img_[1].reshape((self.nz, self.nx))
        ground_truth = result_img_[2].reshape((self.nz, self.nx))

        plt.figure(figsize=(10, 8))
        plt.subplot(2, bs // 2, 1)
        plt.imshow(con_data, cmap='gray')
        plt.subplot(2, bs // 2, 2)
        plt.imshow(seismic, cmap='seismic')
        plt.subplot(2, bs // 2, 3)
        plt.imshow(ground_truth, cmap='gray')

        for i in range(3, bs, 1):
            inversion_results = result_img_[i].reshape((self.nz, self.nx))
            plt.subplot(2, bs // 2, i + 1)
            plt.imshow(inversion_results, cmap='gray')

        plt.savefig(img_path, dpi=300)

    def train(self):
        self.set_train_phase()

        self.get_real_seismic()

        for epoch in range(self.start_epoch, self.num_epoch):

            for iters, (con_data, seismic, ground_truth) in enumerate(self.dloader, 0):
                bs = ground_truth.size()[0]
                con_data, ground_truth, seismic = util.var(con_data), util.var(ground_truth), util.var(seismic)

                cVAE_data = {'con_data': con_data, 'seismic': seismic, 'ground_truth': ground_truth}
                # 这里别忘了把下标改回1
                cLR_data = {'con_data': con_data, 'seismic': seismic, 'ground_truth': ground_truth}


                mu, log_variance = self.E(cVAE_data['ground_truth'])
                std = torch.exp(log_variance / 2)
                random_z = util.var(torch.randn(bs, self.z_dim))
                encoded_z = (random_z * std) + mu

                fake_img_cVAE = self.G(cVAE_data['con_data'], cVAE_data['seismic'], encoded_z)

                real_d_cVAE_1, real_d_cVAE_2 = self.D_cVAE(cVAE_data['ground_truth'])
                fake_d_cVAE_1, fake_d_cVAE_2 = self.D_cVAE(fake_img_cVAE)

                D_loss_cVAE_1 = mse_loss(real_d_cVAE_1, 1) + mse_loss(fake_d_cVAE_1, 0)
                D_loss_cVAE_2 = mse_loss(real_d_cVAE_2, 1) + mse_loss(fake_d_cVAE_2, 0)

                random_z = util.var(torch.randn(bs, self.z_dim))

                fake_img_cLR = self.G(cLR_data['con_data'], cLR_data['seismic'], random_z)

                real_d_cLR_1, real_d_cLR_2 = self.D_cLR(cLR_data['ground_truth'])
                fake_d_cLR_1, fake_d_cLR_2 = self.D_cLR(fake_img_cLR)

                D_loss_cLR_1 = mse_loss(real_d_cLR_1, 1) + mse_loss(fake_d_cLR_1, 0)
                D_loss_cLR_2 = mse_loss(real_d_cLR_2, 1) + mse_loss(fake_d_cLR_2, 0)

                D_loss = D_loss_cVAE_1 + D_loss_cLR_1 + D_loss_cVAE_2 + D_loss_cLR_2

                self.all_zero_grad()
                D_loss.backward(retain_graph=True)
                self.optim_D_cVAE.step()
                self.optim_D_cLR.step()

                mu, log_variance = self.E(cVAE_data['ground_truth'])
                std = torch.exp(log_variance / 2)
                random_z = util.var(torch.randn(bs, self.z_dim))
                encoded_z = (random_z * std) + mu

                fake_img_cVAE = self.G(cVAE_data['con_data'], cVAE_data['seismic'], encoded_z)
                fake_d_cVAE_1, fake_d_cVAE_2 = self.D_cVAE(fake_img_cVAE)

                GAN_loss_cVAE_1 = mse_loss(fake_d_cVAE_1, 1)
                GAN_loss_cVAE_2 = mse_loss(fake_d_cVAE_2, 1)

                random_z = util.var(torch.randn(bs, self.z_dim))

                fake_img_cLR = self.G(cLR_data['con_data'], cLR_data['seismic'], random_z)
                fake_d_cLR_1, fake_d_cLR_2 = self.D_cLR(fake_img_cLR)

                GAN_loss_cLR_1 = mse_loss(fake_d_cLR_1, 1)
                GAN_loss_cLR_2 = mse_loss(fake_d_cLR_2, 1)

                G_GAN_loss = GAN_loss_cVAE_1 + GAN_loss_cVAE_2 + GAN_loss_cLR_1 + GAN_loss_cLR_2

                KL_div = self.lambda_kl * torch.sum(0.5 * (mu ** 2 + torch.exp(log_variance) - log_variance - 1))

                img_recon_loss = self.lambda_img * L1_loss(fake_img_cVAE, cVAE_data['ground_truth'])

                EG_loss = G_GAN_loss + KL_div + img_recon_loss

                self.all_zero_grad()
                EG_loss.backward(retain_graph=True)
                self.optim_E.step()
                self.optim_G.step()

                random_z = util.var(torch.randn(bs, self.z_dim))
                fake_img_cLR = self.G(cLR_data['con_data'], cLR_data['seismic'], random_z)

                mu_, log_variance_ = self.E(fake_img_cLR)
                z_recon_loss = L1_loss(mu_, random_z)

                G_alone_loss = self.lambda_z * z_recon_loss

                self.all_zero_grad()
                G_alone_loss.backward(retain_graph=True)
                self.optim_G.step()

                if (iters + 1) % self.save_every == 0 or (iters + 1) == len(self.dloader):
                    print(
                        '[Epoch : %d / Iters : %d] => D_loss : %f / G_GAN_loss : %f / KL_div : %f / img_recon_loss : %f / z_recon_loss : %f /' \
                        % (epoch, iters + 1, D_loss.item(), G_GAN_loss.item(), KL_div.item(), img_recon_loss.item(),
                           G_alone_loss.item()))

            if os.path.exists(self.result_dir) is False:
                os.makedirs(self.result_dir)

            result_img = util.make_img(self.dloader, self.G, self.fixed_z, img_num=self.test_img_num, img_size=self.img_size)
            img_name = 'fake_image_{epoch}.png'.format(epoch=epoch)
            img_path = os.path.join(self.result_dir, img_name)

            self.save_inversion_image(result_img, img_path)

            if os.path.exists(self.weight_dir) is False:
                os.makedirs(self.weight_dir)

            self.save_weight(epoch=epoch)